from flask import Blueprint, render_template, request, jsonify, current_app
from services.db_utils import q
from db import get_conn
from werkzeug.utils import secure_filename
import time, os
import tempfile
from PIL import Image
import numpy as np
import re
import unicodedata
from utils import encode_texts
import psycopg2.extras as pg_extras
import faiss
from services.resnet101 import load_model, extract_embedding, preprocess_image  # giữ import như cũ
_MODEL = load_model() 
import json
import os
import random

# optional re-ranker wrapper — non-blocking if torch unavailable
try:
    from services.re_ranker import ReRanker
    _RE_RANKER = ReRanker(model_path=os.environ.get("RE_RANKER_MODEL"))
except Exception:
    class _DummyReRanker:
        def score_candidates(self, feat_list):
            out = []
            for f in feat_list:
                try:
                    out.append(float((f or {}).get("similarity") or (f or {}).get("score") or 0.0))
                except Exception:
                    out.append(0.0)
            return out
    _RE_RANKER = _DummyReRanker()

bp = Blueprint("search_test_bp", __name__)

# Use the local vn_norm implementation
def vn_norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 \-\.x/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _q(sql, params=None, fetch="all"):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        conn.commit()

def parse_vector_str(vec_str: str) -> np.ndarray:
    if not vec_str or not isinstance(vec_str, str):
        return None
    # Remove brackets and split by comma
    vec_str = vec_str.strip('[]')
    try:
        # Convert string to list of floats
        vec = [float(x.strip()) for x in vec_str.split(',')]
        return np.array(vec, dtype='float32')
    except (ValueError, AttributeError) as e:
        current_app.logger.warning(f"Failed to parse vector string: {vec_str[:50]}... (error: {e})")
        return None

@bp.get("/search-test")
def search_test_page():
    return render_template("search_test.html")

@bp.get("/search")
def search_page():
    return render_template("search.html")

@bp.post("/search/text")
def search_text():
    data = request.get_json(silent=True) or {}
    raw_q = (data.get("q") or "").strip()
    qtext = vn_norm(raw_q)
    k = min(int(data.get("k") or 20), 100)
    if not qtext:
        return jsonify({"error": "missing q"}), 400

    try:
        t0 = time.time()
        vec = encode_texts([qtext])[0].tolist()

        # log query
        qid = _q(
            "INSERT INTO queries(type, raw_text, normalized_text) VALUES('text', %s, %s) RETURNING id",
            (raw_q, qtext), fetch="one"
        )[0]

        # lấy top-k theo vector
        rows = _q("""
            SELECT st.sku_id, st.id, st.text, (st.text_vec <-> %s::vector) AS dist
            FROM sku_texts st
            WHERE st.text_vec IS NOT NULL
            ORDER BY st.text_vec <-> %s::vector
            LIMIT %s
        """, (vec, vec, k))

        results = []
        for rank, (sku_id, st_id, text, dist) in enumerate(rows, start=1):
            score = 1.0 - float(dist)
            cand_feats = {
                "similarity": float(score),
                "rank": int(rank),
                "source": "text",
                "text_len": int(len(text or "")),
            }
            _q(
                "INSERT INTO query_candidates(query_id, sku_id, rank, score, candidate_features) "
                "VALUES(%s,%s,%s,%s,%s)",
                (qid, sku_id, rank, float(score), pg_extras.Json(cand_feats)), fetch=None
            )
            results.append({
                "sku_id": int(sku_id),
                "sku_text_id": int(st_id),
                "text": text,
                "dist": float(dist),
                "score": score,
                "candidate_features": cand_feats
            })

        # ---------- ATTACH POPULARITY THEO TRUY VẤN (q_*) + fallback pop_* ----------
        sku_ids = [r["sku_id"] for r in results]
        if sku_ids:
            # popularity theo truy vấn
            q_pop_rows = _q("""
                SELECT sku_id,
                       COALESCE(clicks_30d,0)::int,
                       COALESCE(views_30d,0)::int,
                       COALESCE(ctr_30d,0.0)
                FROM query_sku_popularity_30d
                WHERE qn = %s AND sku_id = ANY(%s)
            """, (qtext, sku_ids), fetch="all") or []
            q_pop_map = {r[0]: {"q_clicks_30d": int(r[1]),
                                "q_views_30d":  int(r[2]),
                                "q_ctr_30d":    float(r[3])} for r in q_pop_rows}

            # popularity toàn cục (làm fallback)
            g_pop_rows = _q("""
                SELECT sku_id,
                       COALESCE(clicks_30d,0)::int,
                       COALESCE(views_30d,0)::int,
                       COALESCE(ctr_30d,0.0)
                FROM sku_popularity_30d
                WHERE sku_id = ANY(%s)
            """, (sku_ids,), fetch="all") or []
            g_pop_map = {r[0]: {"pop_clicks_30d": int(r[1]),
                                "pop_views_30d":  int(r[2]),
                                "pop_ctr_30d":    float(r[3])} for r in g_pop_rows}

            # gắn vào features + update lại candidate_features trong DB
            for r in results:
                sku_id = r["sku_id"]
                feats = r["candidate_features"]
                feats.update(g_pop_map.get(sku_id, {"pop_clicks_30d": 0, "pop_views_30d": 0, "pop_ctr_30d": 0.0}))
                feats.update(q_pop_map.get(sku_id, {"q_clicks_30d": 0, "q_views_30d": 0, "q_ctr_30d": 0.0}))
                try:
                    _q("UPDATE query_candidates SET candidate_features=%s WHERE query_id=%s AND sku_id=%s",
                       (json.dumps(feats), qid, sku_id), fetch=None)
                except Exception:
                    pass

        # ---------- RE-RANK: MODEL + PRIOR QUERY-DEPENDENT ----------
        before = results[:]
        try:
            try:
                _RE_RANKER.reload_if_needed()
            except Exception:
                pass

            feat_list = [r.get("candidate_features") for r in results]
            rerank_scores = _RE_RANKER.score_candidates(feat_list) or []

            # helper cho prior theo query
            import math
            def _prior_from_query(f):
                # q_ctr ∈ [0..1], q_clicks dùng log1p để nén
                q_ctr = float((f or {}).get("q_ctr_30d", 0.0) or 0.0)
                q_clk = float((f or {}).get("q_clicks_30d", 0.0) or 0.0)
                q_clk_n = min(1.0, math.log1p(q_clk) / math.log(101.0))
                prior_q = 0.7 * q_ctr + 0.3 * q_clk_n
                # nếu prior_q=0, fallback nhẹ global prior
                if prior_q <= 0.0:
                    pop_ctr = float((f or {}).get("pop_ctr_30d", 0.0) or 0.0)
                    pop_clk = float((f or {}).get("pop_clicks_30d", 0.0) or 0.0)
                    pop_clk_n = min(1.0, math.log1p(pop_clk) / math.log(101.0))
                    prior_q = 0.2 * (0.5 * pop_ctr + 0.5 * pop_clk_n)
                return max(0.0, min(prior_q, 1.0))

            def _model_norm(s):
                # nén về [0..1] để cộng có ý nghĩa
                return 0.5 * (math.tanh(float(s)) + 1.0)

            PRIOR_W = float(os.getenv("RE_RANKER_PRIOR_WEIGHT_TEXT", "0.4"))
            PRIOR_W = max(0.0, min(PRIOR_W, 1.0))
            MODEL_W = 1.0 - PRIOR_W

            for r, rs in zip(results, rerank_scores):
                r["re_rank_score"] = float(rs)
                f = r.get("candidate_features", {})
                prior_q = _prior_from_query(f)
                base = _model_norm(rs if rs is not None else r.get("score", 0.0))
                r["combined_score"] = MODEL_W * base + PRIOR_W * prior_q
                try:
                    _q("UPDATE query_candidates SET re_rank_score=%s WHERE query_id=%s AND sku_id=%s",
                       (float(rs), qid, r["sku_id"]), fetch=None)
                except Exception:
                    pass

            after = sorted(results, key=lambda x: x.get("combined_score", x.get("re_rank_score", x.get("score", 0.0))), reverse=True)
        except Exception:
            after = results

        # Quyết định có áp dụng re-rank không
        try:
            apply_rerank, applied_reason = _should_apply_rerank(request)
        except Exception:
            apply_rerank, applied_reason = (True, "default")

        out_list = after if apply_rerank else results

        return jsonify({
            "query_id": qid,
            "qtext_norm": qtext,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "applied_re_rank": bool(apply_rerank),
            "applied_reason": applied_reason,
            "results": results[:10],
            "reranked_results": after[:10]
        })

    except Exception as e:
        current_app.logger.exception("Text search error")
        return jsonify({"error": str(e), "results": []}), 500

@bp.get("/search/similar-skus")
def search_similar_skus():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 10))
    threshold = float(request.args.get("threshold", 0.3))

    if not query:
        return jsonify({"error": "Query text is required"}), 400

    try:
        query_vec = embed_text(query)
        query_vec_lit = _vec_literal(_l2norm(query_vec)) if query_vec else None

        if not query_vec_lit:
            return jsonify({"error": "Embedding failed", "results": []}), 500

        sql = """
            WITH combined_results AS (
                SELECT 
                    sc.sku_id, 
                    sc.image_path, 
                    sc.caption_text AS text, 
                    sc.keywords, 
                    sc.colors, 
                    sc.materials,
                    sc.brand_guess, 
                    sc.size_guess, 
                    sc.category_guess,
                    1 - (sc.caption_vec <=> %s::vector) AS similarity,
                    'caption' AS source
                FROM sku_captions sc
                WHERE sc.lang = 'vi' 
                  AND sc.style = 'search'
                  AND sc.caption_vec IS NOT NULL
                  AND 1 - (sc.caption_vec <=> %s::vector) > %s
                  AND to_tsvector('simple', unaccent(sc.caption_text || ' ' || array_to_string(sc.keywords, ' '))) 
                        @@ plainto_tsquery('simple', unaccent(%s))

                UNION ALL

                SELECT 
                    st.sku_id, 
                    NULL AS image_path, 
                    st.text AS text, 
                    ARRAY[]::text[] AS keywords, 
                    ARRAY[]::text[] AS colors, 
                    ARRAY[]::text[] AS materials,
                    NULL AS brand_guess, 
                    NULL AS size_guess, 
                    NULL AS category_guess,
                    1 - (st.text_vec <=> %s::vector) AS similarity,
                    'text' AS source
                FROM sku_texts st
                WHERE st.text_vec IS NOT NULL
                  AND 1 - (st.text_vec <=> %s::vector) > %s
                  AND to_tsvector('simple', unaccent(st.text)) 
                        @@ plainto_tsquery('simple', unaccent(%s))
            )
            SELECT * FROM combined_results
            ORDER BY similarity DESC
            LIMIT %s
        """
        params = (
            query_vec_lit, query_vec_lit, threshold, query,
            query_vec_lit, query_vec_lit, threshold, query,
            top_k
        )

        rows = _q(sql, params)

        # Tạo query log
        qid = _q(
            "INSERT INTO queries(type, raw_text, normalized_text) VALUES('text', %s, %s) RETURNING id",
            (query, vn_norm(query)), fetch="one"
        )[0]

        out = []
        for rank, r in enumerate(rows, start=1):
            feats = {
                "similarity": float(r[9]),
                "rank": int(rank),
                "source": r[10],
                "keywords_count": int(len(r[3]) if r[3] else 0),
            }
            _q(
                "INSERT INTO query_candidates(query_id, sku_id, rank, score, candidate_features) "
                "VALUES(%s,%s,%s,%s,%s)",
                (qid, int(r[0]), rank, float(r[9]), pg_extras.Json(feats)), fetch=None
            )
            out.append({
                "sku_id": int(r[0]),
                "image_path": r[1],
                "text": r[2],
                "keywords": r[3],
                "colors": r[4],
                "materials": r[5],
                "brand_guess": r[6],
                "size_guess": r[7],
                "category_guess": r[8],
                "similarity": float(r[9]),
                "source": r[10],
                "candidate_features": feats
            })

        # RE-RANK (ghi vào cột re_rank_score, không sửa features)
        try:
            try:
                _RE_RANKER.reload_if_needed()
            except Exception:
                pass
            feat_list = [x.get("candidate_features") for x in out]
            rerank_scores = _RE_RANKER.score_candidates(feat_list) or []
            for item, rs in zip(out, rerank_scores):
                item["re_rank_score"] = float(rs)
                try:
                    _q("UPDATE query_candidates SET re_rank_score=%s WHERE query_id=%s AND sku_id=%s",
                       (float(rs), qid, item["sku_id"]), fetch=None)
                except Exception:
                    pass

            reranked = sorted(out,
                              key=lambda x: x.get("re_rank_score", x.get("similarity", 0.0)),
                              reverse=True)
        except Exception:
            reranked = []

        return jsonify({
            "query": query,
            "total": len(out),
            "results": out,
            "reranked_results": reranked
        })

    except Exception as e:
        current_app.logger.exception("Similar SKUs search error")
        return jsonify({"error": str(e), "results": []}), 500

@bp.post("/search/image")
def search_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "results": []})

        file = request.files['file']
        k = min(int(request.form.get('k', 20)), 100)

        if file.filename == '':
            return jsonify({"error": "No file selected", "results": []})

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)

            try:
                from build_faiss_index import search_image_with_faiss
                results = search_image_with_faiss(tmp.name, k=k)

                if not results:
                    return jsonify({"error": "No results found", "results": []}), 404

                qid = _q("INSERT INTO queries(type) VALUES('image') RETURNING id", fetch="one")[0]

                out = []
                for result in results:
                    sku_id = result["sku_id"]
                    rank = result["rank"]
                    score = result["score"]

                    sku_info = _q("""
                        SELECT sk.name, b.name as brand_name, si.ocr_text
                        FROM skus sk
                        LEFT JOIN brands b ON b.id = sk.brand_id
                        LEFT JOIN sku_images si ON si.sku_id = sk.id AND si.image_path = %s
                        WHERE sk.id = %s
                        LIMIT 1
                    """, (result["image_path"], sku_id), fetch="one")

                    sku_name, brand_name, ocr_text = sku_info if sku_info else (None, None, None)

                    cand_feats = {
                        "similarity": float(score),
                        "rank": int(rank),
                        "source": "image",
                        "has_ocr": bool(ocr_text),
                        "brand_guess": brand_name,
                    }

                    _q(
                        "INSERT INTO query_candidates(query_id, sku_id, rank, score, candidate_features) "
                        "VALUES(%s,%s,%s,%s,%s)",
                        (qid, sku_id, rank, float(score), pg_extras.Json(cand_feats)), fetch=None
                    )

                    result["sku_id"] = int(sku_id)
                    result["img_id"] = int(result["img_id"])

                    out.append({
                        **result,
                        "sku_name": sku_name,
                        "brand_name": brand_name,
                        "ocr_text": ocr_text,
                        "candidate_features": cand_feats
                    })
                    
                # === LẤY POPULARITY CHO CẢ SLATE MỘT LẦN ===
                # === LẤY POPULARITY CHO CẢ SLATE MỘT LẦN (sau khi đã build 'out') ===
                sku_ids = [it["sku_id"] for it in out]
                pop_rows = _q("""
                    SELECT sku_id, COALESCE(clicks_30d,0)::int, COALESCE(views_30d,0)::int, COALESCE(ctr_30d,0.0)
                    FROM sku_popularity_30d
                    WHERE sku_id = ANY(%s)
                """, (sku_ids,), fetch="all") or []

                pop_map = {r[0]: {"pop_clicks_30d": int(r[1]),
                                  "pop_views_30d":  int(r[2]),
                                  "pop_ctr_30d":    float(r[3])} for r in pop_rows}

                for it in out:
                    it["candidate_features"].update(pop_map.get(it["sku_id"], {
                        "pop_clicks_30d": 0, "pop_views_30d": 0, "pop_ctr_30d": 0.0
                    }))

                # RE-RANK để hiển thị + có thể áp dụng
                try:
                    try:
                        _RE_RANKER.reload_if_needed()
                    except Exception:
                        pass

                    feat_list = [x.get("candidate_features") for x in out]
                    rerank_scores = _RE_RANKER.score_candidates(feat_list) or []
                    for it, rs in zip(out, rerank_scores):
                        it["re_rank_score"] = float(rs)
                        try:
                            _q("UPDATE query_candidates SET re_rank_score=%s WHERE query_id=%s AND sku_id=%s",
                               (float(rs), qid, it["sku_id"]), fetch=None)
                        except Exception:
                            pass

                    # PRIOR từ popularity (giống bản trước)
                    import math
                    def _prior_from_pop(p):
                        ctr = float(p.get("pop_ctr_30d", 0.0) or 0.0)           # [0..1]
                        clk = float(p.get("pop_clicks_30d", 0.0) or 0.0)
                        clk_norm = min(1.0, math.log1p(clk) / math.log(101.0))  # [0..1]
                        return 0.5 * ctr + 0.5 * clk_norm

                    def _s_model_norm(s):
                        return 0.5 * (math.tanh(float(s)) + 1.0)

                    PRIOR_W = float(os.getenv("RE_RANKER_PRIOR_WEIGHT", "0.5"))
                    PRIOR_W = max(0.0, min(PRIOR_W, 1.0))
                    MODEL_W = 1.0 - PRIOR_W

                    # (tùy chọn) auto-boost nếu prior chênh lệch quá lớn
                    priors = [_prior_from_pop(it["candidate_features"]) for it in out]
                    if priors:
                        sp = sorted(priors, reverse=True)
                        if (sp[0] - (sp[1] if len(sp) > 1 else 0.0)) > 0.25:
                            PRIOR_W = min(0.8, max(PRIOR_W, 0.6))
                            MODEL_W = 1.0 - PRIOR_W

                    for it in out:
                        p = it["candidate_features"]
                        prior = _prior_from_pop(p)
                        base  = _s_model_norm(it.get("re_rank_score", it.get("score", 0.0)))
                        it["combined_score"] = MODEL_W * base + PRIOR_W * prior

                    reranked = sorted(out, key=lambda x: x.get("combined_score", x.get("re_rank_score", x.get("score", 0.0))), reverse=True)
                except Exception:
                    reranked = []

                # Áp dụng theo cờ/canary
                try:
                    apply_rerank, applied_reason = _should_apply_rerank(request)
                except Exception:
                    apply_rerank, applied_reason = (True, 'default')

                results_to_return = reranked if apply_rerank and reranked else out

                return jsonify({
                    "query_id": qid,
                    "filename": secure_filename(file.filename),
                    "total": len(out),
                    "results": results_to_return,
                    "reranked_results": reranked,
                    "applied_re_rank": bool(apply_rerank and bool(reranked)),
                    "applied_reason": applied_reason
                })

            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    except Exception as e:
        current_app.logger.exception("Image search error")
        return jsonify({"error": str(e), "results": []}), 500

    
def embed_text(text):
    try:
        from utils import encode_texts
        emb = encode_texts([text])[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)
    except Exception as e:
        current_app.logger.warning("embed_text fallback: %s", e)
        return None

def _vec_literal(vec):
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]" if vec else None

def _l2norm(v):
    import math
    n = math.sqrt(sum(float(x)*float(x) for x in v)) or 1.0
    return [float(x)/n for x in v]


def _should_apply_rerank(req=None):
    """Decide whether to apply re-ranker ordering for a given request.

    Priority: request override -> canary -> global default.
    Returns (bool, reason_str).
    """
    # global default from env
    apply_env = os.environ.get('RE_RANKER_APPLY', '1')
    default_apply = True if str(apply_env) not in ['0', 'false', 'False'] else False

    # canary percent (0-100)
    try:
        canary = int(os.environ.get('RE_RANKER_CANARY', '0') or 0)
        if canary < 0:
            canary = 0
        if canary > 100:
            canary = 100
    except Exception:
        canary = 0

    # request-level override (form param or json)
    if req is not None:
        try:
            # prefer form data for image uploads
            val = None
            if hasattr(req, 'form'):
                val = req.form.get('apply_rerank')
            if val is None:
                try:
                    j = req.get_json(silent=True) or {}
                    val = j.get('apply_rerank')
                except Exception:
                    val = None
            if val is not None:
                if str(val).lower() in ['0', 'false', 'False']:
                    return False, 'request_override_false'
                return True, 'request_override_true'
        except Exception:
            pass

    # canary sampling
    if canary > 0:
        r = random.randint(1, 100)
        if r <= canary:
            return True, f'canary({canary}%)'
        else:
            return False, f'control({canary}%)'

    # fall back to default
    return default_apply, 'default'


@bp.post('/events/candidate_action')
def candidate_action():
    """Record user interaction with a candidate. Expects JSON with:
    { query_id: int, sku_id: int, action: 'click'|'purchase'|'dwell', dwell_time: optional seconds }
    Updates query_candidates.was_clicked / dwell_time / purchased accordingly.
    """
    try:
        data = request.get_json(silent=True) or {}
        qid = data.get('query_id')
        sku_id = data.get('sku_id')
        action = data.get('action')
        dwell = data.get('dwell_time')
        if not qid or not sku_id or not action:
            return jsonify({'error': 'missing query_id/sku_id/action'}), 400

        if action == 'click':
            # set was_clicked true and update dwell_time if provided
            if dwell is not None:
                _q("UPDATE query_candidates SET was_clicked = TRUE, dwell_time = %s WHERE query_id=%s AND sku_id=%s",
                   (float(dwell), qid, sku_id), fetch=None)
            else:
                _q("UPDATE query_candidates SET was_clicked = TRUE WHERE query_id=%s AND sku_id=%s",
                   (qid, sku_id), fetch=None)
            return jsonify({'ok': True})

        if action == 'purchase':
            _q("UPDATE query_candidates SET purchased = TRUE WHERE query_id=%s AND sku_id=%s",
               (qid, sku_id), fetch=None)
            return jsonify({'ok': True})

        if action == 'dwell':
            if dwell is None:
                return jsonify({'error': 'missing dwell_time'}), 400
            _q("UPDATE query_candidates SET dwell_time = %s WHERE query_id=%s AND sku_id=%s",
               (float(dwell), qid, sku_id), fetch=None)
            return jsonify({'ok': True})

        return jsonify({'error': 'unknown action'}), 400

    except Exception as e:
        current_app.logger.exception('candidate_action error')
        return jsonify({'error': str(e)}), 500


@bp.post('/admin/reload_re_ranker')
def admin_reload_re_ranker():
    """Admin endpoint to force the re-ranker to reload its model from disk.
    Call this after a retrain completes to atomically switch to the new model.
    """
    try:
        # optional JSON body: { "model_path": "/abs/or/rel/path/to/model.pt" }
        data = request.get_json(silent=True) or {}
        model_path = data.get('model_path')

        # If caller didn't provide a model_path, try to find the newest metadata in ./models
        if not model_path:
            try:
                md_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
                md_dir = os.path.abspath(md_dir)
            except Exception:
                md_dir = os.path.abspath('models')
            model_path = None
            try:
                # look for files named re_ranker_*.json
                metas = sorted([os.path.join(md_dir, p) for p in os.listdir(md_dir) if p.startswith('re_ranker_') and p.endswith('.json')], key=lambda x: os.path.getmtime(x), reverse=True)
                if metas:
                    try:
                        with open(metas[0], 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            model_path = meta.get('model_path')
                    except Exception:
                        model_path = None
            except Exception:
                model_path = None

        if model_path:
            # try to load the specific model file into the re-ranker
            try:
                # If ReRanker exposes load(), use it. Otherwise try reload_if_needed.
                if hasattr(_RE_RANKER, 'load'):
                    _RE_RANKER.load(model_path)
                    return jsonify({'ok': True, 'loaded_model': model_path})
                elif hasattr(_RE_RANKER, 'reload_if_needed'):
                    # set the model_path on the instance and force reload
                    try:
                        _RE_RANKER.model_path = model_path
                    except Exception:
                        pass
                    ok = _RE_RANKER.reload_if_needed(force=True)
                    return jsonify({'ok': True, 'reloaded': bool(ok), 'model_path': model_path})
            except Exception as e:
                current_app.logger.exception('failed to load specified model')
                return jsonify({'ok': False, 'error': str(e)}), 500

        # fallback: call reload_if_needed without explicit model path
        if hasattr(_RE_RANKER, 'reload_if_needed'):
            ok = _RE_RANKER.reload_if_needed(force=True)
            return jsonify({'ok': True, 'reloaded': bool(ok)})

        return jsonify({'ok': False, 'error': 're-ranker does not support reload'}), 501
    except Exception as e:
        current_app.logger.exception('reload_re_ranker error')
        return jsonify({'ok': False, 'error': str(e)}), 500


@bp.get('/admin/re_ranker_status')
def admin_re_ranker_status():
    """Return current re-ranker status: model_path, last_mtime and availability."""
    try:
        info = {
            'has_re_ranker': True,
            'has_torch': False,
            'model_path': None,
            'last_mtime': None,
            'using_model': False
        }
        # detect torch availability from the instance if possible
        try:
            info['has_torch'] = hasattr(_RE_RANKER, 'model') and _RE_RANKER.model is not None
        except Exception:
            info['has_torch'] = False

        try:
            if hasattr(_RE_RANKER, 'model_path'):
                info['model_path'] = _RE_RANKER.model_path
            if hasattr(_RE_RANKER, '_last_mtime'):
                info['last_mtime'] = _RE_RANKER._last_mtime
            info['using_model'] = bool(info['model_path']) and info['has_torch']
            # report current env settings for apply/canary
            try:
                info['RE_RANKER_APPLY'] = os.environ.get('RE_RANKER_APPLY', None)
                info['RE_RANKER_CANARY'] = os.environ.get('RE_RANKER_CANARY', None)
            except Exception:
                pass
        except Exception:
            pass

        return jsonify(info)
    except Exception as e:
        current_app.logger.exception('re_ranker_status error')
        return jsonify({'error': str(e)}), 500