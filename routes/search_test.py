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

@bp.post("/search/text_with_image")
def search_text_with_image():
    try:
        data = request.get_json(silent=True) or {}
        raw_q = (data.get("q") or "").strip()
        norm_q = vn_norm(raw_q)

        if not norm_q:
            return jsonify({"error": "missing q", "results": []}), 400

        # Giới hạn top-k
        try:
            k = int(data.get("k") or 20)
            k = max(1, min(k, 100))
        except Exception:
            k = 20

        t0 = time.time()

        # ==== Encode text để tìm kiếm ====
        from utils import encode_texts
        vec = encode_texts([norm_q])[0]
        vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        vec = _l2norm(vec)

        # ==== Ghi log truy vấn ====
        qid_row = _q(
            "INSERT INTO queries(type, raw_text, normalized_text) VALUES('text', %s, %s) RETURNING id",
            (raw_q, norm_q), fetch="one"
        )
        qid = int(qid_row[0]) if qid_row else None

        # ==== Truy vấn top-k kết quả ====
        rows = _q("""
            SELECT st.sku_id,
                   st.id AS sku_text_id,
                   st.text AS raw_text,
                   (st.text_vec <=> %s::vector) AS dist
            FROM sku_texts st
            WHERE st.text_vec IS NOT NULL
            ORDER BY st.text_vec <=> %s::vector
            LIMIT %s
        """, (vec, vec, k), fetch="all")

        if not rows:
            return jsonify({
                "query_id": qid,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "results": []
            })

        sku_ids = [int(r[0]) for r in rows]

        # ==== Ghép thêm thông tin mô tả có dấu + ảnh ====
        enrich = _q("""
            SELECT sk.id AS sku_id,
                   sk.name AS sku_name,
                   COALESCE(b.name, '') AS brand_name,
                   COALESCE(sc.caption_text, si.ocr_text, st.text, '') AS description,  -- Ưu tiên caption_text có dấu
                   COALESCE(sc.image_path, si.image_path, '') AS image_path
            FROM skus sk
            LEFT JOIN brands b ON b.id = sk.brand_id
            LEFT JOIN LATERAL (
                SELECT caption_text, image_path
                FROM sku_captions
                WHERE sku_id = sk.id AND lang = 'vi' AND style = 'search'
                ORDER BY id ASC LIMIT 1
            ) sc ON TRUE
            LEFT JOIN LATERAL (
                SELECT image_path, ocr_text
                FROM sku_images
                WHERE sku_id = sk.id
                ORDER BY (ocr_text IS NULL) ASC, id ASC
                LIMIT 1
            ) si ON TRUE
            LEFT JOIN LATERAL (
                SELECT text
                FROM sku_texts
                WHERE sku_id = sk.id
                LIMIT 1
            ) st ON TRUE
            WHERE sk.id = ANY(%s)
        """, (sku_ids,), fetch="all")

        info = {
            int(r[0]): {
                "sku_name": r[1],
                "brand_name": r[2],
                "description": r[3],
                "image_path": r[4],
            } for r in enrich or []
        }

        # ==== Gộp kết quả + lưu vào query_candidates ====
        results = []
        for rank, (sku_id, sku_text_id, raw_text, dist) in enumerate(rows, start=1):
            try:
                sku_id = int(sku_id)
                sku_text_id = int(sku_text_id)
                dist = float(dist)
            except Exception:
                continue

            score = 1.0 - dist
            if qid is not None:
                _q("""
                    INSERT INTO query_candidates(query_id, sku_id, rank, score)
                    VALUES(%s, %s, %s, %s)
                """, (qid, sku_id, rank, score), fetch=None)

            extra = info.get(sku_id, {})
            results.append({
                "sku_id": sku_id,
                "sku_text_id": sku_text_id,
                "description": extra.get("description") or raw_text,
                "score": score,
                "dist": dist,
                "sku_name": extra.get("sku_name"),
                "brand_name": extra.get("brand_name"),
                "image_path": extra.get("image_path"),

                # Cho Android
                "skuId": sku_id,
                "skuTextId": sku_text_id,
                "skuName": extra.get("sku_name"),
                "brandName": extra.get("brand_name"),
                "imagePath": extra.get("image_path"),
                "descriptionText": extra.get("description") or raw_text,
            })

        return jsonify({
            "query_id": qid,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "query_text": raw_q,  # ✅ bản có dấu
            "total": len(results),
            "results": results
        })

    except Exception as e:
        current_app.logger.exception("Text-with-image search error")
        return jsonify({"error": str(e), "results": []}), 500

@bp.get("/search/similar-skus")
def search_similar_skus():
    query = request.args.get("q", "").strip()
    page = max(int(request.args.get("page", 1) or 1), 1)
    page_size = max(1, min(int(request.args.get("page_size", 5) or 5), 50))
    threshold = float(request.args.get("threshold", 0.3))

    if not query:
        return jsonify({"error": "Query text is required"}), 400

    try:
        query_vec = embed_text(query)
        query_vec_lit = _vec_literal(_l2norm(query_vec)) if query_vec else None
        if not query_vec_lit:
            return jsonify({"error": "Embedding failed", "results": []}), 500

        base_sql = """
            WITH base AS (
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
            ),
            collapsed AS (
                SELECT DISTINCT ON (sku_id)
                    sku_id, image_path, text, keywords, colors, materials,
                    brand_guess, size_guess, category_guess, similarity, source
                FROM base
                ORDER BY sku_id, similarity DESC, CASE WHEN source='caption' THEN 1 ELSE 0 END DESC
            ),
            ordered AS (
                SELECT *
                FROM collapsed
                ORDER BY similarity DESC, sku_id DESC
            )
        """
        base_params = (
            query_vec_lit, query_vec_lit, threshold, query,
            query_vec_lit, query_vec_lit, threshold, query,
        )

        total = int(_q(base_sql + " SELECT COUNT(*) FROM ordered; ", base_params, fetch="one")[0])

        offset = (page - 1) * page_size
        rows = _q(base_sql + " SELECT * FROM ordered OFFSET %s LIMIT %s; ",
                  base_params + (offset, page_size), fetch="all")

        qid = _q(
            "INSERT INTO queries(type, raw_text, normalized_text) VALUES('text', %s, %s) RETURNING id",
            (query, vn_norm(query)), fetch="one"
        )[0]

        # map rows -> raw_items
        raw_items = []
        for idx, r in enumerate(rows, start=1):
            raw_items.append({
                "sku_id": int(r[0]),
                "image_path": r[1],
                "base_text": r[2],
                "score": float(r[9]),
                "rank": int(offset + idx),
                "source": r[10],
                "candidate_features": {
                    "keywords_count": int(len(r[3]) if r[3] else 0)
                }
            })

        enrich_map = _enrich_skus([x["sku_id"] for x in raw_items])
        items = _build_output_items(raw_items, enrich_map, qid, query_type="text")
        _attach_popularity(items)
        items = _rerank_and_sort(items, qid, prior_weight_env_key="RE_RANKER_PRIOR_WEIGHT")

        total_pages = (total + page_size - 1) // page_size
        return jsonify({
            "query_id": qid,
            "query_text": query,
            "paging": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "sql_offset": offset,
                "sql_limit": page_size
            },
            "results": items
        }), 200

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
        try:
            threshold = float(request.form.get('threshold', 0.0))  # tuỳ chọn lọc theo score
        except Exception:
            threshold = 0.0

        if file.filename == '':
            return jsonify({"error": "No file selected", "results": []})

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)

            try:
                # ========= TÌM ẢNH → ẢNH (FAISS) =========
                from build_faiss_index import search_image_with_faiss
                raw_results = search_image_with_faiss(tmp.name, k=k) or []
                # lọc theo threshold nếu có
                results = [r for r in raw_results if float(r.get("score", 0.0)) >= float(threshold)]

                if not results:
                    return jsonify({"error": "No results found", "results": []}), 404

                # ========= GHI QUERY =========
                qid = _q("INSERT INTO queries(type) VALUES('image') RETURNING id", fetch="one")[0]

                # map FAISS -> raw_items
                raw_items = []
                for idx, r in enumerate(results, start=1):
                    sku_id   = int(r["sku_id"])
                    rank     = int(r.get("rank", idx))
                    score    = float(r.get("score", 0.0))
                    img_path = r.get("image_path")

                    # Nếu muốn rơi về OCR đúng ảnh khi enrich không có mô tả:
                    base_txt = _ocr_for_image(sku_id, img_path) or None

                    raw_items.append({
                        "sku_id": sku_id,
                        "image_path": img_path,
                        "base_text": base_txt,     # có thể None, _build_output_items sẽ fallback
                        "score": score,
                        "rank": rank,
                        "source": "image",
                        "candidate_features": {
                            "has_ocr": bool(base_txt)
                        }
                    })

                # ===== HẬU XỬ LÝ CHUNG =====
                enrich_map = _enrich_skus([x["sku_id"] for x in raw_items])
                items = _build_output_items(raw_items, enrich_map, qid, query_type="image")
                _attach_popularity(items)
                items = _rerank_and_sort(items, qid, prior_weight_env_key="RE_RANKER_PRIOR_WEIGHT")

                # ===== APPLY FLAG (giữ nguyên logic của bạn) =====
                try:
                    apply_rerank, applied_reason = _should_apply_rerank(request)
                except Exception:
                    apply_rerank, applied_reason = (True, 'default')

                items_to_return = items if apply_rerank else sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)

                return jsonify({
                    "query_id": qid,
                    "filename": secure_filename(file.filename),
                    "total": len(items),
                    "results": items_to_return,
                    "applied_re_rank": bool(apply_rerank),
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

# ====== SHARED HELPERS (enrich + popularity + rerank + build json) ======

def _enrich_skus(sku_ids):
    """Trả về map {sku_id: {sku_name, brand_name, description, image_path}} giống search_text_with_image."""
    if not sku_ids:
        return {}
    rows = _q("""
        SELECT sk.id AS sku_id,
               sk.name AS sku_name,
               COALESCE(b.name, '') AS brand_name,
               COALESCE(sc.caption_text, si.ocr_text, st.text, '') AS description,
               COALESCE(sc.image_path, si.image_path, '') AS image_path
        FROM skus sk
        LEFT JOIN brands b ON b.id = sk.brand_id
        LEFT JOIN LATERAL (
            SELECT caption_text, image_path
            FROM sku_captions
            WHERE sku_id = sk.id AND lang='vi' AND style='search'
            ORDER BY id ASC LIMIT 1
        ) sc ON TRUE
        LEFT JOIN LATERAL (
            SELECT image_path, ocr_text
            FROM sku_images
            WHERE sku_id = sk.id
            ORDER BY (ocr_text IS NULL) ASC, id ASC
            LIMIT 1
        ) si ON TRUE
        LEFT JOIN LATERAL (
            SELECT text
            FROM sku_texts
            WHERE sku_id = sk.id
            ORDER BY id ASC LIMIT 1
        ) st ON TRUE
        WHERE sk.id = ANY(%s)
    """, (list(set(sku_ids)),), fetch="all") or []
    return {
        int(r[0]): {
            "sku_name": r[1],
            "brand_name": r[2],
            "description": r[3],
            "image_path": r[4],
        } for r in rows
    }

def _attach_popularity(items):
    """Gắn chỉ số popularity toàn cục (pop_*) vào candidate_features (nếu có bảng)."""
    try:
        sku_ids = [it["sku_id"] for it in items] or []
        if not sku_ids:
            return
        rows = _q("""
            SELECT sku_id, COALESCE(clicks_30d,0)::int, COALESCE(views_30d,0)::int, COALESCE(ctr_30d,0.0)
            FROM sku_popularity_30d
            WHERE sku_id = ANY(%s)
        """, (sku_ids,), fetch="all") or []
        pop = {r[0]: {"pop_clicks_30d": int(r[1]),
                      "pop_views_30d":  int(r[2]),
                      "pop_ctr_30d":    float(r[3])} for r in rows}
        for it in items:
            it.setdefault("candidate_features", {})
            it["candidate_features"].update(pop.get(it["sku_id"], {
                "pop_clicks_30d": 0, "pop_views_30d": 0, "pop_ctr_30d": 0.0
            }))
    except Exception:
        pass

def _rerank_and_sort(items, qid, prior_weight_env_key="RE_RANKER_PRIOR_WEIGHT"):
    """Chạy re-ranker + prior (popularity) và sort; trả về list đã sắp xếp."""
    try:
        try:
            _RE_RANKER.reload_if_needed()
        except Exception:
            pass

        feat_list = [x.get("candidate_features") for x in items]
        rerank_scores = _RE_RANKER.score_candidates(feat_list) or []
        for it, rs in zip(items, rerank_scores):
            it["re_rank_score"] = float(rs)
            try:
                _q("UPDATE query_candidates SET re_rank_score=%s WHERE query_id=%s AND sku_id=%s",
                   (float(rs), qid, it["sku_id"]), fetch=None)
            except Exception:
                pass

        import math
        def _prior_from_pop(p):
            ctr = float((p or {}).get("pop_ctr_30d", 0.0) or 0.0)
            clk = float((p or {}).get("pop_clicks_30d", 0.0) or 0.0)
            clk_norm = min(1.0, math.log1p(clk) / math.log(101.0))
            return 0.5 * ctr + 0.5 * clk_norm

        def _model_norm(s):
            return 0.5 * (math.tanh(float(s)) + 1.0)

        PRIOR_W = float(os.getenv(prior_weight_env_key, "0.5"))
        PRIOR_W = max(0.0, min(PRIOR_W, 1.0))
        MODEL_W = 1.0 - PRIOR_W

        priors = [_prior_from_pop(it.get("candidate_features", {})) for it in items]
        if priors:
            sp = sorted(priors, reverse=True)
            if (sp[0] - (sp[1] if len(sp) > 1 else 0.0)) > 0.25:
                PRIOR_W = min(0.8, max(PRIOR_W, 0.6))
                MODEL_W = 1.0 - PRIOR_W

        for it in items:
            prior = _prior_from_pop(it.get("candidate_features", {}))
            base  = _model_norm(it.get("re_rank_score", it.get("score", 0.0)))
            it["combined_score"] = MODEL_W * base + PRIOR_W * prior

        items.sort(key=lambda x: x.get("combined_score", x.get("re_rank_score", x.get("score", 0.0))), reverse=True)
    except Exception:
        pass
    return items

def _build_output_items(raw_items, enrich_map, qid, query_type="text"):
    """
    raw_items: list dict gồm tối thiểu:
      - sku_id (int), score (float), rank (int),
      - optional: source (str), image_path (str), base_text (str), candidate_features (dict)
    Trả về list item chuẩn hóa field giống search_text_with_image (snake_case + camelCase) và ghi query_candidates.
    """
    out = []
    for it in raw_items:
        sku_id   = int(it["sku_id"])
        score    = float(it.get("score", 0.0))
        rank     = int(it.get("rank", 0))
        source   = it.get("source", query_type)
        img_src  = it.get("image_path")
        base_txt = it.get("base_text")

        meta = enrich_map.get(sku_id, {})
        description = meta.get("description") or (base_txt or "")
        image_path  = meta.get("image_path") or img_src

        feats = dict(it.get("candidate_features", {}))
        feats.update({
            "similarity": score,
            "rank": rank,
            "source": source,
        })

        # ghi candidate
        try:
            _q("""INSERT INTO query_candidates(query_id, sku_id, rank, score, candidate_features)
                  VALUES(%s,%s,%s,%s,%s)""",
               (qid, sku_id, rank, score, pg_extras.Json(feats)), fetch=None)
        except Exception:
            pass

        out.append({
            "sku_id": sku_id,
            "sku_text_id": None,
            "description": description,
            "score": score,
            "dist": 1.0 - score,
            "sku_name": meta.get("sku_name"),
            "brand_name": meta.get("brand_name"),
            "image_path": image_path,

            # camelCase
            "skuId": sku_id,
            "skuTextId": None,
            "skuName": meta.get("sku_name"),
            "brandName": meta.get("brand_name"),
            "imagePath": image_path,
            "descriptionText": description,

            # giữ thêm
            "source": source,
            "candidate_features": feats
        })
    return out

def _ocr_for_image(sku_id, img_path):
    """Lấy OCR đúng ảnh; dùng khi muốn rơi về OCR nếu enrich không có mô tả."""
    try:
        row = _q("SELECT ocr_text FROM sku_images WHERE sku_id=%s AND image_path=%s LIMIT 1",
                 (sku_id, img_path), fetch="one")
        return (row and row[0]) or None
    except Exception:
        return None

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