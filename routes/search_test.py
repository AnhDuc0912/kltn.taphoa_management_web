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
import faiss
from services.resnet101 import load_model, extract_embedding, preprocess_image  # giữ import như cũ
_MODEL = load_model() 

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
    qtext = vn_norm((data.get("q") or "").strip())
    k = min(int(data.get("k") or 20), 100)
    if not qtext:
        return jsonify({"error": "missing q"}), 400

    try:
        t0 = time.time()
        vec = encode_texts([qtext])[0].tolist()

        qid = _q(
            "INSERT INTO queries(type, raw_text, normalized_text) VALUES('text', %s, %s) RETURNING id",
            (qtext, vn_norm(qtext)), fetch="one"
        )[0]

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
            _q("INSERT INTO query_candidates(query_id, sku_id, rank, score) VALUES(%s,%s,%s,%s)",
               (qid, sku_id, rank, score), fetch=None)
            results.append({
                "sku_id": sku_id, "sku_text_id": st_id, "text": text,
                "dist": float(dist), "score": score
            })

        return jsonify({
            "query_id": qid,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "results": results
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

        # 🔹 Dùng unaccent nếu có
        sql = """
            WITH combined_results AS (
                SELECT 
                    s.id AS sku_id,
                    s.name AS sku_name,
                    b.name AS brand_name,
                    sc.image_path,
                    sc.caption_text AS text,
                    1 - (sc.caption_vec <=> %s::vector) AS similarity,
                    'caption' AS source
                FROM sku_captions sc
                JOIN skus s ON s.id = sc.sku_id
                LEFT JOIN brands b ON b.id = s.brand_id
                WHERE sc.lang = 'vi'
                  AND sc.style = 'search'
                  AND sc.caption_vec IS NOT NULL
                  AND 1 - (sc.caption_vec <=> %s::vector) > %s
                  AND to_tsvector('simple', unaccent(sc.caption_text)) @@ plainto_tsquery('simple', unaccent(%s))

                UNION ALL

                SELECT 
                    s.id AS sku_id,
                    s.name AS sku_name,
                    b.name AS brand_name,
                    COALESCE(sc.image_path, '') AS image_path,
                    st.text AS text,
                    1 - (st.text_vec <=> %s::vector) AS similarity,
                    'text' AS source
                FROM sku_texts st
                JOIN skus s ON s.id = st.sku_id
                LEFT JOIN brands b ON b.id = s.brand_id
                LEFT JOIN sku_captions sc ON sc.sku_id = s.id
                WHERE st.text_vec IS NOT NULL
                  AND 1 - (st.text_vec <=> %s::vector) > %s
                  AND to_tsvector('simple', unaccent(st.text)) @@ plainto_tsquery('simple', unaccent(%s))
            )
            SELECT * FROM combined_results
            ORDER BY similarity DESC
            LIMIT %s
        """

        params = (
            query_vec_lit, query_vec_lit, threshold, query,  # block 1
            query_vec_lit, query_vec_lit, threshold, query,  # block 2
            top_k
        )

        rows = q(sql, params)

        out = []
        for r in rows:
            out.append({
                "sku_id": int(r[0]),
                "sku_name": r[1],
                "brand_name": r[2],
                "image_path": r[3],
                "text": r[4],
                "similarity": float(r[5]),
                "source": r[6]
            })

        return jsonify({
            "query": query,
            "total": len(out),
            "results": out
        })

    except Exception as e:
        current_app.logger.exception("Similar SKUs search error")
        return jsonify({"error": str(e), "results": []}), 500

@bp.post("/search/image")
def search_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "results": []}), 400

        file = request.files['file']
        if not file or file.filename.strip() == '':
            return jsonify({"error": "No file selected", "results": []}), 400

        # k: 1..100 (mặc định 20)
        try:
            k = int(request.form.get('k', 20))
            k = max(1, min(k, 100))
        except Exception:
            k = 20

        t0 = time.time()

        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # === Tìm top-k bằng FAISS (giữ nguyên pipeline của bạn) ===
            from build_faiss_index import search_image_with_faiss
            results = search_image_with_faiss(tmp_path, k=k)

            if not results:
                return jsonify({"error": "No results found", "results": []}), 404

            # Tạo query id
            qid_row = _q("INSERT INTO queries(type) VALUES('image') RETURNING id", fetch="one")
            qid = int(qid_row[0]) if qid_row else None

            # Gom sku_id để enrich 1 lần
            sku_ids = [int(r.get("sku_id")) for r in results if r.get("sku_id") is not None]
            if not sku_ids:
                return jsonify({
                    "query_id": qid, "filename": secure_filename(file.filename),
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "total": 0, "results": []
                })

            # === ENRICH: lấy mô tả CÓ DẤU + ảnh đại diện ===
            # Ưu tiên: captions.vi.style=search  ->  images.ocr_text  ->  texts.text
            enrich_rows = _q("""
                SELECT sk.id AS sku_id,
                       sk.name AS sku_name,
                       COALESCE(b.name,'') AS brand_name,
                       COALESCE(sc.caption_text, si.ocr_text, st.text, '') AS description,
                       COALESCE(sc.image_path, si.image_path, '') AS image_path,
                       COALESCE(si.ocr_text, '') AS ocr_text
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

            enrich_map = {
                int(r[0]): {
                    "sku_name": r[1],
                    "brand_name": r[2],
                    "description": r[3],   # ✅ có dấu nếu caption/ocr có
                    "image_path": r[4],
                    "ocr_text": r[5],
                } for r in enrich_rows or []
            }

            out = []
            for r in results:
                sku_id = int(r.get("sku_id"))
                score = float(r.get("score", 0.0))
                rank  = int(r.get("rank", 0))

                # lưu ứng viên
                if 'img_id' in r and r['img_id'] is not None:
                    try: r['img_id'] = int(r['img_id'])
                    except: pass

                if qid is not None:
                    _q("""
                        INSERT INTO query_candidates(query_id, sku_id, rank, score)
                        VALUES(%s, %s, %s, %s)
                    """, (qid, sku_id, rank, score), fetch=None)

                ex = enrich_map.get(sku_id, {})
                desc = ex.get("description")  # có dấu nếu có caption/ocr
                image_path = ex.get("image_path") or r.get("image_path")  # giữ fallback từ kết quả FAISS nếu có

                out.append({
                    # từ FAISS
                    **r,
                    "sku_id": sku_id,
                    "score": score,
                    "rank": rank,

                    # enrich
                    "sku_name": ex.get("sku_name"),
                    "brand_name": ex.get("brand_name"),
                    "description": desc,
                    "image_path": image_path,
                    "ocr_text": ex.get("ocr_text"),

                    # alias camelCase cho Android
                    "skuId": sku_id,
                    "skuName": ex.get("sku_name"),
                    "brandName": ex.get("brand_name"),
                    "descriptionText": desc,
                    "imagePath": image_path,
                    "ocrText": ex.get("ocr_text"),
                })

            return jsonify({
                "query_id": qid,
                "filename": secure_filename(file.filename),
                "elapsed_ms": int((time.time() - t0) * 1000),
                "total": len(out),
                "results": out
            })

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        current_app.logger.exception("Image search error")
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
    n = math.sqrt(sum(float(x) * float(x) for x in v)) or 1.0
    return [float(x) / n for x in v]
