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

        # --------- PHƯƠNG ÁN A: dùng unaccent (khuyến nghị) ----------
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
            query_vec_lit, query_vec_lit, threshold, query,  # block 1
            query_vec_lit, query_vec_lit, threshold, query,  # block 2
            top_k
        )

        # --------- PHƯƠNG ÁN B (nếu chưa có extension unaccent) ----------
        # Chỉ cần thay sql ở trên bằng đoạn dưới (và giữ nguyên params):
        #
        # sql = """
        #     WITH combined_results AS (
        #         SELECT 
        #             sc.sku_id, 
        #             sc.image_path, 
        #             sc.caption_text AS text, 
        #             sc.keywords, 
        #             sc.colors, 
        #             sc.materials,
        #             sc.brand_guess, 
        #             sc.size_guess, 
        #             sc.category_guess,
        #             1 - (sc.caption_vec <=> %s::vector) AS similarity,
        #             'caption' AS source
        #         FROM sku_captions sc
        #         WHERE sc.lang = 'vi' 
        #           AND sc.style = 'search'
        #           AND sc.caption_vec IS NOT NULL
        #           AND 1 - (sc.caption_vec <=> %s::vector) > %s
        #           AND to_tsvector('simple', sc.caption_text || ' ' || array_to_string(sc.keywords, ' ')) 
        #                 @@ plainto_tsquery('simple', %s)
        #
        #         UNION ALL
        #
        #         SELECT 
        #             st.sku_id, 
        #             NULL AS image_path, 
        #             st.text AS text, 
        #             ARRAY[]::text[] AS keywords, 
        #             ARRAY[]::text[] AS colors, 
        #             ARRAY[]::text[] AS materials,
        #             NULL AS brand_guess, 
        #             NULL AS size_guess, 
        #             NULL AS category_guess,
        #             1 - (st.text_vec <=> %s::vector) AS similarity,
        #             'text' AS source
        #         FROM sku_texts st
        #         WHERE st.text_vec IS NOT NULL
        #           AND 1 - (st.text_vec <=> %s::vector) > %s
        #           AND to_tsvector('simple', st.text) 
        #                 @@ plainto_tsquery('simple', %s)
        #     )
        #     SELECT * FROM combined_results
        #     ORDER BY similarity DESC
        #     LIMIT %s
        # """

        rows = q(sql, params)

        out = []
        for r in rows:
            out.append({
                "sku_id": r[0],
                "image_path": r[1],
                "text": r[2],
                "keywords": r[3],
                "colors": r[4],
                "materials": r[5],
                "brand_guess": r[6],
                "size_guess": r[7],
                "category_guess": r[8],
                "similarity": float(r[9]),
                "source": r[10]
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
            return jsonify({"error": "No file uploaded", "results": []})

        file = request.files['file']
        k = min(int(request.form.get('k', 20)), 100)

        if file.filename == '':
            return jsonify({"error": "No file selected", "results": []})

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)

            try:
                # Import search function
                from build_faiss_index import search_image_with_faiss
                
                # Search using FAISS (follows notebook workflow)
                results = search_image_with_faiss(tmp.name, k=k)
                
                if not results:
                    return jsonify({"error": "No results found", "results": []}), 404
                
                # Create query record
                qid = _q("INSERT INTO queries(type) VALUES('image') RETURNING id", fetch="one")[0]
                
                # Save candidates to DB
                out = []
                for result in results:
                    sku_id = result["sku_id"]
                    rank = result["rank"]
                    score = result["score"]
                    
                    _q("INSERT INTO query_candidates(query_id, sku_id, rank, score) VALUES(%s,%s,%s,%s)",
                       (qid, sku_id, rank, score), fetch=None)
                    
                    # Get additional SKU info
                    sku_info = _q("""
                        SELECT sk.name, b.name as brand_name, si.ocr_text
                        FROM skus sk
                        LEFT JOIN brands b ON b.id = sk.brand_id
                        LEFT JOIN sku_images si ON si.sku_id = sk.id AND si.image_path = %s
                        WHERE sk.id = %s
                        LIMIT 1
                    """, (result["image_path"], sku_id), fetch="one")
                    
                    sku_name, brand_name, ocr_text = sku_info if sku_info else (None, None, None)
                    
                    out.append({
                        **result,
                        "sku_name": sku_name,
                        "brand_name": brand_name,
                        "ocr_text": ocr_text
                    })

                return jsonify({
                    "query_id": qid,
                    "filename": secure_filename(file.filename),
                    "total": len(out),
                    "results": out
                })

            finally:
                try:
                    os.unlink(tmp.name)
                except:
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