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
from utils import encode_texts  # Giả sử encode_texts từ utils, nếu không thì xóa
from services.resnet101 import load_model, extract_embedding  # Tích hợp ResNet

bp = Blueprint("search_test_bp", __name__)

# Use the local vn_norm implementation in this file (do NOT import from utils)
def vn_norm(s: str) -> str:
    s = (s or "").strip().lower()
    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # keep a-z 0-9 and separators
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

@bp.get("/search-test")
def search_test_page():
    """ Trang test search UI"""
    return render_template("search_test.html")

@bp.get("/search")
def search_page():
    """ Trang test search UI"""
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
        vec = encode_texts([qtext])[0].tolist()  # Giữ nguyên encode_texts

        # log query
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
    query = request.args.get("q", "").strip()  # Query text từ GET param, ví dụ: ?q=bia+lon+xanh+lá
    top_k = int(request.args.get("top_k", 10))  # Số lượng kết quả, mặc định 10
    threshold = float(request.args.get("threshold", 0.3))  # Similarity threshold, mặc định 0.3
    
    if not query:
        return jsonify({"error": "Query text is required"}), 400
    
    try:
        # Embed query thành vector 512D
        query_vec = embed_text(query)  # Sửa fallback
        query_vec_lit = _vec_literal(_l2norm(query_vec))  # Chuẩn hóa và định dạng string cho SQL
        
        # Query SQL: Kết hợp sku_captions và sku_texts
        sql = """
            WITH combined_results AS (
                -- Tìm từ sku_captions
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
                  AND to_tsvector('simple', sc.caption_text || ' ' || array_to_string(sc.keywords, ' ')) @@ to_tsquery('simple', %s)
                
                UNION ALL
                
                -- Tìm từ sku_texts
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
                  AND to_tsvector('simple', st.text) @@ to_tsquery('simple', %s)
            )
            SELECT * FROM combined_results
            ORDER BY similarity DESC
            LIMIT %s
        """
        params = (query_vec_lit, query_vec_lit, threshold, vn_norm(query), 
                  query_vec_lit, query_vec_lit, threshold, vn_norm(query), top_k)
        
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
                "similarity": r[9],
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
    """API search bằng upload ảnh"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "results": []})

        file = request.files['file']
        k = min(int(request.form.get('k', 20)), 100)

        if file.filename == '':
            return jsonify({"error": "No file selected", "results": []})

        # Save temp file và extract features với ResNet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)

            try:
                model = load_model()  # Load ResNet từ resnet101.py
                query_vector = extract_embedding(model, tmp.name)  # Sử dụng ResNet để embedding chính xác hơn

                if query_vector is None:
                    return jsonify({"error": "Embedding extraction failed", "results": []}), 500

                # Log query (merge từ khối đầu)
                qid = _q("INSERT INTO queries(type) VALUES('image') RETURNING id", fetch="one")[0]

                # Vector similarity search trong sku_images với COALESCE
                rows = q("""
                    SELECT 
                        si.sku_id,
                        si.image_path,
                        1 - (COALESCE(si.image_vec_768, si.image_vec) <=> %s::vector) as score,
                        sk.name as sku_name,
                        b.name as brand_name,
                        si.ocr_text
                    FROM sku_images si
                    LEFT JOIN skus sk ON sk.id = si.sku_id  
                    LEFT JOIN brands b ON b.id = sk.brand_id
                    WHERE COALESCE(si.image_vec_768, si.image_vec) IS NOT NULL
                    ORDER BY COALESCE(si.image_vec_768, si.image_vec) <=> %s::vector
                    LIMIT %s
                """, (query_vector, query_vector, k))

                out = []
                if rows:
                    for rank, r in enumerate(rows, start=1):
                        sku_id, image_path, score, sku_name, brand_name, ocr_text = r
                        _q("INSERT INTO query_candidates(query_id, sku_id, rank, score) VALUES(%s,%s,%s,%s)",
                           (qid, sku_id, rank, score), fetch=None)
                        out.append({
                            "sku_id": sku_id,
                            "image_path": image_path,
                            "score": float(score) if score is not None else None,
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
    """Embed text thành vector 512D"""
    try:
        from utils import encode_texts
        emb = encode_texts([text])[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)
    except Exception as e:
        current_app.logger.warning("embed_text fallback: %s", e)
        # Không dùng random, trả None hoặc dùng fallback model khác để tăng chính xác
        return None  # Hoặc tích hợp model khác nếu cần

# Hàm hỗ trợ (nếu chưa có, thêm vào file)
def _vec_literal(vec):
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]" if vec else None

def _l2norm(v):
    import math
    n = math.sqrt(sum(float(x)*float(x) for x in v)) or 1.0
    return [float(x)/n for x in v]