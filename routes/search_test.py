from flask import Blueprint, render_template, request, jsonify, current_app
from services.db_utils import q
from db import get_conn
from werkzeug.utils import secure_filename
import io, time, os
import tempfile
from PIL import Image
import numpy as np
import re
import unicodedata
import sys
from utils import encode_texts, encode_images  # DO NOT import vn_norm here

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

bp = Blueprint("search_test_bp", __name__)

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
    """Trang test search UI"""
    return render_template("search_test.html")

@bp.post("/search/text")
def search_text():
    data = request.get_json(silent=True) or {}
    qtext = vn_norm((data.get("q") or "").strip())
    k = min(int(data.get("k") or 20), 100)
    if not qtext:
        return jsonify({"error": "missing q"}), 400

    try:
        t0 = time.time()
        try:
            vec = encode_texts([qtext])[0].tolist()
        except Exception as e:
            current_app.logger.exception("encode_texts failed")
            return jsonify({"error": f"encode_texts failed: {e}", "results": []}), 500

        # log query (use local vn_norm)
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

        # Save temp file và extract features
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)

            try:
                img = Image.open(tmp.name).convert('RGB')

                # Use real encoder if available
                try:
                    query_vector = encode_images([img])[0].tolist()
                except Exception:
                    # fallback (for testing) to random vector
                    current_app.logger.warning("encode_images failed, using random vector fallback")
                    query_vector = np.random.rand(512).tolist()

                # Vector similarity search trong sku_images
                rows = q("""
                    SELECT 
                        si.sku_id,
                        si.image_path,
                        1 - (si.image_vector <=> %s::vector) as score,
                        sk.name as sku_name,
                        b.name as brand_name,
                        si.ocr_text
                    FROM sku_images si
                    LEFT JOIN skus sk ON sk.id = si.sku_id  
                    LEFT JOIN brands b ON b.id = sk.brand_id
                    WHERE si.image_vector IS NOT NULL
                    ORDER BY si.image_vector <=> %s::vector
                    LIMIT %s
                """, (query_vector, query_vector, k))

                out = []
                if rows:
                    for r in rows:
                        out.append({
                            "sku_id": r[0],
                            "image_path": r[1],
                            "score": float(r[2]) if r[2] is not None else None,
                            "sku_name": r[3],
                            "brand_name": r[4],
                            "ocr_text": r[5]
                        })

                return jsonify({
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