from flask import Blueprint, render_template, request, jsonify, current_app
from services.db_utils import q
from werkzeug.utils import secure_filename
import os
import tempfile
from PIL import Image
import numpy as np

bp = Blueprint("search_test_bp", __name__)

@bp.get("/search-test")
def search_test_page():
    """Trang test search UI"""
    return render_template("search_test.html")

@bp.post("/search/text")
def search_text():
    """API search bằng text/caption"""
    try:
        data = request.get_json() or {}
        query = (data.get("q") or "").strip()
        k = min(int(data.get("k", 20)), 100)
        
        if not query:
            return jsonify({"error": "Missing query", "results": []})

        # Search trong sku_search_corpus với full-text + vector similarity
        results = q("""
            WITH text_search AS (
                SELECT DISTINCT
                    sku_id,
                    text_content as text,
                    ts_rank_cd(search_vector, plainto_tsquery('simple', %s)) as text_score,
                    image_path
                FROM sku_search_corpus 
                WHERE search_vector @@ plainto_tsquery('simple', %s)
                   OR text_content ILIKE %s
                ORDER BY text_score DESC
                LIMIT %s
            )
            SELECT 
                s.sku_id,
                s.text,
                s.text_score as score,
                s.image_path,
                sk.name as sku_name,
                b.name as brand_name
            FROM text_search s
            LEFT JOIN skus sk ON sk.id = s.sku_id
            LEFT JOIN brands b ON b.id = sk.brand_id
            ORDER BY s.text_score DESC
        """, (query, query, f"%{query}%", k))

        return jsonify({
            "query": query,
            "total": len(results),
            "results": [dict(r._asdict()) for r in results] if results else []
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
                # Load image và tạo embedding (giả sử có hàm tạo vector)
                img = Image.open(tmp.name).convert('RGB')
                
                # TODO: Thay bằng model embedding thực tế (CLIP, etc.)
                # Tạm thời dùng vector random để demo
                query_vector = np.random.rand(512).tolist()
                
                # Vector similarity search trong sku_images
                results = q("""
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

                return jsonify({
                    "filename": secure_filename(file.filename),
                    "total": len(results),
                    "results": [dict(r._asdict()) for r in results] if results else []
                })

            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp.name)
                except:
                    pass

    except Exception as e:
        current_app.logger.exception("Image search error")
        return jsonify({"error": str(e), "results": []}), 500