# routes/skus.py
import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash
from services.db_utils import q, exec_sql
from utils import vn_norm

bp = Blueprint("api_skus_bp", __name__)

@bp.get("/skus", endpoint="skus")
def skus():
    brand_id = request.args.get("brand_id")
    cat_id   = request.args.get("category_id")
    qtext    = (request.args.get("q") or "").strip()

    # Lấy cover + mô tả (description) mới nhất từ sku_search_corpus
    sql = """
    SELECT 
        s.id, s.name, s.variant, s.size_text,
        b.name AS brand, 
        c.name AS cat, 
        s.barcode, 
        s.is_active,
        (SELECT image_path FROM sku_images WHERE sku_id = s.id AND is_primary IS TRUE LIMIT 1) AS cover,
        -- mô tả: lấy text_content mới nhất nếu có updated_at, fallback theo id
        COALESCE(
            (
                SELECT corpus.text_content
                FROM sku_search_corpus corpus
                WHERE corpus.sku_id = s.id
                ORDER BY corpus.updated_at DESC NULLS LAST, corpus.id DESC
                LIMIT 1
            ), ''
        ) AS description
    FROM skus s
    LEFT JOIN brands b ON b.id = s.brand_id
    LEFT JOIN categories c ON c.id = s.category_id
    WHERE 1=1
    """
    params = []
    if brand_id:
        sql += " AND s.brand_id = %s"
        params.append(brand_id)
    if cat_id:
        sql += " AND s.category_id = %s"
        params.append(cat_id)
    if qtext:
        sql += " AND EXISTS (SELECT 1 FROM sku_texts st WHERE st.sku_id = s.id AND st.text ILIKE %s)"
        params.append(f"%{vn_norm(qtext)}%")
        sql += """
        AND EXISTS (
            SELECT 1 FROM sku_search_corpus c2
            WHERE c2.sku_id = s.id
              AND (
                    c2.search_vector @@ plainto_tsquery('simple', %s)
                 OR c2.text_content ILIKE %s
              )
        )
        """
        params.extend([vn_norm(qtext), f"%{vn_norm(qtext)}%"])

    sql += " ORDER BY s.updated_at DESC NULLS LAST, s.id DESC"

    rows   = q(sql, params)
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")

    # Chuẩn hoá schema cho Home/Cart:
    # { skuId, brandName, productName, productPrice, productSimilarity, imagePath, description }
    view_rows = []
    for r in rows:
        sku_id     = r[0]
        name       = (r[1] or "").strip()
        variant    = (r[2] or "").strip()
        size_text  = (r[3] or "").strip()
        brand_name = (r[4] or "").strip()
        cover      = (r[8] or "").strip()
        desc       = (r[9] or "").strip()

        parts = [name]
        if variant:   parts.append(variant)
        if size_text: parts.append(size_text)
        product_name = " ".join(p for p in parts if p)

        view_rows.append({
            "skuId": sku_id,
            "brandName": brand_name,      # giữ để tương thích ngược
            "productName": product_name,
            "productPrice": None,         # chưa join bảng giá
            "productSimilarity": None,    # không phải kết quả search ảnh
            "imagePath": cover,
            "description": desc           # <-- UI Home/Cart dùng trường này để hiển thị
        })

    return render_template("skus.html",
                           rows=view_rows,            # giờ rows có 'description'
                           brands=brands, cats=cats,
                           sel_brand=brand_id, sel_cat=cat_id, qtext=qtext,
                           captions_map={}, facets_map={}, colors_map={},
                           has_refresh_corpus=True, has_sku_texts=True)

