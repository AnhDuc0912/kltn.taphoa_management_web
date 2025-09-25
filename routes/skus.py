# routes/skus.py
import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash
from services.db_utils import q, exec_sql
from utils import vn_norm

bp = Blueprint("skus_bp", __name__)

@bp.get("/skus", endpoint="skus")
def skus():
    brand_id = request.args.get("brand_id")
    cat_id   = request.args.get("category_id")
    qtext    = (request.args.get("q") or "").strip()

    sql = """
    SELECT s.id, s.name, s.variant, s.size_text, b.name AS brand, c.name AS cat, s.barcode, s.is_active,
           (SELECT image_path FROM sku_images WHERE sku_id=s.id AND is_primary IS TRUE LIMIT 1) AS cover
    FROM skus s
    LEFT JOIN brands b ON b.id=s.brand_id
    LEFT JOIN categories c ON c.id=s.category_id
    WHERE 1=1
    """
    params = []
    if brand_id: sql += " AND s.brand_id=%s"; params.append(brand_id)
    if cat_id:   sql += " AND s.category_id=%s"; params.append(cat_id)
    if qtext:
        sql += " AND EXISTS (SELECT 1 FROM sku_texts st WHERE st.sku_id=s.id AND st.text ILIKE %s)"
        params.append(f"%{vn_norm(qtext)}%")
        # Search in sku_search_corpus using correct column names
        sql += """
        AND EXISTS (
            SELECT 1 FROM sku_search_corpus c
            WHERE c.sku_id=s.id AND
                  (c.search_vector @@ plainto_tsquery('simple', %s) OR c.text_content ILIKE %s)
        )
        """
        params.extend([vn_norm(qtext), f"%{vn_norm(qtext)}%"])
    else:
        # No search query - don't filter by search corpus
        pass

    rows   = q(sql, params)
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")

    return render_template("skus.html", rows=rows, brands=brands, cats=cats,
                           sel_brand=brand_id, sel_cat=cat_id, qtext=qtext,
                           captions_map={}, facets_map={}, colors_map={},
                           has_refresh_corpus=True, has_sku_texts=True)

@bp.get("/skus/new", endpoint="sku_new")
def sku_new():
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")
    return render_template("sku_form.html", sku=None, brands=brands, cats=cats)

@bp.post("/skus", endpoint="sku_create")
def sku_create():
    brand_id = request.form.get("brand_id") or None
    cat_id   = request.form.get("category_id") or None
    name     = (request.form.get("name") or "").strip()
    variant  = request.form.get("variant") or None
    size_tx  = request.form.get("size_text") or None
    barcode  = request.form.get("barcode") or None
    active   = True if request.form.get("is_active")=="1" else False
    if not name:
        flash("TÃªn sáº£n pháº©m khÃ´ng Ä‘Æ°á»£c rá»—ng","danger")
        return redirect(url_for("sku_new"))
    row = exec_sql("""INSERT INTO skus(brand_id,category_id,name,variant,size_text,barcode,is_active)
                      VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                   (brand_id,cat_id,name,variant,size_tx,barcode,active), returning=True)
    sku_id = row[0]
    # sku_text chuáº©n hoÃ¡
    brand_name = q("SELECT name FROM brands WHERE id=%s", (brand_id,), fetch="one")
    brand_name = brand_name[0] if brand_name else ""
    norm_text = vn_norm(" ".join([brand_name, name, variant or "", size_tx or ""]))
    exec_sql("INSERT INTO sku_texts(sku_id,text) VALUES(%s,%s) ON CONFLICT DO NOTHING", (sku_id, norm_text))
    flash("ÄÃ£ táº¡o SKU","success")
    return redirect(url_for("sku_edit", sku_id=sku_id))

@bp.get("/skus/<int:sku_id>/edit", endpoint="sku_edit")
def sku_edit(sku_id):
    sku = q("""SELECT id,brand_id,category_id,name,variant,size_text,barcode,is_active
               FROM skus WHERE id=%s""", (sku_id,), fetch="one")
    if not sku:
        flash("KhÃ´ng tÃ¬m tháº¥y SKU","danger")
        return redirect(url_for("skus"))
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")
    return render_template("sku_form.html", sku=sku, brands=brands, cats=cats)

@bp.post("/skus/<int:sku_id>", endpoint="sku_update")
def sku_update(sku_id):
    brand_id = request.form.get("brand_id") or None
    cat_id   = request.form.get("category_id") or None
    name     = (request.form.get("name") or "").strip()
    variant  = request.form.get("variant") or None
    size_tx  = request.form.get("size_text") or None
    barcode  = request.form.get("barcode") or None
    active   = True if request.form.get("is_active")=="1" else False
    exec_sql("""UPDATE skus SET brand_id=%s, category_id=%s, name=%s, variant=%s,
                              size_text=%s, barcode=%s, is_active=%s, updated_at=%s
                WHERE id=%s""",
             (brand_id,cat_id,name,variant,size_tx,barcode,active,datetime.datetime.utcnow(),sku_id))

    brand_name = q("SELECT name FROM brands WHERE id=%s", (brand_id,), fetch="one")
    brand_name = brand_name[0] if brand_name else ""
    norm_text  = vn_norm(" ".join([brand_name, name, variant or "", size_tx or ""]))
    existing   = q("SELECT id FROM sku_texts WHERE sku_id=%s LIMIT 1", (sku_id,), fetch="one")
    if existing:
        exec_sql("UPDATE sku_texts SET text=%s WHERE id=%s", (norm_text, existing[0]))
    else:
        exec_sql("INSERT INTO sku_texts(sku_id,text) VALUES(%s,%s)", (sku_id, norm_text))

    flash("ÄÃ£ cáº­p nháº­t SKU","success")
    return redirect(url_for("skus"))

@bp.post("/skus/<int:sku_id>/delete", endpoint="sku_delete")
def sku_delete(sku_id):
    exec_sql("DELETE FROM skus WHERE id=%s", (sku_id,))
    flash("ÄÃ£ xoÃ¡ SKU","success")
    return redirect(url_for("skus"))

