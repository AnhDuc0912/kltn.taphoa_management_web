# routes/skus.py
import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from services.db_utils import q, exec_sql, get_conn
from utils import vn_norm

bp = Blueprint("skus", "skus", url_prefix="/skus")

@bp.get("", endpoint="skus")
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
    
    if brand_id: 
        sql += " AND s.brand_id=%s"
        params.append(brand_id)
    
    if cat_id:   
        sql += " AND s.category_id=%s"
        params.append(cat_id)
    
    if qtext:
        # Normalize search text
        norm_q = vn_norm(qtext)
        
        # Search in sku_texts with ILIKE (simple text search)
        sql += " AND EXISTS (SELECT 1 FROM sku_texts st WHERE st.sku_id=s.id AND st.text ILIKE %s)"
        params.append(f"%{norm_q}%")
        
        # Search in sku_search_corpus using websearch_to_tsquery for better handling
        # websearch_to_tsquery handles phrases and special characters better than plainto_tsquery
        sql += """
        AND EXISTS (
            SELECT 1 FROM sku_search_corpus c
            WHERE c.sku_id=s.id AND (
                c.search_vector @@ websearch_to_tsquery('simple', %s) 
                OR c.text_content ILIKE %s
            )
        )
        """
        params.extend([norm_q, f"%{norm_q}%"])

    sql += " ORDER BY s.id DESC LIMIT 100"
    
    rows   = q(sql, params)
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")

    return render_template("skus.html", rows=rows, brands=brands, cats=cats,
                           sel_brand=brand_id, sel_cat=cat_id, qtext=qtext,
                           captions_map={}, facets_map={}, colors_map={},
                           has_refresh_corpus=True, has_sku_texts=True)


@bp.get("/new", endpoint="sku_new")
def sku_new():
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")
    return render_template("sku_form.html", sku=None, brands=brands, cats=cats)


@bp.post("", endpoint="sku_create")
def sku_create():
    brand_id = request.form.get("brand_id") or None
    cat_id   = request.form.get("category_id") or None
    name     = (request.form.get("name") or "").strip()
    variant  = request.form.get("variant") or None
    size_tx  = request.form.get("size_text") or None
    barcode  = request.form.get("barcode") or None
    active   = True if request.form.get("is_active") == "1" else False
    
    if not name:
        flash("Tên sản phẩm không được rỗng", "danger")
        return redirect(url_for("skus.sku_new"))
    
    row = exec_sql("""INSERT INTO skus(brand_id,category_id,name,variant,size_text,barcode,is_active)
                      VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                   (brand_id, cat_id, name, variant, size_tx, barcode, active), returning=True)
    sku_id = row[0]
    
    # Create normalized text for search
    brand_name = q("SELECT name FROM brands WHERE id=%s", (brand_id,), fetch="one")
    brand_name = brand_name[0] if brand_name else ""
    norm_text = vn_norm(" ".join([brand_name, name, variant or "", size_tx or ""]))
    exec_sql("INSERT INTO sku_texts(sku_id,text) VALUES(%s,%s) ON CONFLICT DO NOTHING", (sku_id, norm_text))
    
    flash("Đã tạo SKU", "success")
    return redirect(url_for("skus.sku_edit", sku_id=sku_id))


@bp.post("/bulk", endpoint="sku_bulk_create")
def sku_bulk_create():
    """
    Body JSON:
    {
      "items": [
        {
          "brand_id": 1,                # hoặc null
          "category_id": 3507,          # hoặc null
          "name": "Xúc xích heo Ponies",
          "variant": "5 cây",
          "size_text": "175g",
          "barcode": "1052852000188",
          "is_active": true
        },
        ...
      ],
      "dedupe": true,                    # optional: true => bỏ qua nếu trùng
      "dedupe_keys": ["brand_id","name","variant","size_text"]  # optional: tiêu chí trùng
    }
    """
    try:
        payload = request.get_json(silent=True) or {}
        items   = payload.get("items") or []
        
        if not isinstance(items, list) or not items:
            return jsonify({"ok": False, "message": "items must be a non-empty array"}), 400

        dedupe = bool(payload.get("dedupe", True))
        dedupe_keys = payload.get("dedupe_keys") or ["brand_id", "name", "variant", "size_text"]

        # Giới hạn batch
        if len(items) > 2000:
            return jsonify({"ok": False, "message": "Too many items (max 2000 per request)"}), 413

        created = []
        errors  = []

        with get_conn() as conn, conn.cursor() as cur:
            for idx, raw in enumerate(items):
                # Extract + sanitize
                brand_id = raw.get("brand_id")
                cat_id   = raw.get("category_id")
                name     = (raw.get("name") or "").strip()
                variant  = (raw.get("variant") or None)
                size_tx  = (raw.get("size_text") or None)
                barcode  = (raw.get("barcode") or None)
                active   = bool(raw.get("is_active", True))

                if not name:
                    errors.append({"index": idx, "error": "name is required"})
                    continue

                # Dedupe check
                if dedupe:
                    where_clauses = []
                    params = []

                    for k in dedupe_keys:
                        if k == "brand_id":
                            if brand_id is None:
                                where_clauses.append("brand_id IS NULL")
                            else:
                                where_clauses.append("brand_id = %s")
                                params.append(brand_id)
                        elif k == "category_id":
                            if cat_id is None:
                                where_clauses.append("category_id IS NULL")
                            else:
                                where_clauses.append("category_id = %s")
                                params.append(cat_id)
                        elif k == "name":
                            where_clauses.append("LOWER(name) = LOWER(%s)")
                            params.append(name)
                        elif k == "variant":
                            if variant is None:
                                where_clauses.append("variant IS NULL")
                            else:
                                where_clauses.append("variant = %s")
                                params.append(variant)
                        elif k == "size_text":
                            if size_tx is None:
                                where_clauses.append("size_text IS NULL")
                            else:
                                where_clauses.append("size_text = %s")
                                params.append(size_tx)
                        elif k == "barcode":
                            if barcode is None:
                                where_clauses.append("barcode IS NULL")
                            else:
                                where_clauses.append("barcode = %s")
                                params.append(barcode)

                    if where_clauses:
                        sql_check = "SELECT id FROM skus WHERE " + " AND ".join(where_clauses) + " LIMIT 1"
                        cur.execute(sql_check, params)
                        row = cur.fetchone()
                        if row:
                            created.append({"index": idx, "sku_id": row[0], "deduped": True})
                            continue

                # Insert SKU
                try:
                    sql = """
                        INSERT INTO skus(brand_id, category_id, name, variant, size_text, barcode, is_active)
                        VALUES(%s,%s,%s,%s,%s,%s,%s)
                        RETURNING id
                    """
                    cur.execute(sql, (brand_id, cat_id, name, variant, size_tx, barcode, active))
                    sku_id = cur.fetchone()[0]

                    # Get brand name for normalization
                    brand_name = ""
                    if brand_id:
                        cur.execute("SELECT name FROM brands WHERE id=%s", (brand_id,))
                        br = cur.fetchone()
                        brand_name = br[0] if br else ""

                    norm_text = vn_norm(" ".join([brand_name, name, variant or "", size_tx or ""]))
                    cur.execute(
                        "INSERT INTO sku_texts(sku_id, text) VALUES(%s, %s) ON CONFLICT DO NOTHING",
                        (sku_id, norm_text)
                    )
                    created.append({"index": idx, "sku_id": sku_id, "deduped": False})
                    
                except Exception as e:
                    errors.append({"index": idx, "error": str(e)})

            conn.commit()

        return jsonify({
            "ok": True,
            "created_count": len([x for x in created if not x.get("deduped")]),
            "deduped_count": len([x for x in created if x.get("deduped")]),
            "error_count": len(errors),
            "created": created,
            "errors": errors
        })
        
    except Exception as ex:
        return jsonify({"ok": False, "message": "bulk insert failed", "error": str(ex)}), 500


@bp.get("/<int:sku_id>/edit", endpoint="sku_edit")
def sku_edit(sku_id):
    sku = q("""SELECT id,brand_id,category_id,name,variant,size_text,barcode,is_active
               FROM skus WHERE id=%s""", (sku_id,), fetch="one")
    if not sku:
        flash("Không tìm thấy SKU", "danger")
        return redirect(url_for("skus.skus"))
    
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")
    return render_template("sku_form.html", sku=sku, brands=brands, cats=cats)


@bp.post("/<int:sku_id>", endpoint="sku_update")
def sku_update(sku_id):
    brand_id = request.form.get("brand_id") or None
    cat_id   = request.form.get("category_id") or None
    name     = (request.form.get("name") or "").strip()
    variant  = request.form.get("variant") or None
    size_tx  = request.form.get("size_text") or None
    barcode  = request.form.get("barcode") or None
    active   = True if request.form.get("is_active") == "1" else False
    
    exec_sql("""UPDATE skus SET brand_id=%s, category_id=%s, name=%s, variant=%s,
                              size_text=%s, barcode=%s, is_active=%s, updated_at=%s
                WHERE id=%s""",
             (brand_id, cat_id, name, variant, size_tx, barcode, active, datetime.datetime.utcnow(), sku_id))

    brand_name = q("SELECT name FROM brands WHERE id=%s", (brand_id,), fetch="one")
    brand_name = brand_name[0] if brand_name else ""
    norm_text  = vn_norm(" ".join([brand_name, name, variant or "", size_tx or ""]))
    
    existing   = q("SELECT id FROM sku_texts WHERE sku_id=%s LIMIT 1", (sku_id,), fetch="one")
    if existing:
        exec_sql("UPDATE sku_texts SET text=%s WHERE id=%s", (norm_text, existing[0]))
    else:
        exec_sql("INSERT INTO sku_texts(sku_id,text) VALUES(%s,%s)", (sku_id, norm_text))

    flash("Đã cập nhật SKU", "success")
    return redirect(url_for("skus.skus"))


@bp.post("/<int:sku_id>/delete", endpoint="sku_delete")
def sku_delete(sku_id):
    exec_sql("DELETE FROM skus WHERE id=%s", (sku_id,))
    flash("Đã xóa SKU", "success")
    return redirect(url_for("skus.skus"))

