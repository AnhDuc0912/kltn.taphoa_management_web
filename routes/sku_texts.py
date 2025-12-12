# routes/sku_texts.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
import os
from services.db_utils import q, exec_sql
from db import get_conn

bp = Blueprint("sku_texts_bp", __name__)

@bp.get("/skus/<int:sku_id>/texts", endpoint="sku_texts_page")
def sku_texts_page(sku_id):
    sku = q("SELECT id,name FROM skus WHERE id=%s", (sku_id,), fetch="one")
    if not sku:
        flash("KhÃ´ng tÃ¬m tháº¥y SKU","danger")
        return redirect(url_for("skus.skus"))
    rows = q("SELECT id, text FROM sku_texts WHERE sku_id=%s ORDER BY id", (sku_id,))

    # Primary image + OCR
    img_row = q("SELECT image_path, ocr_text FROM sku_images WHERE sku_id=%s ORDER BY is_primary DESC, id LIMIT 1", (sku_id,), fetch="one")
    cover_url = None
    ocr_text = None
    if img_row:
        image_path, ocr_text = img_row
        upload_folder = os.getenv("UPLOAD_FOLDER", "uploads").strip().strip("/")
        app_host = os.getenv("APP_HOST", "").strip().rstrip("/")
        if image_path:
            if app_host:
                cover_url = f"{app_host}/{upload_folder}/{image_path}"
            else:
                cover_url = f"/{upload_folder}/{image_path}"

    # Latest accepted caption (approved)
    cap_row = q("SELECT caption_text FROM sku_captions WHERE sku_id=%s AND needs_review=FALSE ORDER BY created_at DESC LIMIT 1", (sku_id,), fetch="one")
    caption_latest_text = cap_row[0] if cap_row else None

    # Captions list for UI (include semantic fields)
    rows_captions_raw = q("""
        SELECT c.id, c.sku_id, c.image_path, c.style, c.caption_text, c.model_name, c.prompt_version,
               c.needs_review, c.is_ground_truth, c.created_at, c.keywords, c.colors, c.shapes, c.materials,
               c.packaging, c.taste, c.texture, c.brand_guess, c.variant_guess, c.size_guess, c.category_guess,
               si.ocr_text
        FROM sku_captions c
        LEFT JOIN sku_images si ON si.sku_id = c.sku_id AND si.image_path = c.image_path
        WHERE c.sku_id=%s
        ORDER BY c.image_path, c.style, c.created_at DESC
    """, (sku_id,)) or []

    # Normalize rows to list of dicts for template convenience
    rows_captions = []
    col_names = [
        "id","sku_id","image_path","style","caption_text","model_name","prompt_version",
        "needs_review","is_ground_truth","created_at","keywords","colors","shapes","materials",
        "packaging","taste","texture","brand_guess","variant_guess","size_guess","category_guess",
        "ocr_text"
    ]
    for r in rows_captions_raw:
        if hasattr(r, "_asdict"):
            rows_captions.append(dict(r._asdict()))
        else:
            # r is a tuple
            rows_captions.append({col_names[i]: r[i] if i < len(r) else None for i in range(len(col_names))})

    # Derive simple facet lists from captions
    sku_facets = []
    colors = []
    try:
        kws = set()
        cols = set()
        for r in (rows_captions or []):
            k = r.get("keywords")
            c = r.get("colors")
            if k:
                for x in k:
                    kws.add(x)
            if c:
                for x in c:
                    cols.add(x)
        sku_facets = sorted(kws)
        colors = sorted(cols)
    except Exception:
        sku_facets = []
        colors = []

    return render_template("sku_texts.html", sku=sku, rows=rows,
                           cover_url=cover_url, ocr_text=ocr_text,
                           caption_latest_text=caption_latest_text,
                           rows_captions=rows_captions,
                           sku_facets=sku_facets, colors=colors)

@bp.post("/skus/<int:sku_id>/texts", endpoint="sku_texts_add")
def sku_texts_add(sku_id):
    txt = (request.form.get("text") or "").strip()
    if not txt:
        flash("Chưa nhập mô tả", "danger")
        return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))
    
    try:
        from utils import vn_norm
        norm = vn_norm(txt)
        exec_sql("INSERT INTO sku_texts(sku_id, text) VALUES(%s, %s)", (sku_id, norm))
        flash("Đã thêm mô tả", "success")
        current_app.logger.info("Added sku_text for sku_id=%s", sku_id)
    except Exception as e:
        flash(f"Lỗi: {e}", "danger")
        current_app.logger.exception("Error adding sku_text for sku_id=%s", sku_id)
    
    return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/texts/<int:text_id>/update", endpoint="sku_texts_update")
def sku_texts_update(sku_id, text_id):
    txt = (request.form.get("text") or "").strip()
    if not txt:
        flash("Chưa nhập mô tả", "danger")
        return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))
    
    try:
        from utils import vn_norm
        norm = vn_norm(txt)
        # Update text và clear vector để force re-backfill
        exec_sql("UPDATE sku_texts SET text=%s, text_vec=NULL WHERE id=%s AND sku_id=%s", 
                (norm, text_id, sku_id))
        flash("Đã lưu mô tả", "success")
        current_app.logger.info("Updated sku_text id=%s", text_id)
    except Exception as e:
        flash(f"Lỗi: {e}", "danger")
        current_app.logger.exception("Error updating sku_text id=%s", text_id)
    
    return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/texts/<int:text_id>/delete", endpoint="sku_texts_delete")
def sku_texts_delete(sku_id, text_id):
    exec_sql("DELETE FROM sku_texts WHERE id=%s AND sku_id=%s", (text_id, sku_id))
    flash("ÄÃ£ xoÃ¡ mÃ´ táº£","success")
    return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))

