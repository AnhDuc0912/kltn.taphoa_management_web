# routes/sku_texts.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from services.db_utils import q, exec_sql
from db import get_conn

bp = Blueprint("sku_texts_bp", __name__)

@bp.get("/skus/<int:sku_id>/texts", endpoint="sku_texts_page")
def sku_texts_page(sku_id):
    sku = q("SELECT id,name FROM skus WHERE id=%s", (sku_id,), fetch="one")
    if not sku:
        flash("KhÃ´ng tÃ¬m tháº¥y SKU","danger")
        return redirect(url_for("skus_bp.skus"))
    rows = q("SELECT id, text FROM sku_texts WHERE sku_id=%s ORDER BY id", (sku_id,))
    return render_template("sku_texts.html", sku=sku, rows=rows)

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

