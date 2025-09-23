# routes/brands.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from services.db_utils import q, exec_sql

bp = Blueprint("brands_bp", __name__)

@bp.get("/brands", endpoint="brands")
def brands():
    rows = q("SELECT id, name FROM brands ORDER BY name")
    return render_template("brands.html", rows=rows)

@bp.post("/brands", endpoint="brand_create")
def brand_create():
    name = (request.form.get("name") or "").strip()
    if not name:
        flash("TÃªn brand khÃ´ng Ä‘Æ°á»£c rá»—ng","danger")
        return redirect(url_for("brands"))
    exec_sql("INSERT INTO brands(name) VALUES(%s) ON CONFLICT(name) DO NOTHING", (name,))
    flash("ÄÃ£ thÃªm brand","success")
    return redirect(url_for("brands"))

@bp.post("/brands/<int:bid>/delete", endpoint="brand_delete")
def brand_delete(bid):
    exec_sql("DELETE FROM brands WHERE id=%s", (bid,))
    flash("ÄÃ£ xoÃ¡ brand","success")
    return redirect(url_for("brands"))

