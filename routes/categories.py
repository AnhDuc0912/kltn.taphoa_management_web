# routes/categories.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from services.db_utils import q, exec_sql

bp = Blueprint("categories_bp", __name__)

@bp.get("/categories", endpoint="categories")
def categories():
    rows = q("SELECT id, name, parent_id FROM categories ORDER BY name")
    return render_template("categories.html", rows=rows)

@bp.post("/categories", endpoint="category_create")
def category_create():
    name = (request.form.get("name") or "").strip()
    parent_id = request.form.get("parent_id") or None
    exec_sql("INSERT INTO categories(name, parent_id) VALUES(%s,%s)", (name, parent_id))
    flash("ÄÃ£ thÃªm category","success")
    return redirect(url_for("categories"))

@bp.post("/categories/<int:cid>/delete", endpoint="category_delete")
def category_delete(cid):
    exec_sql("DELETE FROM categories WHERE id=%s", (cid,))
    flash("ÄÃ£ xoÃ¡ category","success")
    return redirect(url_for("categories"))

