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
    text = (request.form.get("text") or "").strip()
    if not text:
        flash("ChÆ°a nháº­p mÃ´ táº£.", "warning")
        return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))

    # náº¿u cÃ³ hÃ m chuáº©n hoÃ¡ trong project, gá»i á»Ÿ Ä‘Ã¢y; fallback dÃ¹ng text tháº³ng
    try:
        normalize = globals().get("vn_norm") or (lambda s: s)
        normed = normalize(text)

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sku_texts (sku_id, text, normalized_text, created_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (sku_id, text) DO NOTHING
                """,
                (sku_id, text, normed),
            )
            if cur.rowcount and cur.rowcount > 0:
                flash("ÄÃ£ thÃªm mÃ´ táº£.", "success")
            else:
                flash("MÃ´ táº£ trÃ¹ng láº·p â€” Ä‘Ã£ bá» qua.", "info")
            conn.commit()
    except Exception as e:
        current_app.logger.exception("Lá»—i khi thÃªm sku_texts")
        flash("Lá»—i khi thÃªm mÃ´ táº£.", "danger")

    return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/texts/<int:text_id>/update", endpoint="sku_texts_update")
def sku_texts_update(sku_id, text_id):
    txt = (request.form.get("text") or "").strip()
    if not txt:
        flash("ChÆ°a nháº­p mÃ´ táº£","danger")
        return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))
    # normalize using available normalizer if present
    norm = globals().get("vn_norm") or (lambda s: s)
    exec_sql("UPDATE sku_texts SET text=%s, normalized_text=%s WHERE id=%s AND sku_id=%s", (txt, norm(txt), text_id, sku_id))
    flash("ÄÃ£ lÆ°u mÃ´ táº£","success")
    return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/texts/<int:text_id>/delete", endpoint="sku_texts_delete")
def sku_texts_delete(sku_id, text_id):
    exec_sql("DELETE FROM sku_texts WHERE id=%s AND sku_id=%s", (text_id, sku_id))
    flash("ÄÃ£ xoÃ¡ mÃ´ táº£","success")
    return redirect(url_for("sku_texts_bp.sku_texts_page", sku_id=sku_id))

