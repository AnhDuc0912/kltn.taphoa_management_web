# routes/sku_images.py
import os, uuid
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from services.db_utils import q, exec_sql

bp = Blueprint("sku_images_bp", __name__)

@bp.get("/skus/<int:sku_id>/images", endpoint="sku_images")
def sku_images(sku_id):
    sku  = q("SELECT id,name FROM skus WHERE id=%s", (sku_id,), fetch="one")
    imgs = q("SELECT id, image_path, is_primary FROM sku_images WHERE sku_id=%s ORDER BY id DESC", (sku_id,))
    return render_template("sku_images.html", sku=sku, imgs=imgs)

@bp.post("/skus/<int:sku_id>/images", endpoint="sku_image_add")
def sku_image_add(sku_id):
    f = request.files.get("image")
    is_primary = request.form.get("is_primary") == "1"
    if not f:
        flash("ChÆ°a chá»n áº£nh", "danger")
        return redirect(url_for("sku_images", sku_id=sku_id))

    ext   = os.path.splitext(f.filename)[1].lower()
    fname = secure_filename(f"{sku_id}_{uuid.uuid4().hex}{ext}")
    fpath = os.path.join(current_app.config["UPLOAD_DIR"], fname)
    f.save(fpath)

    if is_primary:
        exec_sql("UPDATE sku_images SET is_primary=FALSE WHERE sku_id=%s", (sku_id,))
    exec_sql("INSERT INTO sku_images(sku_id,image_path,is_primary) VALUES(%s,%s,%s)", (sku_id, fname, is_primary))
    flash("ÄÃ£ thÃªm áº£nh","success")
    return redirect(url_for("sku_images", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/images/<int:img_id>/set-primary", endpoint="sku_image_set_primary")
def sku_image_set_primary(sku_id, img_id):
    exec_sql("UPDATE sku_images SET is_primary=FALSE WHERE sku_id=%s", (sku_id,))
    exec_sql("UPDATE sku_images SET is_primary=TRUE WHERE id=%s", (img_id,))
    flash("ÄÃ£ Ä‘áº·t áº£nh chÃ­nh","success")
    return redirect(url_for("sku_images", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/images/<int:img_id>/delete", endpoint="sku_image_delete")
def sku_image_delete(sku_id, img_id):
    row = q("SELECT image_path FROM sku_images WHERE id=%s", (img_id,), fetch="one")
    if row and row[0]:
        fpath = row[0]
        if not os.path.isabs(fpath):
            fpath = os.path.join(current_app.config["UPLOAD_DIR"], fpath)
        if os.path.exists(fpath):
            try: os.remove(fpath)
            except: pass
    exec_sql("DELETE FROM sku_images WHERE id=%s", (img_id,))
    flash("ÄÃ£ xoÃ¡ áº£nh","success")
    return redirect(url_for("sku_images", sku_id=sku_id))

