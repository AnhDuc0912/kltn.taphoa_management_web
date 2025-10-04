# routes/sku_images.py
import os
import uuid
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from services.db_utils import q, exec_sql, get_connection
from services.resnet101 import extract_embedding, load_model  # Import từ resnet101.py
from psycopg2.extras import execute_values

bp = Blueprint("sku_images_bp", __name__)

@bp.get("/skus/<int:sku_id>/images", endpoint="sku_images")
def sku_images(sku_id):
    sku = q("SELECT id, name FROM skus WHERE id=%s", (sku_id,), fetch="one")
    imgs = q("SELECT id, image_path, is_primary FROM sku_images WHERE sku_id=%s ORDER BY id DESC", (sku_id,))
    return render_template("sku_images.html", sku=sku, imgs=imgs)

@bp.post("/skus/<int:sku_id>/images", endpoint="sku_image_add")
def sku_image_add(sku_id):
    f = request.files.get("image")
    is_primary = request.form.get("is_primary") == "1"
    if not f:
        flash("Chưa chọn ảnh", "danger")
        return redirect(url_for("sku_images", sku_id=sku_id))

    ext = os.path.splitext(f.filename)[1].lower()
    fname = secure_filename(f"{sku_id}_{uuid.uuid4().hex}{ext}")
    fpath = os.path.join(current_app.config["UPLOAD_DIR"], fname)
    f.save(fpath)

    if is_primary:
        exec_sql("UPDATE sku_images SET is_primary=FALSE WHERE sku_id=%s", (sku_id,))
    exec_sql("INSERT INTO sku_images(sku_id, image_path, is_primary) VALUES(%s, %s, %s)", (sku_id, fname, is_primary))

    # Backfill vector ngay sau khi upload
    model = load_model()  # Load model từ resnet101.py
    vector = extract_embedding(model, fpath)
    if vector:
        exec_sql("UPDATE sku_images SET image_vec = %s::vector WHERE image_path = %s", (vector, fname))

    flash("Đã thêm ảnh", "success")
    return redirect(url_for("sku_images", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/images/<int:img_id>/set-primary", endpoint="sku_image_set_primary")
def sku_image_set_primary(sku_id, img_id):
    exec_sql("UPDATE sku_images SET is_primary=FALSE WHERE sku_id=%s", (sku_id,))
    exec_sql("UPDATE sku_images SET is_primary=TRUE WHERE id=%s", (img_id,))
    flash("Đã đặt ảnh chính", "success")
    return redirect(url_for("sku_images", sku_id=sku_id))

@bp.post("/skus/<int:sku_id>/images/<int:img_id>/delete", endpoint="sku_image_delete")
def sku_image_delete(sku_id, img_id):
    row = q("SELECT image_path FROM sku_images WHERE id=%s", (img_id,), fetch="one")
    if row and row[0]:
        fpath = row[0]
        if not os.path.isabs(fpath):
            fpath = os.path.join(current_app.config["UPLOAD_DIR"], fpath)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass
    exec_sql("DELETE FROM sku_images WHERE id=%s", (img_id,))
    flash("Đã xóa ảnh", "success")
    return redirect(url_for("sku_images", sku_id=sku_id))

# API để backfill vector cho tất cả ảnh chưa có vector
@bp.route('/backfill_vectors', methods=['GET'])
def backfill_vectors():
    model = load_model()  # Load model từ resnet101.py
    conn = get_connection()
    cursor = conn.cursor()

    # Lấy danh sách ảnh chưa có vector
    cursor.execute("SELECT id, sku_id, image_path FROM sku_images WHERE image_vec IS NULL")
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return jsonify({"message": "No images to backfill"}), 200

    base_path = current_app.config["UPLOAD_DIR"]
    updates = []
    for row in rows:
        img_id, sku_id, img_path = row
        full_path = os.path.join(base_path, img_path)
        if os.path.exists(full_path):
            vector = extract_embedding(model, full_path)
            if vector:
                updates.append((vector, img_id))

    if updates:
        # Bulk update using execute_values: (vector_text, id)
        execute_values(
            cursor,
            "UPDATE sku_images AS s SET image_vec = v.vector::vector FROM (VALUES %s) AS v(vector, id) WHERE s.id = v.id",
            [(str(v), img_id) for v, img_id in updates],
            template="(%s, %s)"
        )
        conn.commit()
        conn.close()
        return jsonify({"message": f"Backfilled {len(updates)} vectors", "success": True}), 200
    else:
        conn.close()
        return jsonify({"message": "No vectors processed due to errors", "success": False}), 500

@bp.route('/reset_vectors', methods=['GET'])
def reset_vectors():
    """
    Reset tất cả vector trong cột image_vec của bảng sku_images về NULL.
    Yêu cầu phương thức POST để tránh vô tình reset.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Cập nhật tất cả image_vec thành NULL
        cursor.execute("UPDATE sku_images SET image_vec = NULL")
        affected_rows = cursor.rowcount

        conn.commit()
        conn.close()
        
        return jsonify({
            "message": f"Reset {affected_rows} image vectors to NULL",
            "success": True,
            "affected_rows": affected_rows
        }), 200

    except Exception as e:
        conn.rollback()
        conn.close()
        current_app.logger.exception(f"Reset vectors error: {str(e)}")
        return jsonify({"message": f"Error resetting vectors: {str(e)}", "success": False}), 500