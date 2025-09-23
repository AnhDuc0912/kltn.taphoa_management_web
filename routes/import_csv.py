# routes/import_csv.py
import os, csv, uuid
from io import StringIO
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, Response, current_app
from services.db_utils import q, exec_sql
from utils import vn_norm

bp = Blueprint("import_bp", __name__)

@bp.get("/import", endpoint="import_csv_page")
def import_csv_page():
    brands = q("SELECT id, name FROM brands ORDER BY name")
    cats   = q("SELECT id, name FROM categories ORDER BY name")
    return render_template("import_csv.html", brands=brands, cats=cats)

@bp.get("/import/sample", endpoint="import_csv_sample")
def import_csv_sample():
    sample = StringIO()
    sample.write("brand,category,name,variant,size_text,barcode,image_file\n")
    sample.write("Háº£o Háº£o,MÃ¬ gÃ³i,MÃ¬ Háº£o Háº£o,TÃ´m chua cay,75g,893456...,/path/to/haohao.jpg\n")
    sample.seek(0)
    return Response(sample.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment;filename=sample_import.csv"})

@bp.post("/import", endpoint="import_csv")
def import_csv():
    file = request.files.get("file")
    if not file:
        flash("ChÆ°a chá»n file CSV","danger")
        return redirect(url_for("import_csv_page"))

    reader = csv.DictReader(file.stream.read().decode("utf-8").splitlines())
    rows_added = 0
    for r in reader:
        brand = (r.get("brand") or "").strip()
        cat   = (r.get("category") or "").strip()
        name  = (r.get("name") or "").strip()
        if not name: continue
        variant = (r.get("variant") or "").strip() or None
        size_tx = (r.get("size_text") or "").strip() or None
        barcode = (r.get("barcode") or "").strip() or None
        image_file = (r.get("image_file") or "").strip()

        b = exec_sql("INSERT INTO brands(name) VALUES(%s) ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name RETURNING id", (brand,), returning=True) if brand else (None,)
        brand_id = b[0] if b else None
        c = exec_sql("INSERT INTO categories(name) VALUES(%s) ON CONFLICT DO NOTHING RETURNING id", (cat,), returning=True) if cat else (None,)
        cat_id = c[0] if c else None

        s = exec_sql("""INSERT INTO skus(brand_id,category_id,name,variant,size_text,barcode,is_active)
                        VALUES(%s,%s,%s,%s,%s,%s,TRUE) RETURNING id""",
                     (brand_id,cat_id,name,variant,size_tx,barcode), returning=True)
        sku_id = s[0]

        norm = vn_norm(" ".join([brand, name, variant or "", size_tx or ""]))
        exec_sql("INSERT INTO sku_texts(sku_id,text) VALUES(%s,%s)", (sku_id, norm))

        if image_file and os.path.exists(image_file):
            ext   = os.path.splitext(image_file)[1].lower()
            fname = secure_filename(f"{sku_id}_{uuid.uuid4().hex}{ext}")
            dst   = os.path.join(current_app.config["UPLOAD_DIR"], fname)
            Image.open(image_file).convert("RGB").save(dst)
            exec_sql("INSERT INTO sku_images(sku_id,image_path,is_primary) VALUES(%s,%s,%s)",
                     (sku_id, fname, True))
        rows_added += 1

    flash(f"ÄÃ£ nháº­p {rows_added} dÃ²ng","success")
    return redirect(url_for("skus"))

