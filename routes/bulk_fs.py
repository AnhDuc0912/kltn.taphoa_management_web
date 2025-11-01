# routes/bulk_fs.py
import os, json, csv, random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, request, jsonify
from db import get_conn
from services.db_utils import q
import psycopg2.extras as pg_extras

bp_bulk = Blueprint("bulk_fs", __name__)

ALLOWED_EXTS = {"jpg","jpeg","png","webp"}
TIMEOUT = (5, 20)

def _now_iso(): return datetime.utcnow().isoformat() + "Z"

def _data_dir() -> Path:
    base = Path(current_app.config.get("DATA_DIR", "data")).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base

def _uploads_dir() -> Path:
    up = Path(current_app.config.get("UPLOAD_FOLDER", "uploads")).resolve()
    (up / "sku_images").mkdir(parents=True, exist_ok=True)
    return up

def _load_json(name: str, default: Any):
    p = _data_dir() / name
    if not p.exists(): return default
    return json.loads(p.read_text(encoding="utf-8") or json.dumps(default))

def _save_json(name: str, data: Any):
    p = _data_dir() / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)  

def _next_id(kind: str) -> int:
    meta = _load_json("meta.json", {})
    key = f"{kind}_last_id"
    nxt = int(meta.get(key, 0)) + 1
    meta[key] = nxt
    _save_json("meta.json", meta)
    return nxt

def _list_skus(): return _load_json("skus.json", [])
def _save_skus(rows): _save_json("skus.json", rows)
def _list_images(): return _load_json("sku_images.json", [])
def _save_images(rows): _save_json("sku_images.json", rows)

def _ext_from_ct(ct: str) -> str:
    ct = (ct or "").lower()
    if "jpeg" in ct: return "jpg"
    if "png" in ct: return "png"
    if "webp" in ct: return "webp"
    return ""

@bp_bulk.post("/api/bulk/skus-with-images")
def bulk_skus_with_images():
    """
    Body JSON: list các item
    [
      {
        "external_id": "SUNL-250-CHANH",
        "sku": {
          "name": "...", "brand_id":..., "category_id":..., "variant":"...", "size_text":"...",
          "barcode":"...", "attrs": {...}, "facets": {...}, "is_active": true
        },
        "images": [
          {
            "url": "https://.../img1.jpg",
            "is_primary": true,
            "ocr_text": "optional",
            "colors": ["xanh lá"],
            "color_scores": {"xanh lá": 0.9},
            "detected_facets": {"packaging": "chai"},
            "image_vec": null,
            "image_vec_768": null
          }
        ]
      },
      ...
    ]
    """
    items = request.get_json(silent=True)
    if not isinstance(items, list):
        return jsonify({"ok": False, "error": "Body phải là array JSON"}), 400

    skus = _list_skus()
    images = _list_images()
    results = []

    for item in items:
        ext_id = (item.get("external_id") or "").strip() or None
        sku_data = item.get("sku") or {}
        imgs     = item.get("images") or []

        name = (sku_data.get("name") or "").strip()
        if not name:
            results.append({"external_id": ext_id, "ok": False, "error": "Thiếu sku.name"})
            continue

        # tạo SKU
        sku_id = _next_id("sku")
        row = {
            "id": sku_id,
            "brand_id": sku_data.get("brand_id"),
            "category_id": sku_data.get("category_id"),
            "name": name,
            "variant": sku_data.get("variant"),
            "size_text": sku_data.get("size_text"),
            "barcode": sku_data.get("barcode"),
            "attrs": sku_data.get("attrs") or {},
            "is_active": bool(sku_data.get("is_active", True)),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "image": None,
            "facets": sku_data.get("facets") or {},
        }
        skus.append(row)

        uploaded = []
        primary_set = False

        # tải & lưu ảnh
        for im in imgs:
            url = im.get("url")
            if not url: continue
            try:
                # Không tải ảnh, chỉ lưu đường dẫn URL
                rel_path = url
            except Exception as e:
                uploaded.append({"url": url, "ok": False, "error": str(e)})
                continue

            img_id = _next_id("sku_image")
            rec = {
                "id": img_id,
                "sku_id": sku_id,
                "image_path": rel_path,
                "ocr_text": im.get("ocr_text"),
                "is_primary": bool(im.get("is_primary", False)) and not primary_set,
                "created_at": _now_iso(),
                "colors": im.get("colors") or [],
                "color_scores": im.get("color_scores") or {},
                "detected_facets": im.get("detected_facets") or {},
                "image_vec": im.get("image_vec"),
                "image_vec_768": im.get("image_vec_768"),
            }
            images.append(rec)
            uploaded.append({"url": url, "ok": True, "image_path": rel_path, "id": img_id})
            if rec["is_primary"] and not primary_set:
                primary_set = True
                row["image"] = rel_path

        results.append({
            "name": name,
            "category_id": sku_data.get("category_id"),
            "image_paths": [u["image_path"] for u in uploaded if u.get("ok")]
        })

    _save_skus(skus)
    _save_images(images)
    return jsonify({"ok": True, "results": results})

@bp_bulk.post("/api/bulk/import-random-from-csv")
def import_random_from_csv():
    """
    Import 10 sản phẩm ngẫu nhiên từ file CSV bhx_gallery_by_ajax_with_name.csv.
    Kiểm tra không trùng tên sản phẩm với CSDL hiện tại.
    """
    csv_path = Path(__file__).parent / "bhx_gallery_by_ajax_with_name.csv"
    print(csv_path)
    if not csv_path.exists():
        return jsonify({"ok": False, "error": "File CSV không tồn tại"}), 404

    # Đọc CSV và nhóm theo product_id
    products = {}
    with open(csv_path, 'r', encoding='cp1252') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['product_id']
            if pid not in products:
                products[pid] = {
                    'name': row['product_name'],
                    'images': []
                }
            products[pid]['images'].append(row['local_url'])

    # Strip prefix từ image paths
    prefix = "/content/drive/MyDrive/Data BHX/bhx_gallery/"
    for prod in products.values():
        prod['images'] = [img.replace(prefix, "", 1) if img.startswith(prefix) else img for img in prod['images']]

    # Lấy danh sách unique products
    product_list = list(products.values())

    # Lấy existing names từ DB
    existing_names = {row[0] for row in q("SELECT name FROM skus", fetch="all")}

    # Chọn ngẫu nhiên 10 không trùng tên
    available = [p for p in product_list if p['name'] not in existing_names]
    if len(available) < 10:
        return jsonify({"ok": False, "error": f"Chỉ có {len(available)} sản phẩm không trùng tên"}), 400

    selected = random.sample(available, 10)

    # Thêm vào DB
    results = []

    for prod in selected:
        # Insert SKU
        sku_row = q("""
            INSERT INTO skus (brand_id, category_id, name, variant, size_text, barcode, attrs, is_active, image, facets)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (None, None, prod['name'], None, None, None, pg_extras.Json({}), True, prod['images'][0] if prod['images'] else None, pg_extras.Json({})), fetch="one")
        sku_id = sku_row[0]

        # Insert images
        for i, img_url in enumerate(prod['images']):
            q("""
                INSERT INTO sku_images (sku_id, image_path, is_primary, colors, color_scores, detected_facets)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (sku_id, img_url, i == 0, [], pg_extras.Json({}), pg_extras.Json({})), fetch=None)

        results.append({
            "name": prod['name'],
            "category_id": None,
            "image_paths": prod['images']
        })

    return jsonify({"ok": True, "results": results})
