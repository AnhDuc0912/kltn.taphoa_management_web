# routes/skus.py
import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash
from services.db_utils import q, exec_sql
from utils import vn_norm

bp = Blueprint("api_skus_bp", __name__)

@bp.get("/cart/summary")
def cart_summary():
    """
    Tóm tắt giỏ hàng để hiển thị:
    - Input (query):
        ids=1,2,3             # bắt buộc
        qty=2,1,5              # tùy chọn, khớp thứ tự với ids
    - Output:
        {
          "items": [
            {
              "sku_id": 1,
              "sku_name": "...",
              "brand_name": "...",
              "image_path": "path/to.jpg",
              "image_url": "http://<host>/media/path/to.jpg",
              "description": "...",
              "price": 15000.0,
              "qty": 2,
              "line_total": 30000.0
            },
            ...
          ],
          "subtotal": 123000.0,
          "count_distinct": 3
        }
    """
    from urllib.parse import urljoin

    def _abs_url(path: str) -> str | None:
        if not path:
            return None
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(request.host_url, f"media/{path.lstrip('/')}")  # cần route /media/<path> đã cấu hình

    ids_raw = (request.args.get("ids") or "").strip()
    qty_raw = (request.args.get("qty") or "").strip()

    if not ids_raw:
        return jsonify({"error": "ids is required, e.g. ids=12,34"}), 400

    # Parse ids & qty
    try:
        sku_ids_in_order = [int(x) for x in ids_raw.split(",") if x.strip() != ""]
    except Exception:
        return jsonify({"error": "ids must be comma-separated integers"}), 400

    if not sku_ids_in_order:
        return jsonify({"error": "no valid sku_id provided"}), 400

    qty_list = []
    if qty_raw:
        try:
            qty_list = [int(x) for x in qty_raw.split(",")]
        except Exception:
            qty_list = []
    # Chuẩn hóa độ dài qty = len(ids), mặc định 1 nếu thiếu
    if len(qty_list) != len(sku_ids_in_order):
        qty_list = [1] * len(sku_ids_in_order)
    # Số lượng tối thiểu 1
    qty_list = [max(1, int(q)) for q in qty_list]

    try:
        # Enrich 1 lần cho tất cả sku_id
        rows = _q("""
            SELECT
                sk.id AS sku_id,                          -- 0
                sk.name AS sku_name,                      -- 1
                COALESCE(b.name, '') AS brand_name,       -- 2
                -- Giá: lấy mới nhất từ sku_prices, rơi về sk.price nếu null
                COALESCE(sp.price, sk.price, 0)::float AS price, -- 3

                -- Ảnh & mô tả đại diện
                COALESCE(sc.image_path, si.image_path, '') AS rep_image_path,  -- 4
                COALESCE(sc.caption_text, si.ocr_text, st.text, '') AS description -- 5
            FROM skus sk
            LEFT JOIN brands b ON b.id = sk.brand_id

            -- Lấy giá mới nhất (nếu có)
            LEFT JOIN LATERAL (
                SELECT price
                FROM sku_prices
                WHERE sku_id = sk.id
                ORDER BY updated_at DESC NULLS LAST, id DESC
                LIMIT 1
            ) sp ON TRUE

            -- Lấy caption + ảnh đại diện
            LEFT JOIN LATERAL (
                SELECT caption_text, image_path
                FROM sku_captions
                WHERE sku_id = sk.id AND lang='vi' AND style='search'
                ORDER BY id ASC
                LIMIT 1
            ) sc ON TRUE

            -- Lấy ảnh/ocr fallback
            LEFT JOIN LATERAL (
                SELECT image_path, ocr_text
                FROM sku_images
                WHERE sku_id = sk.id
                ORDER BY (ocr_text IS NULL) ASC, id ASC
                LIMIT 1
            ) si ON TRUE

            -- Lấy text fallback
            LEFT JOIN LATERAL (
                SELECT text
                FROM sku_texts
                WHERE sku_id = sk.id
                ORDER BY id ASC
                LIMIT 1
            ) st ON TRUE

            WHERE sk.id = ANY(%s)
        """, (sku_ids_in_order,), fetch="all") or []

        # Map theo sku_id để giữ thứ tự đầu vào & qty tương ứng
        info = {
            int(r[0]): {
                "sku_name": r[1],
                "brand_name": r[2],
                "price": float(r[3] or 0.0),
                "image_path": r[4],
                "description": r[5],
            } for r in rows
        }

        items = []
        subtotal = 0.0

        for sku_id, qty in zip(sku_ids_in_order, qty_list):
            meta = info.get(int(sku_id))
            if not meta:
                # SKU không tồn tại → vẫn trả placeholder để UI hiển thị được
                item = {
                    "sku_id": int(sku_id),
                    "sku_name": None,
                    "brand_name": None,
                    "image_path": None,
                    "image_url": None,
                    "description": None,
                    "price": 0.0,
                    "qty": int(qty),
                    "line_total": 0.0,
                }
                items.append(item)
                continue

            price = float(meta["price"])
            line_total = float(price * qty)
            subtotal += line_total

            image_path = meta.get("image_path")
            image_url = _abs_url(image_path)

            item = {
                "sku_id": int(sku_id),
                "sku_name": meta.get("sku_name"),
                "brand_name": meta.get("brand_name"),
                "image_path": image_path,
                "image_url": image_url,
                "description": meta.get("description"),
                "price": price,
                "qty": int(qty),
                "line_total": line_total,

                # mirrors cho Android
                "skuId": int(sku_id),
                "skuName": meta.get("sku_name"),
                "brandName": meta.get("brand_name"),
                "imagePath": image_path,
                "imageUrl": image_url,
                "descriptionText": meta.get("description"),
                "unitPrice": price,
                "quantity": int(qty),
                "lineTotal": line_total,
            }
            items.append(item)

        resp = {
            "items": items,
            "subtotal": float(subtotal),
            "count_distinct": len(items)
        }
        return jsonify(resp), 200

    except Exception as e:
        current_app.logger.exception("Cart summary error")
        return jsonify({"error": str(e)}), 500

