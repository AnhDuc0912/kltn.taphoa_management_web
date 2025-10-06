import psycopg2
import os
import csv
from services.db_utils import get_conn  # Giả sử từ db_utils
import random

# Hàm kết nối CSDL
def connect_db():
    return get_conn()

# Hàm xuất dữ liệu triplet theo sku_id
def export_retrieval_dataset(output_file="retrieval_triplet_dataset.csv", base_path="E:/api_hango/flask_pgvector_shop/flask_pgvector_shop/uploads/"):
    conn = connect_db()
    cursor = conn.cursor()

    # Query tất cả dữ liệu từ sku_images
    cursor.execute("""
        SELECT id, sku_id, image_path, image_vec, ocr_text, is_primary, created_at, 
               image_vec_768, colors, color_scores, detected_facets
        FROM sku_images
        ORDER BY sku_id, id
    """)
    rows = cursor.fetchall()

    # Nhóm dữ liệu theo sku_id
    sku_data = {}
    for row in rows:
        sku_id = row[1]  # sku_id
        if sku_id not in sku_data:
            sku_data[sku_id] = []
        
        # Xử lý image_path: ghép base_path nếu không phải URL
        image_path = row[2]
        if image_path and not image_path.startswith('http'):
            full_path = os.path.join(base_path, image_path)
            if not os.path.exists(full_path):
                full_path = image_path  # Giữ nguyên nếu không tìm thấy
        else:
            full_path = image_path

        sku_data[sku_id].append({
            "id": row[0],
            "image_path": image_path,
            "image_vec": str(row[3]) if row[3] else None,
            "ocr_text": row[4],
            "is_primary": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
            "image_vec_768": str(row[7]) if row[7] else None,
            "colors": str(row[8]) if row[8] else '{}',
            "color_scores": str(row[9]) if row[9] else '{}',
            "detected_facets": str(row[10]) if row[10] else '{}'
        })

    cursor.close()
    conn.close()

    # Tạo triplet dataset
    triplets = []
    sku_ids = list(sku_data.keys())
    for sku_id in sku_ids:
        images = sku_data[sku_id]
        if len(images) < 2:  # Cần ít nhất 2 ảnh để tạo anchor và positive
            continue
        
        # Tạo tối đa 5 triplet mỗi SKU
        for _ in range(min(5, len(images))):
            anchor = random.choice(images)
            positive = random.choice([img for img in images if img != anchor])
            negative_sku = random.choice([s for s in sku_ids if s != sku_id])
            negative = random.choice(sku_data[negative_sku])

            triplets.append({
                "anchor_image_path": anchor["image_path"],
                "positive_image_path": positive["image_path"],
                "negative_image_path": negative["image_path"],
                "sku_id": sku_id,
                "anchor_id": anchor["id"],
                "positive_id": positive["id"],
                "negative_id": negative["id"],
                "anchor_colors": anchor["colors"],
                "positive_colors": positive["colors"],
                "negative_colors": negative["colors"],
                "anchor_color_scores": anchor["color_scores"],
                "positive_color_scores": positive["color_scores"],
                "negative_color_scores": negative["color_scores"]
            })

    # Xuất CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["sku_id", "anchor_image_path", "positive_image_path", "negative_image_path",
                      "anchor_id", "positive_id", "negative_id", "anchor_colors", "positive_colors",
                      "negative_colors", "anchor_color_scores", "positive_color_scores", "negative_color_scores"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(triplets)

    print(f"Đã xuất {len(triplets)} triplet vào {output_file}")

# Main execution
if __name__ == "__main__":
    export_retrieval_dataset()