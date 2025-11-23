"""
Build FAISS index from all sku_images in database
Based on Copy_of_finetune_ResNet_101.ipynb workflow
"""
import os
import sys
import json
from pathlib import Path

import faiss
import numpy as np
import psycopg2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.db_utils import q, exec_sql
from services.resnet101 import RetrievalNet, load_model, collate_fn_filter_none

# Constants
EMB_DIM = 512
BATCH_SIZE = 128
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")  # thư mục uploads trên server
UPLOAD_URL_PREFIX = os.getenv("UPLOAD_URL_PREFIX", "/uploads")  # prefix URL để trả ra client


class DatabaseImagesDataset(Dataset):
    """Dataset for loading images from database"""

    def __init__(self, image_records, upload_dir, transform=None):
        """
        Args:
            image_records: List of (id, sku_id, image_path) tuples
            upload_dir: Root directory for images
            transform: Image transforms
        """
        self.records = image_records
        self.upload_dir = upload_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_id, sku_id, image_path = self.records[idx]

        # image_path trong DB có thể là full path hoặc relative
        # Chuẩn hoá lại: chỉ lấy tên file rồi join với UPLOAD_DIR
        norm_path = image_path.replace("\\", "/")
        filename = norm_path.split("/")[-1]
        full_path = os.path.join(self.upload_dir, filename)

        try:
            image = Image.open(full_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, (img_id, sku_id, filename)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None, (img_id, sku_id, filename)


def build_faiss_index_from_db(checkpoint_path="out-resnet101-model/finetuned_resnet101.pt"):
    """
    Build FAISS index từ vector image_vec trong DB.
    Chỉ dùng vector hợp lệ (đúng 512 chiều) và lưu vào bảng faiss_indexes.
    """

    print("=== Build FAISS index từ image_vec trong DB ===")

    rows = q("""
        SELECT id, sku_id, image_path, image_vec
        FROM sku_images
        WHERE image_vec IS NOT NULL
        ORDER BY sku_id, is_primary DESC, id
    """)

    if not rows:
        print("❌ Không có image_vec trong DB.")
        return None

    vectors = []
    metadata_list = []
    EXPECTED_DIM = 512   # số chiều vector mong đợi

    for img_id, sku_id, image_path, image_vec in rows:
        # print("DEBUG type:", img_id, type(image_vec))

        vec = None

        # 1) image_vec là string (case của mày)
        if isinstance(image_vec, str):
            s = image_vec.strip()
            try:
                # Nếu là JSON thuần: "[0.1, 0.2, ...]"
                vec_list = json.loads(s)
            except Exception:
                # Nếu là format khác: "{0.1,0.2,...}" hoặc "(0.1,0.2,...)"
                for ch in "[](){}":
                    s = s.replace(ch, "")
                parts = [p for p in s.split(",") if p.strip()]
                try:
                    vec_list = [float(p) for p in parts]
                except Exception as e:
                    print(f"⚠️ Skip: không parse được image_vec string (id={img_id}): {e}")
                    continue

            vec = np.asarray(vec_list, dtype="float32")

        # 2) list / tuple
        elif isinstance(image_vec, (list, tuple)):
            vec = np.asarray(image_vec, dtype="float32")

        # 3) numpy array
        elif isinstance(image_vec, np.ndarray):
            vec = image_vec.astype("float32")

        # 4) bytes / memoryview (bytea)
        elif isinstance(image_vec, (bytes, bytearray, memoryview)):
            buf = bytes(image_vec)
            vec = np.frombuffer(buf, dtype="float32")

        else:
            print(f"⚠️ Skip: image_vec format không hỗ trợ (id={img_id}, type={type(image_vec)})")
            continue

        # 5) Kiểm tra vector sau khi convert
        if vec is None or vec.size == 0:
            print(f"⚠️ Skip: vector rỗng hoặc None (id={img_id})")
            continue

        vec = vec.reshape(-1)

        if vec.shape[0] != EXPECTED_DIM:
            print(f"⚠️ Skip: vector size {vec.shape} != {EXPECTED_DIM} (id={img_id})")
            continue

        # vector OK → lưu lại
        vectors.append(vec)

        # chuẩn hoá filename
        filename = image_path.replace("\\", "/").split("/")[-1]
        metadata_list.append((img_id, sku_id, filename))

    if not vectors:
        print("❌ Không có vector hợp lệ nào để build FAISS index")
        return None

    all_embeddings = np.vstack(vectors).astype("float32")

    print(f"➡ Tổng vector hợp lệ: {len(vectors)}")
    print(f"➡ Embedding shape: {all_embeddings.shape}")

    # Build FAISS index
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_embeddings)

    print(f"➡ FAISS index size: {index.ntotal} items")

    # Serialize index + metadata → LƯU VÀO DB
    serialized = faiss.serialize_index(index)
    metadata_json = json.dumps([list(m) for m in metadata_list])

    exec_sql("""
        INSERT INTO faiss_indexes (name, index_data, index_type, metadata)
        VALUES (%s, %s, 'IndexFlatL2', %s)
    """, ("resnet101_faiss", psycopg2.Binary(serialized), metadata_json))

    print("✅ Saved FAISS index to DB.")

    return index, metadata_list

def search_image_with_faiss(query_image_path, k=5):
    """
    Search FAISS index lưu trong bảng faiss_indexes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    model.eval()

    row = q("""
        SELECT index_data, metadata
        FROM faiss_indexes
        ORDER BY id DESC
        LIMIT 1
    """, fetch="one")

    if not row:
        print("❌ No FAISS index found in database. Run build_faiss_index_from_db() first.")
        return []

    serialized_index, metadata_json = row

    # 🔧 deserialize index từ bytes
    index_arr = np.frombuffer(serialized_index, dtype="uint8")
    index = faiss.deserialize_index(index_arr)

    # 🔧 metadata có thể là string JSON hoặc list (jsonb)
    if isinstance(metadata_json, (str, bytes, bytearray)):
        metadata_list = json.loads(metadata_json)
    else:
        metadata_list = metadata_json

    print(f"Loaded FAISS index with {index.ntotal} embeddings")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    try:
        query_image = Image.open(query_image_path).convert("RGB")
        transformed_image = transform(query_image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error loading query image: {e}")
        return []

    with torch.no_grad():
        query_embedding = model(transformed_image).cpu().numpy().astype("float32")

    distances, indices = index.search(query_embedding, k)

    print(f"Found {k} nearest neighbors:")

    results = []
    for i in range(k):
        idx = indices[0][i]
        distance = distances[0][i]

        if 0 <= idx < len(metadata_list):
            img_id, sku_id, filename = metadata_list[idx]
            image_url = f"{UPLOAD_URL_PREFIX.rstrip('/')}/{filename}"

            print(f"Rank {i+1}: SKU {sku_id} | {filename} | Distance: {distance:.4f}")

            results.append({
                "rank": i + 1,
                "img_id": img_id,
                "sku_id": sku_id,
                "image_filename": filename,
                "image_url": image_url,
                "distance": float(distance),
                "score": 1.0 / (1.0 + distance),
            })

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build FAISS index or search images"
    )
    parser.add_argument(
        "--build", action="store_true", help="Build FAISS index from database"
    )
    parser.add_argument(
        "--search", type=str, help="Path to query image for search"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of results to return"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="out-resnet101-model/finetuned_resnet101.pt",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    if args.build:
        print("Building FAISS index...")
        build_faiss_index_from_db(checkpoint_path=args.checkpoint)
    elif args.search:
        print(f"Searching for similar images to: {args.search}")
        results = search_image_with_faiss(args.search, k=args.k)
        if results:
            print(f"\n✅ Found {len(results)} similar images")
            for r in results:
                print(r)
    else:
        parser.print_help()
