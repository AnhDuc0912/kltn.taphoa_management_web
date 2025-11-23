#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/backfill_image_vecs.py

Backfill `image_vec` column in `sku_images` using the ResNet101 embedding model.

This script mirrors the logic in `routes/sku_images.py` but is a CLI tool suitable
for long-running batch jobs (no HTTP server required).

Usage:
  python tools/backfill_image_vecs.py --limit 1000 --batch-size 64

Options:
  --limit N       Process at most N suspect images (default: all)
  --batch-size N  Compute embeddings in batches (default: 128)
  --offset N      Skip first N rows (useful to resume)
  --dry-run       Don't write to DB, just report
  --checkpoint    Path to model checkpoint (passed to load_model)

The script updates DB in bulk using psycopg2.extras.execute_values for efficiency.
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db_utils import get_connection
from services.resnet101 import load_model, extract_embedding
from psycopg2.extras import execute_values

def iter_image_rows(conn, limit: int = None, offset: int = 0, null_only: bool = True, id_min: int | None = None, id_max: int | None = None):
    cur = conn.cursor()
    # Build WHERE clauses incrementally so we can combine null-only and id filters
    where_clauses = []
    if null_only:
        where_clauses.append("image_vec IS NULL")

    if id_min is not None:
        where_clauses.append(f"id >= {int(id_min)}")
    if id_max is not None:
        where_clauses.append(f"id <= {int(id_max)}")

    where_sql = ""
    if where_clauses:
        where_sql = " WHERE " + " AND ".join(where_clauses)

    q = f"SELECT id, sku_id, image_path FROM sku_images{where_sql} ORDER BY id"

    if limit is not None:
        q = q + f" LIMIT {limit} OFFSET {offset}"
    cur.execute(q)
    rows = cur.fetchall()
    cur.close()
    for r in rows:
        yield r


def chunked(it, size):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill image_vec for sku_images using ResNet101 embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding extraction")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rows before processing")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--checkpoint", type=str, default="out-resnet101-model/finetuned_resnet101.pt", help="Model checkpoint path")
    parser.add_argument("--all-images", action="store_true", help="Process all images, not just ones with NULL image_vec")
    parser.add_argument("--id-min", type=int, default=None, help="Minimum image id to include (inclusive)")
    parser.add_argument("--id-max", type=int, default=None, help="Maximum image id to include (inclusive)")
    args = parser.parse_args()

    conn = get_connection()
    model = load_model(checkpoint_path=args.checkpoint)

    # Collect rows to process
    null_only = not args.all_images
    rows = list(iter_image_rows(conn, limit=args.limit, offset=args.offset, null_only=null_only, id_min=args.id_min, id_max=args.id_max))
    total = len(rows)
    
    range_msg = ""
    if args.id_min is not None or args.id_max is not None:
        range_msg = f" id_range=({args.id_min or '-'}..{args.id_max or '-'})"

    if null_only:
        print(f"Found {total} images with NULL image_vec to process (offset={args.offset}){range_msg}")
    else:
        print(f"Found {total} total images to process (offset={args.offset}){range_msg}")
    
    if total == 0:
        return
    
    from PIL import Image

    processed = 0
    base_dir = os.getenv("UPLOAD_DIR", "uploads")

    for batch in chunked(rows, args.batch_size):
        # batch: list of (id, sku_id, image_path)
        valid_rows = []
        image_paths = []

        for img_id, sku_id, db_path in batch:
            raw_path = (db_path or "").strip()

            # Bỏ record image_path rỗng
            if not raw_path:
                print(f"⚠️ Skipping id={img_id}: empty image_path in DB")
                continue

            # Chuẩn hoá path: đổi \ -> /, bỏ / đầu
            p = raw_path.replace("\\", "/").lstrip("/")

            # Nếu trong DB lỡ lưu 'uploads/xxx' thì bỏ prefix đó đi,
            # vì base_dir đã là 'uploads'
            if p.lower().startswith("uploads/"):
                p = p[len("uploads/"):]

            full_path = os.path.join(base_dir, p)

            # Nếu không phải file → bỏ qua (folder / không tồn tại)
            if not os.path.exists(full_path) or not os.path.isfile(full_path):
                print(f"⚠️ Skipping id={img_id}, image_path='{db_path}': not a file ({full_path})")
                continue

            # Test ảnh có đọc được không trước khi đưa cho extract_embedding
            try:
                with Image.open(full_path) as im:
                    im.verify()
            except Exception as e:
                print(f"⚠️ Skipping id={img_id}, invalid image '{full_path}': {e}")
                continue

            valid_rows.append((img_id, sku_id, db_path))
            image_paths.append(full_path)

        if not image_paths:
            # Cả batch toàn rác
            print("⚠️ No valid images in this batch, skipping.")
            continue

        # Gọi extract_embedding cho batch ảnh hợp lệ
        vectors = extract_embedding(model, image_paths, batch_size=args.batch_size)
        if vectors is None:
            print("Warning: no vectors returned for batch, skipping")
            continue

        if len(vectors) != len(valid_rows):
            print(f"Warning: vector count {len(vectors)} != valid_row count {len(valid_rows)}; skipping this batch")
            continue

        updates = []
        ids = []
        for v, (img_id, sku_id, db_path) in zip(vectors, valid_rows):
            # v lúc này giống như code cũ đã dùng: cả vector
            updates.append((str(v), img_id))
            ids.append(img_id)

        if not updates:
            print("⚠️ No updates generated for this batch, skipping.")
            continue

        if args.dry_run:
            print(f"[DRY] Would update {len(updates)} rows (ids {ids[0]}..{ids[-1]})")
        else:
            cur = conn.cursor()
            if args.all_images:
                execute_values(
                    cur,
                    """
                    UPDATE sku_images AS s
                    SET image_vec = v.vector::vector
                    FROM (VALUES %s) AS v(vector, id)
                    WHERE s.id = v.id
                    """,
                    updates,
                    template="(%s, %s)"
                )
            else:
                execute_values(
                    cur,
                    """
                    UPDATE sku_images AS s
                    SET image_vec = v.vector::vector
                    FROM (VALUES %s) AS v(vector, id)
                    WHERE s.id = v.id AND s.image_vec IS NULL
                    """,
                    updates,
                    template="(%s, %s)"
                )
            conn.commit()
            cur.close()
            print(f"✅ Updated {len(updates)} rows (ids {ids[0]}..{ids[-1]})")

        processed += len(updates)
        time.sleep(0.01)

    print(f"Done. Processed {processed}/{total} images.")


if __name__ == '__main__':
    main()
