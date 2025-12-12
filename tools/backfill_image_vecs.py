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

def iter_image_rows(conn, limit: int = None, offset: int = 0):
    cur = conn.cursor()
    q = "SELECT id, sku_id, image_path FROM sku_images WHERE image_vec IS NULL ORDER BY id"
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
    args = parser.parse_args()

    conn = get_connection()
    model = load_model(checkpoint_path=args.checkpoint)

    # Collect rows to process
    rows = list(iter_image_rows(conn, limit=args.limit, offset=args.offset))
    total = len(rows)
    print(f"Found {total} images to process (offset={args.offset})")
    if total == 0:
        return

    processed = 0
    for batch in chunked(rows, args.batch_size):
        # batch: list of (id, sku_id, image_path)
        image_paths = [os.path.join(os.getenv('UPLOAD_DIR', 'uploads'), r[2]) for r in batch]
        ids = [r[0] for r in batch]

        vectors = extract_embedding(model, image_paths, batch_size=args.batch_size)
        # extract_embedding returns list of vectors
        if vectors is None:
            print("Warning: no vectors returned for batch, skipping")
            continue

        # Ensure vectors align with ids
        if len(vectors) != len(ids):
            print(f"Warning: vector count {len(vectors)} != id count {len(ids)}; skipping batch")
            continue

        updates = []
        for v, sid in zip(vectors, ids):
            # Postgres vector casting expects string representation like '[0.1, 0.2, ...]'
            updates.append((str(v), sid))

        if args.dry_run:
            print(f"[DRY] Would update {len(updates)} rows (ids {ids[0]}..{ids[-1]})")
        else:
            cur = conn.cursor()
            execute_values(
                cur,
                "UPDATE sku_images AS s SET image_vec = v.vector::vector FROM (VALUES %s) AS v(vector, id) WHERE s.id = v.id",
                updates,
                template="(%s, %s)"
            )
            conn.commit()
            cur.close()
            print(f"Updated {len(updates)} rows (ids {ids[0]}..{ids[-1]})")

        processed += len(updates)
        time.sleep(0.01)

    print(f"Done. Processed {processed}/{total} images.")


if __name__ == '__main__':
    main()
