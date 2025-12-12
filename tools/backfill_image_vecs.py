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
import imghdr


def file_diagnostics(path: str):
    """Return diagnostics for a file: exists, size, head bytes hex, imghdr type."""
    info = {
        'exists': False,
        'size': None,
        'head_hex': None,
        'imghdr': None,
    }
    try:
        if os.path.exists(path) and os.path.isfile(path):
            info['exists'] = True
            info['size'] = os.path.getsize(path)
            with open(path, 'rb') as fh:
                head = fh.read(64)
                info['head_hex'] = head[:32].hex()
                info['imghdr'] = imghdr.what(None, h=head)
    except Exception as e:
        info['error'] = str(e)
    return info

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

    processed = 0
    for batch in chunked(rows, args.batch_size):
        # batch: list of (id, sku_id, image_path)
        image_paths_raw = [r[2] for r in batch]
        ids = [r[0] for r in batch]

        # Resolve full paths and pre-filter invalid entries to avoid mismatched vector counts
        full_paths = [os.path.join(os.getenv('UPLOAD_DIR', 'uploads'), p) if p else os.path.join(os.getenv('UPLOAD_DIR', 'uploads'), '') for p in image_paths_raw]
        valid_indices = []
        valid_full_paths = []
        valid_ids = []
        invalid_info = []
        for idx, (fid, raw_p, full_p) in enumerate(zip(ids, image_paths_raw, full_paths)):
            if not raw_p or raw_p.strip() == "":
                invalid_info.append((fid, raw_p, 'empty path'))
                continue
            if not os.path.exists(full_p) or not os.path.isfile(full_p):
                # note: os.path.exists may be False if path points to directory or missing
                invalid_info.append((fid, raw_p, 'missing_or_not_file'))
                continue
            # looks valid
            valid_indices.append(idx)
            valid_full_paths.append(full_p)
            valid_ids.append(fid)

        if invalid_info:
            print("Warning: found invalid image paths in batch:")
            for fid, raw_p, reason in invalid_info:
                print(f"  id={fid} path='{raw_p}' -> {reason}")

        if not valid_full_paths:
            print("No valid images in this batch, skipping")
            continue

        vectors = extract_embedding(model, valid_full_paths, batch_size=args.batch_size)
        # extract_embedding returns list of vectors
        if vectors is None:
            print("Warning: no vectors returned for batch, skipping")
            # Run diagnostics per-file to help find corrupt images
            for p in valid_full_paths:
                d = file_diagnostics(p)
                print(f"  file: {p} -> exists={d['exists']} size={d.get('size')} imghdr={d.get('imghdr')} head={d.get('head_hex')[:32] if d.get('head_hex') else None} error={d.get('error') if 'error' in d else None}")
            continue

        # Ensure vectors align with ids
        if len(vectors) != len(valid_ids):
            print(f"Warning: vector count {len(vectors)} != valid id count {len(valid_ids)}; entering per-image fallback")
            # Diagnose each valid image in the batch
            for p in valid_full_paths:
                d = file_diagnostics(p)
                print(f"  file: {p} -> exists={d['exists']} size={d.get('size')} imghdr={d.get('imghdr')} head={d.get('head_hex')[:32] if d.get('head_hex') else None} error={d.get('error') if 'error' in d else None}")

            # Fallback: try extracting embeddings per-image so one bad file doesn't drop the whole batch
            per_successes = []  # tuples of (vector_str, id)
            per_failures = []   # tuples of (id, path, error)
            for fid, p in zip(valid_ids, valid_full_paths):
                try:
                    single_vecs = extract_embedding(model, [p], batch_size=1)
                    if single_vecs and len(single_vecs) == 1 and single_vecs[0] is not None:
                        per_successes.append((str(single_vecs[0]), fid))
                    else:
                        per_failures.append((fid, p, 'no_vector_returned'))
                except Exception as e:
                    per_failures.append((fid, p, str(e)))
                    print(f"    per-image error id={fid} path={p}: {e}")

            if per_successes:
                cur = conn.cursor()
                if args.all_images:
                    execute_values(
                        cur,
                        "UPDATE sku_images AS s SET image_vec = v.vector::vector FROM (VALUES %s) AS v(vector, id) WHERE s.id = v.id",
                        per_successes,
                        template="(%s, %s)"
                    )
                else:
                    execute_values(
                        cur,
                        "UPDATE sku_images AS s SET image_vec = v.vector::vector FROM (VALUES %s) AS v(vector, id) WHERE s.id = v.id AND s.image_vec IS NULL",
                        per_successes,
                        template="(%s, %s)"
                    )
                conn.commit()
                cur.close()
                print(f"Per-image fallback updated {len(per_successes)} rows (of {len(valid_ids)})")

            if per_failures:
                print(f"Per-image fallback failed for {len(per_failures)} images; first examples:")
                for ex in per_failures[:10]:
                    print(f"  id={ex[0]} path={ex[1]} error={ex[2]}")

            # move on to next batch after fallback attempt
            processed += len(per_successes)
            continue

        updates = []
        # Map vectors to valid_ids
        for v, sid in zip(vectors, valid_ids):
            # Postgres vector casting expects string representation like '[0.1, 0.2, ...]'
            updates.append((str(v), sid))

        if args.dry_run:
            print(f"[DRY] Would update {len(updates)} rows (ids {ids[0]}..{ids[-1]})")
        else:
            cur = conn.cursor()
            if args.all_images:
                # For --all-images, always update regardless of current value
                execute_values(
                    cur,
                    "UPDATE sku_images AS s SET image_vec = v.vector::vector FROM (VALUES %s) AS v(vector, id) WHERE s.id = v.id",
                    updates,
                    template="(%s, %s)"
                )
            else:
                # For normal mode, only update NULL values (safer)
                execute_values(
                    cur,
                    "UPDATE sku_images AS s SET image_vec = v.vector::vector FROM (VALUES %s) AS v(vector, id) WHERE s.id = v.id AND s.image_vec IS NULL",
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
