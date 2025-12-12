#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/faiss_index_db.py

Helpers to save/load FAISS indexes into PostgreSQL (bytea) and store small metadata.

Schema suggestion (run once in psql):

CREATE TABLE IF NOT EXISTS faiss_indexes (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  index_data BYTEA NOT NULL,
  d INTEGER,
  index_type TEXT,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

Usage:
  from tools.faiss_index_db import save_index_to_db, load_index_from_db
  save_index_to_db(index, name="my_index", d=512, index_type="IVF", metadata={})
  idx = load_index_from_db(id=1)

This file uses faiss.write_index/read_index as a fallback to support older faiss python builds.
"""
import os
import json
import tempfile
from typing import Optional, Any, Dict

import faiss
from db import get_conn
import psycopg2


def _index_to_bytes(index) -> bytes:
    """Serialize FAISS index to bytes. Try in-memory serialize if available, else write/read temp file."""
    # Preferred: faiss.serialize_index / faiss.serialize_index can return bytes on many builds
    try:
        # new-style API
        if hasattr(faiss, "serialize_index"):
            b = faiss.serialize_index(index)
            if isinstance(b, (bytes, bytearray)):
                return bytes(b)
    except Exception:
        pass

    # Fallback: write to temp file and read bytes
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    try:
        faiss.write_index(index, tf.name)
        with open(tf.name, "rb") as f:
            data = f.read()
        return data
    finally:
        try:
            os.unlink(tf.name)
        except Exception:
            pass


def _bytes_to_index(b: bytes):
    """Deserialize bytes to FAISS index. Try in-memory deserialize then fallback to temp file."""
    try:
        if hasattr(faiss, "deserialize_index"):
            idx = faiss.deserialize_index(b)
            return idx
    except Exception:
        pass

    tf = tempfile.NamedTemporaryFile(delete=False)
    try:
        with open(tf.name, "wb") as f:
            f.write(b)
        idx = faiss.read_index(tf.name)
        return idx
    finally:
        try:
            os.unlink(tf.name)
        except Exception:
            pass


def save_index_to_db(index, name: str, d: Optional[int] = None, index_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
    """Serialize FAISS index and insert into DB. Returns inserted row id."""
    data = _index_to_bytes(index)
    md = metadata or {}
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO faiss_indexes (name, index_data, d, index_type, metadata) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (name, psycopg2.Binary(data), d, index_type, json.dumps(md)),
            )
            row = cur.fetchone()
            return row[0]
    finally:
        conn.close()


def load_index_from_db(id: Optional[int] = None, name: Optional[str] = None):
    """Load FAISS index by id or name. Returns deserialized index or None."""
    if id is None and name is None:
        raise ValueError("Provide id or name")
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            if id is not None:
                cur.execute("SELECT index_data FROM faiss_indexes WHERE id = %s", (id,))
            else:
                cur.execute("SELECT index_data FROM faiss_indexes WHERE name = %s ORDER BY created_at DESC LIMIT 1", (name,))
            row = cur.fetchone()
            if not row:
                return None
            b = bytes(row[0])
            return _bytes_to_index(b)
    finally:
        conn.close()


def list_indexes(limit: int = 50):
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT id, name, d, index_type, metadata, created_at FROM faiss_indexes ORDER BY created_at DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save/load FAISS index to/from Postgres")
    parser.add_argument("--dump", help="Path to local index file to load and save to DB")
    parser.add_argument("--name", help="Name for the DB entry")
    parser.add_argument("--list", action="store_true", help="List recent indexes")
    parser.add_argument("--get-id", type=int, help="Load index by id and write to stdout file path")
    parser.add_argument("--out", help="When --get-id set, write index bytes to this path")
    args = parser.parse_args()

    if args.list:
        for r in list_indexes():
            print(r)
        raise SystemExit(0)

    if args.dump:
        # read a local index file and store
        idx = faiss.read_index(args.dump)
        rid = save_index_to_db(idx, name=(args.name or os.path.basename(args.dump)), d=None, index_type=None, metadata={})
        print(f"Saved index as id={rid}")
        raise SystemExit(0)

    if args.get_id:
        idx = load_index_from_db(id=args.get_id)
        if idx is None:
            print("Not found")
            raise SystemExit(2)
        if not args.out:
            print("Please supply --out to write index file")
            raise SystemExit(2)
        faiss.write_index(idx, args.out)
        print(f"Wrote index to {args.out}")
        raise SystemExit(0)

    parser.print_help()
