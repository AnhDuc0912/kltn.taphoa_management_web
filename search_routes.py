
import io, time, os
from flask import Blueprint, request, jsonify
from PIL import Image
from db import get_conn
from utils import vn_norm, encode_texts, encode_images

bp = Blueprint("search", __name__)

def _q(sql, params=None, fetch="all"):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        conn.commit()

@bp.post("/embed/backfill/text")
def backfill_text_vec():
    limit = int(request.args.get("limit", 500))
    rows = _q("""
        SELECT id, text FROM sku_texts
        WHERE text_vec IS NULL
        ORDER BY id ASC
        LIMIT %s
    """, (limit,))
    if not rows: 
        return jsonify({"updated": 0, "done": True})

    vecs = encode_texts([r[1] for r in rows])
    for (row, vec) in zip(rows, vecs):
        _q("UPDATE sku_texts SET text_vec=%s::vector WHERE id=%s", (vec.tolist(), row[0]), fetch=None)

    return jsonify({"updated": len(rows)})


@bp.post("/embed/backfill/image")
def backfill_image_vec():
    limit = int(request.args.get("limit", 200))
    rows = _q("""
        SELECT id, image_path FROM sku_images
        WHERE image_vec IS NULL
        ORDER BY id ASC
        LIMIT %s
    """, (limit,))
    if not rows:
        return jsonify({"updated": 0, "done": True})

    pils = []
    ids = []
    for rid, path in rows:
        try:
            with open(path, "rb") as f:
                pils.append(Image.open(io.BytesIO(f.read())))
                ids.append(rid)
        except Exception:
            # náº¿u áº£nh há»ng => bá» qua
            pass
    if not pils:
        return jsonify({"updated": 0})
    vecs = encode_images(pils)
    for rid, vec in zip(ids, vecs):
        _q("UPDATE sku_images SET image_vec=%s::vector WHERE id=%s", (vec.tolist(), rid), fetch=None)
    return jsonify({"updated": len(ids)})


@bp.post("/search/text")
def search_text():
    data = request.get_json(silent=True) or {}
    qtext = (data.get("q") or "").strip()
    k = int(data.get("k") or 20)
    if not qtext:
        return jsonify({"error": "missing q"}), 400

    t0 = time.time()
    vec = encode_texts([qtext])[0].tolist()

    # log query
    qid = _q(
        "INSERT INTO queries(type, raw_text, normalized_text) VALUES('text', %s, %s) RETURNING id",
        (qtext, vn_norm(qtext)), fetch="one"
    )[0]

    rows = _q("""
        SELECT st.sku_id, st.id, st.text, (st.text_vec <-> %s::vector) AS dist
        FROM sku_texts st
        WHERE st.text_vec IS NOT NULL
        ORDER BY st.text_vec <-> %s::vector
        LIMIT %s
    """, (vec, vec, k))

    results = []
    for rank, (sku_id, st_id, text, dist) in enumerate(rows, start=1):
        score = 1.0 - float(dist)
        _q("INSERT INTO query_candidates(query_id, sku_id, rank, score) VALUES(%s,%s,%s,%s)",
           (qid, sku_id, rank, score), fetch=None)
        results.append({
            "sku_id": sku_id, "sku_text_id": st_id, "text": text,
            "dist": float(dist), "score": score
        })

    return jsonify({
        "query_id": qid,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "results": results
    })


@bp.post("/search/image")
def search_image():
    file = request.files.get("file")
    k = int(request.form.get("k") or 20)
    if not file:
        return jsonify({"error": "missing file"}), 400

    t0 = time.time()
    img = Image.open(file.stream).convert("RGB")
    vec = encode_images([img])[0].tolist()

    # log query
    qid = _q("INSERT INTO queries(type) VALUES('image') RETURNING id", fetch="one")[0]

    rows = _q("""
        SELECT si.sku_id, si.id, si.image_path, (si.image_vec <-> %s::vector) AS dist
        FROM sku_images si
        WHERE si.image_vec IS NOT NULL
        ORDER BY si.image_vec <-> %s::vector
        LIMIT %s
    """, (vec, vec, k))

    results = []
    for rank, (sku_id, img_id, image_path, dist) in enumerate(rows, start=1):
        score = 1.0 - float(dist)
        _q("INSERT INTO query_candidates(query_id, sku_id, rank, score) VALUES(%s,%s,%s,%s)",
           (qid, sku_id, rank, score), fetch=None)
        results.append({
            "sku_id": sku_id, "sku_image_id": img_id, "image_path": image_path,
            "dist": float(dist), "score": score
        })

    return jsonify({
        "query_id": qid,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "results": results
    })

