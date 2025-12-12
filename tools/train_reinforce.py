"""Offline REINFORCE training script for the re-ranker.

Reads episodes from DB (queries + query_candidates), computes simple reward,
and performs policy-gradient updates on the SimpleRanker model. Saves model
to models/re_ranker.pt.

This is intentionally lightweight — adapt feature mapping and reward shaping
to your needs.
"""
import os
import json
import time
import math
import argparse
from collections import defaultdict

# Ensure project root is on sys.path so imports like `services.re_ranker` work
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
except Exception:
    raise RuntimeError("train_reinforce requires torch and numpy installed")

import psycopg2
from services.re_ranker import SimpleRanker


def fetch_episodes(conn, limit=1000):
    """
    Lấy toàn bộ candidates cho những query có hành vi (click/purchase/dwell).
    'limit' bây giờ là số query tối đa, không phải số hàng.
    """
    sql = """
    WITH pos AS (
      SELECT DISTINCT query_id
      FROM query_candidates
      WHERE (was_clicked = TRUE OR purchased = TRUE OR dwell_time IS NOT NULL)
        AND created_at > now() - interval '30 days'
      ORDER BY query_id DESC
      LIMIT %s
    )
    SELECT qc.query_id, qc.id, qc.sku_id, qc.rank, qc.score,
           qc.candidate_features, qc.was_clicked, qc.dwell_time, qc.purchased
    FROM query_candidates qc
    JOIN pos p ON p.query_id = qc.query_id
    ORDER BY qc.query_id, qc.rank;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()

    episodes = defaultdict(list)
    for row in rows:
        qid = row[0]
        feats = row[5]
        # đảm bảo candidate_features là dict
        if isinstance(feats, str):
            try:
                feats = json.loads(feats)
            except Exception:
                feats = {}
        episodes[qid].append({
            "qc_id": row[1],
            "sku_id": row[2],
            "rank": row[3],
            "score": row[4],
            "features": feats or {},
            "was_clicked": row[6],
            "dwell_time": row[7],
            "purchased": row[8],
        })
    return list(episodes.items())


def reward_for_candidate(rec):
    r = 0.0
    if rec.get("was_clicked"):
        r += 1.0
    if rec.get("purchased"):
        r += 5.0
    if rec.get("dwell_time"):
        r += min(float(rec["dwell_time"]) / 30.0, 1.0) * 0.5
    return r


def feat_to_vec(f, input_dim=16):
    # bool -> float
    f = dict(f or {})
    for k, v in list(f.items()):
        if isinstance(v, bool):
            f[k] = 1.0 if v else 0.0

    # hỗ trợ cả keywords_len/colors_len hoặc mảng gốc
    kw_len = f.get("keywords_len")
    if kw_len is None:
        kw_len = float(len(f.get("keywords") or []))
    col_len = f.get("colors_len")
    if col_len is None:
        col_len = float(len(f.get("colors") or []))

    # bỏ rác không nên đưa vào model
    for bad in ("re_rank_score", "debug", "source"):
        if bad in f: f.pop(bad, None)

    vec10 = [
        float(f.get("similarity", 0.0)),
        float(f.get("text_sim", 0.0)),
        float(f.get("image_sim", 0.0)),
        float(f.get("rank", 0.0)),
        float(f.get("brand_match", 0.0)),
        float(f.get("has_ocr", 0.0)),
        float(f.get("is_primary", 0.0)),
        float(kw_len),
        float(col_len),
        float(f.get("score", 0.0)),
    ]
    return np.array(vec10 + [0.0] * (input_dim - 10), dtype=np.float32)

def train_once(conn, model, optimizer, device, input_dim=16):
    episodes = fetch_episodes(conn, limit=2000)
    if not episodes:
        print("No episodes to train on")
        return
    baseline = 0.0
    alpha = 0.99
    losses = []
    for qid, items in episodes:
        X = np.stack([feat_to_vec(it.get("features", {}), input_dim) for it in items])
        rewards = np.array([reward_for_candidate(it) for it in items], dtype=np.float32)
        # convert to tensors
        Xt = torch.from_numpy(X).to(device)
        rt = torch.from_numpy(rewards).to(device)

        # logits
        logits = model(Xt)
        probs = torch.softmax(logits, dim=0)

        # choose logged action (clicked or purchased)
        # trong train_once, đoạn chọn logged action:
        chosen_idx = None
        for i, it in enumerate(items):
            if it.get("purchased"):
                chosen_idx = i; break
        if chosen_idx is None:
            for i, it in enumerate(items):
                if it.get("was_clicked"):
                    chosen_idx = i; break
        if chosen_idx is None:
            continue

        reward = rt[chosen_idx]
        baseline = alpha * baseline + (1 - alpha) * float(reward.item())
        adv = reward - baseline
        logp = torch.log(probs[chosen_idx] + 1e-9)
        loss = - logp * adv
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    avg = float(np.mean(losses)) if losses else None
    print("avg loss", avg)
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=False, help="Postgres DSN for training DB. If omitted, will use project's db.get_conn()/.env")
    p.add_argument("--out", default="models/re_ranker.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dsn:
        conn = psycopg2.connect(args.dsn)
    else:
        # Use project's db.get_conn() which loads .env via python-dotenv
        try:
            from db import get_conn
            conn = get_conn()
        except Exception as e:
            raise RuntimeError("No DSN provided and failed to get connection from db.get_conn(): %s" % e)
    model = SimpleRanker(input_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    base_out = args.out
    base_dir = os.path.dirname(base_out) or "."
    base_name = os.path.splitext(os.path.basename(base_out))[0]
    os.makedirs(base_dir, exist_ok=True)

    for epoch in range(1, 51):
        avg_loss = train_once(conn, model, optimizer, device, input_dim=16)
        if epoch % 5 == 0:
            ts = int(time.time())
            out_path = os.path.join(base_dir, f"{base_name}_{ts}.pt")
            meta_path = os.path.join(base_dir, f"{base_name}_{ts}.json")
            # store absolute model path in metadata so the running app can load it
            out_path = os.path.abspath(out_path)
            meta_path = os.path.abspath(meta_path)
            torch.save({"state_dict": model.state_dict(),
            "feature_keys": [
                "similarity","text_sim","image_sim","rank",
                "brand_match","has_ocr","is_primary",
                "keywords_len","colors_len","score"
            ]}, out_path)
            
            # write metadata for the saved model
            meta = {
                "model_path": out_path,
                "saved_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts)),
                "epoch": epoch,
                "input_dim": 16,
                "avg_loss": avg_loss,
                "feature_keys": [
                    "similarity","text_sim","image_sim","rank",
                    "brand_match","has_ocr","is_primary",
                    "keywords_len","colors_len","score"
                ],
            }
            try:
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass
            print(f"Saved model to {out_path} (meta: {meta_path})")

    conn.close()


if __name__ == "__main__":
    main()
