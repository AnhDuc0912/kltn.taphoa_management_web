#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/fix_sku_names_gemini.py

Fix SKU names in the `skus` table using Gemini to restore Vietnamese diacritics
and repair common mojibake/encoding issues.

Features:
- Scans SKUs (optionally only 'suspect' names containing '?', 'Ã', or replacement chars)
- Calls Gemini to normalize each name to a single UTF-8 string
- Shows diffs and (optionally) updates the DB
- Safe defaults: --dry-run prints changes without writing

Usage examples:
  # Dry run for up to 100 suspect SKUs
  python tools/fix_sku_names_gemini.py --limit 100 --dry-run

  # Fix a single SKU by id and update DB
  python tools/fix_sku_names_gemini.py --sku-id 123 --update

"""
import os
import re
import time
import json
import logging
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except Exception:
    genai = None

from services.db_utils import q, exec_sql

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_suspect_name(name: str) -> bool:
    if not name:
        return False
    # Common mojibake indicators: literal question marks in place of diacritics,
    # replacement char U+FFFD, or sequences like 'Ã' often seen in mojibake from UTF-8<->latin1
    if "?" in name:
        return True
    if "�" in name:
        return True
    if "Ã" in name or "â" in name and any(ch in name for ch in ["Ã", "â"]):
        return True
    # If name contains many ascii-only words but also digits/units, leave to LLM only when flagged
    return False


def configure_genai():
    if genai is None:
        raise RuntimeError("google.generativeai not installed. Run: pip install google-generativeai")
    api_key = os.getenv("API_KEY_GEMINI")
    if not api_key:
        raise RuntimeError("API_KEY_GEMINI not set in environment/.env")
    genai.configure(api_key=api_key)


def call_gemini_correct_name(raw_name: str, model_name: str, timeout: int = 30, retry: int = 3) -> str:
    """Call Gemini to correct a single SKU name. Returns corrected name (string).
    Uses a conservative temperature and tries retries on failure.
    """
    if genai is None:
        raise RuntimeError("google.generativeai not installed")

    prompt = (
        "Bạn là trợ lý chỉnh sửa TÊN SẢN PHẨM bằng tiếng Việt.\n"
        "Nhiệm vụ: nhận vào một tên sản phẩm có thể bị lỗi mã hóa (thiếu dấu hoặc mojibake) \n"
        "và trả về một CHUỖI duy nhất là tên sản phẩm đã được sửa chính xác, bằng tiếng Việt có dấu (UTF-8).\n"
        "Nguyên tắc:\n"
        "- Giữ nguyên số lượng, trọng lượng, dung tích và đơn vị (vd: 720g, 1.9l, 5.5kg)\n"
        "- Không thêm hoặc bớt thông tin ngoài việc sửa chính tả và dấu\n"
        "- Trả về DUY NHẤT tên đã sửa, không giải thích, không thêm ký tự khác\n"
        "Ví dụ:\n"
        "Input: B?t gi?t nhi?t Aba h??ng n??c hoa 720g -> Output: Bột giặt nhiệt Aba hương nước hoa 720g\n"
        "Input: N??c r?a chén Sunlight matcha 1.9l và t?y b?n c?u VIM Zero chanh 750ml -> Output: Nước rửa chén Sunlight matcha 1.9l và tẩy bồn cầu VIM Zero chanh 750ml\n"
        "Now correct this single input exactly:\n"
    )

    model = genai.GenerativeModel(model_name)
    parts = [prompt, f"Input: {raw_name}\nOutput:"]

    # Try to import ResourceExhausted for specific quota handling
    ResourceExhausted = None
    try:
        from google.api_core.exceptions import ResourceExhausted as _RE
        ResourceExhausted = _RE
    except Exception:
        ResourceExhausted = None

    for attempt in range(1, retry + 1):
        try:
            res = model.generate_content(
                parts,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 64
                },
                request_options={"timeout": timeout}
            )
            text = getattr(res, "text", "") or ""
            text = text.strip()
            # Sometimes model returns JSON-like or with quotes; normalize to single-line name
            # Remove leading/trailing quotes and whitespace
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            # If model returned 'Output: ...', strip prefix
            text = re.sub(r"^Output:\s*", "", text, flags=re.IGNORECASE).strip()
            # As a final safeguard, collapse multiple spaces
            text = re.sub(r"\s+", " ", text)
            if text:
                return text
            else:
                logger.warning("Empty response from Gemini on attempt %d", attempt)
        except Exception as e:
            # Detect quota errors and back off more aggressively
            is_quota = False
            try:
                if ResourceExhausted and isinstance(e, ResourceExhausted):
                    is_quota = True
            except Exception:
                pass
            if not is_quota:
                msg = repr(e)
                if "ResourceExhausted" in msg or "quota" in msg.lower() or "429" in msg:
                    is_quota = True

            if is_quota:
                # exponential backoff with cap and small jitter
                backoff = min(60, 2 ** attempt)
                jitter = (attempt % 3) * 1
                sleep_for = backoff + jitter
                logger.warning("Gemini quota/error detected (attempt %d). Backing off for %ds...", attempt, sleep_for)
                try:
                    time.sleep(sleep_for)
                except Exception:
                    pass
                continue

            logger.exception("Gemini call failed on attempt %d: %s", attempt, e)
            try:
                time.sleep(0.5 * attempt)
            except Exception:
                pass

    # Fallback: try a light heuristic replacement of common mojibake tokens
    fallback = raw_name.replace("?", "ạ").replace("�", "").strip()
    logger.warning("Using fallback correction for '%s' -> '%s'", raw_name, fallback)
    return fallback


def fetch_suspect_skus(limit: int = 100, offset: int = 0):
    """Return up to `limit` suspect SKUs, skipping the first `offset` suspect matches.

    We iterate through all SKUs in id order and collect those matching is_suspect_name.
    The offset applies to the suspect matches (so you can resume at a later batch).
    """
    rows = q("SELECT id, name FROM skus ORDER BY id", fetch="all")
    if not rows:
        return []
    suspects = []
    skipped = 0
    for r in rows:
        sid, name = r[0], r[1]
        if is_suspect_name(name):
            if skipped < offset:
                skipped += 1
                continue
            suspects.append((sid, name))
            if len(suspects) >= limit:
                break
    return suspects


def update_sku_name(sku_id: int, new_name: str):
    exec_sql("UPDATE skus SET name = %s WHERE id = %s", (new_name, sku_id))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix SKU names using Gemini (normalize to Vietnamese UTF-8)")
    parser.add_argument("--limit", type=int, default=100, help="Max SKUs to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many suspect SKUs before processing (useful to resume)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes to DB")
    parser.add_argument("--sku-id", type=int, help="Process single SKU id")
    parser.add_argument("--update", action="store_true", help="If present, write the updated names to DB")
    parser.add_argument("--model-name", type=str, default=os.getenv("MODEL_NAME", "gemini-2.0-flash-exp"), help="Gemini model to use")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("GENAI_REQ_TIMEOUT", "60")), help="Request timeout seconds")
    args = parser.parse_args()

    try:
        configure_genai()
    except Exception as e:
        logger.exception("Gemini config error: %s", e)
        return

    to_process = []
    if args.sku_id:
        row = q("SELECT id, name FROM skus WHERE id = %s", (args.sku_id,), fetch="one")
        if not row:
            logger.error("SKU id %s not found", args.sku_id)
            return
        to_process = [(row[0], row[1])]
    else:
        to_process = fetch_suspect_skus(limit=args.limit, offset=args.offset)

    if not to_process:
        logger.info("No suspect SKUs found to process.")
        return

    results = []
    for sid, name in to_process:
        logger.info("Processing SKU %d: %s", sid, name)
        try:
            corrected = call_gemini_correct_name(name, args.model_name, timeout=args.timeout)
            logger.info(" -> Corrected: %s", corrected)
        except Exception as e:
            # Don't crash the whole run on quota/errors; fall back to lightweight heuristic
            logger.exception("Gemini error processing SKU %d: %s", sid, e)
            corrected = name.replace("?", "ạ").replace("�", "").strip()
            logger.info(" -> Fallback corrected: %s", corrected)
        results.append((sid, name, corrected))
        if args.update and not args.dry_run and corrected and corrected != name:
            try:
                update_sku_name(sid, corrected)
                logger.info("Updated SKU %d", sid)
            except Exception as e:
                logger.exception("Failed to update SKU %d: %s", sid, e)

    # Write a small report to stdout
    print("\nSummary:")
    for sid, orig, corr in results:
        changed = orig != corr
        print(f"{sid}: {'[CHANGED]' if changed else '[UNCHANGED]'}\n  orig: {orig}\n  corr: {corr}\n")


if __name__ == "__main__":
    main()
