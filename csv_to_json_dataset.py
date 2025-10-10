#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_to_json_dataset.py
----------------------
Reusable script to:
1) Convert a CSV file to JSON and/or JSONL
2) (Optional) Build a Qwen-style multimodal dataset (messages + images) for fine-tuning

Examples
--------
# RAW export (default mode)
python csv_to_json_dataset.py \
  --input /path/to/sku_captions_export.csv \
  --delimiter ";" \
  --json-out /path/to/sku_captions_export.json \
  --jsonl-out /path/to/sku_captions_export.jsonl

# Qwen dataset export
python csv_to_json_dataset.py \
  --input /path/to/sku_captions_export.csv \
  --mode qwen \
  --delimiter ";" \
  --caption-col caption_text \
  --image-col image_path \
  --base-image-dir /path/to/uploads \
  --jsonl-out /path/to/qwen_finetune_dataset.jsonl

# Qwen dataset with optional image download (when image paths are URLs)
python csv_to_json_dataset.py \
  --input /path/to/sku_captions_export.csv \
  --mode qwen \
  --download-from-url \
  --base-image-dir /path/to/downloaded_images \
  --jsonl-out /path/to/qwen_finetune_dataset.jsonl
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Pandas is required
try:
    import pandas as pd
except Exception as e:
    print("This script requires pandas. Please install it: pip install pandas", file=sys.stderr)
    raise

# Optional imports for downloading/validating images (only used when --download-from-url)
def _lazy_import_requests():
    try:
        import requests  # type: ignore
        return requests
    except Exception:
        return None

def _lazy_import_pil():
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception:
        return None


def read_csv_safely(path: str, delimiter: str, encoding: Optional[str]) -> "pd.DataFrame":
    tries = [encoding] if encoding else []
    # sensible fallbacks
    for enc in ["utf-8", "utf-8-sig", "cp1252"]:
        if enc not in tries:
            tries.append(enc)
    last_err = None
    for enc in tries:
        try:
            df = pd.read_csv(path, sep=delimiter, encoding=enc)
            print(f"[read_csv] Loaded with encoding='{enc}' and delimiter='{delimiter}'. Rows={len(df)} Cols={len(df.columns)}")
            return df
        except Exception as e:
            last_err = e
            continue
    print(f"[read_csv] Failed to read CSV with tried encodings {tries}.", file=sys.stderr)
    if last_err:
        raise last_err
    raise RuntimeError("Failed to read CSV.")


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def write_json(records: List[Dict[str, Any]], out_path: str, minify: bool):
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        if minify:
            json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[write_json] Wrote {len(records)} records -> {out_path}")


def write_jsonl(records: List[Dict[str, Any]], out_path: str):
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[write_jsonl] Wrote {len(records)} records -> {out_path}")


def is_url(s: str) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))


def download_image(url: str, save_path: str) -> Optional[str]:
    """Download an image URL to save_path. Validate with PIL if available."""
    requests = _lazy_import_requests()
    if requests is None:
        print("[download_image] requests not available; skipping download.", file=sys.stderr)
        return None
    try:
        ensure_dir(save_path)
        print(f"[download_image] GET {url}")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        Image = _lazy_import_pil()
        if Image is not None:
            try:
                Image.open(save_path).verify()
            except Exception as ve:
                print(f"[download_image] PIL verify failed: {ve}. Removing file.", file=sys.stderr)
                try:
                    os.remove(save_path)
                except Exception:
                    pass
                return None
        return save_path
    except Exception as e:
        print(f"[download_image] Failed: {e}", file=sys.stderr)
        return None


# --- Helpers for Qwen dataset mode ---
_QWEN_FACET_KEYS = [
    "caption", "keywords", "colors", "shapes", "materials",
    "packaging", "taste", "texture",
    "brand_guess", "variant_guess", "size_guess", "category_guess",
    "facet_scores"
]


def parse_listish(value: Any, is_scores: bool = False) -> Any:
    """
    Turn a raw cell into a list/dict where possible.
    - If already list/dict -> return as-is
    - If str -> try json.loads; otherwise split by comma
    - Fallbacks: [] for lists, {} for scores
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {} if is_scores else []
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return {} if is_scores else []
        
        # Try JSON first
        try:
            parsed = json.loads(v)
            if is_scores:
                # Convert facet_scores array to dict for easier use
                if isinstance(parsed, list):
                    scores_dict = {}
                    for item in parsed:
                        if isinstance(item, dict) and 'facet' in item and 'score' in item:
                            scores_dict[item['facet']] = item['score']
                    return scores_dict
                elif isinstance(parsed, dict):
                    return parsed
                else:
                    return {}
            else:
                if isinstance(parsed, list):
                    return parsed
        except Exception:
            pass
        
        # Fallback: comma-split (for list fields only)
        if not is_scores:
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return parts
        return {}
    
    # Other types
    return {} if is_scores else [str(value)]


def build_facets_from_row(row: "pd.Series") -> Dict[str, Any]:
    facets: Dict[str, Any] = {}
    # Basic text fields
    facets["caption"] = str(row.get("caption_text", "") or "").strip() or "Ảnh sản phẩm không có mô tả"

    # Lists
    for k in ["keywords", "colors", "shapes", "materials", "packaging", "taste", "texture"]:
        facets[k] = parse_listish(row.get(k), is_scores=False)

    # Scores (dict)
    facets["facet_scores"] = parse_listish(row.get("facet_scores"), is_scores=True)

    # Nullable strings
    for k in ["brand_guess", "variant_guess", "size_guess", "category_guess"]:
        v = row.get(k)
        v = None if (v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "") else str(v).strip()
        facets[k] = v

    # Ensure required lists have at least 1 value
    for k in ["keywords", "colors", "shapes", "materials"]:
        if not facets.get(k):
            facets[k] = ["không xác định"]

    return facets


def make_qwen_record(
    row: "pd.Series",
    image_col: str,
    caption_col: str,
    base_image_dir: Optional[str],
    download_from_url: bool,
    system_prompt: str
) -> Optional[Dict[str, Any]]:
    # Resolve image path/filename
    raw_path = str(row.get(image_col, "") or "").strip()
    if not raw_path:
        return None

    # If it's a URL and downloading is enabled
    filename = os.path.basename(raw_path)
    resolved_path = raw_path

    if is_url(raw_path):
        if download_from_url:
            if not base_image_dir:
                print("[qwen] --download-from-url set but --base-image-dir missing; skipping row.", file=sys.stderr)
                return None
            local_dir = os.path.abspath(base_image_dir)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            if not os.path.exists(local_path):
                saved = download_image(raw_path, local_path)
                if saved is None:
                    return None
            resolved_path = local_path
        else:
            # keep URL, but messages expect a local filename; we keep just the filename
            pass
    else:
        # Local or relative path
        if base_image_dir and not os.path.isabs(raw_path):
            candidate = os.path.join(base_image_dir, raw_path)
        else:
            candidate = raw_path
        if os.path.exists(candidate):
            resolved_path = candidate
            filename = os.path.basename(candidate)
        else:
            # If file not found, still keep the filename so you can fix paths later
            filename = os.path.basename(raw_path)

    # Build user prompt
    caption_text = str(row.get(caption_col, "") or "").strip()
    user_prompt = (
        "Schema JSON bắt buộc:\n"
        '{ "caption": "...", "keywords": [], "colors": [], "shapes": [], "materials": [], '
        '"packaging": [], "taste": [], "texture": [], '
        '"brand_guess": null, "variant_guess": null, "size_guess": null, "category_guess": null, '
        '"facet_scores": {} }\n'
        "Nguồn:\n"
        f"Caption: '{caption_text}'\n"
        "OCR: ''\n"
        "Yêu cầu: Trích xuất facet chính xác từ caption và OCR. "
        "Điền các trường keywords, colors, materials, shapes BẮT BUỘC có ít nhất một giá trị, "
        "dựa trên caption hoặc suy luận hợp lý (ví dụ: nếu không rõ màu thì dùng 'không xác định', "
        "nếu không rõ chất liệu thì dùng 'nhựa' hoặc 'vải' tùy danh mục). "
        "Các trường khác để rỗng hoặc null nếu không có dữ liệu. Trả đúng JSON."
    )

    # Build facets object from row
    facets = build_facets_from_row(row)

    record = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": filename},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(facets, ensure_ascii=False)}],
            },
        ],
        "images": [filename],
    }
    return record


def run_raw_mode(df: "pd.DataFrame", args):
    # Just dump each row to dict
    records = df.to_dict(orient="records")
    if args.json_out:
        write_json(records, args.json_out, minify=args.minify)
    if args.jsonl_out:
        write_jsonl(records, args.jsonl_out)
    if not (args.json_out or args.jsonl_out):
        print("[raw] Nothing to write; please specify --json-out and/or --jsonl-out.", file=sys.stderr)


def run_qwen_mode(df: "pd.DataFrame", args):
    out_records: List[Dict[str, Any]] = []
    total = len(df)
    for idx, row in df.iterrows():
        # filter missing caption or image if requested
        if args.require_caption and not str(row.get(args.caption_col, "") or "").strip():
            continue
        if args.require_image and not str(row.get(args.image_col, "") or "").strip():
            continue

        rec = make_qwen_record(
            row=row,
            image_col=args.image_col,
            caption_col=args.caption_col,
            base_image_dir=args.base_image_dir,
            download_from_url=args.download_from_url,
            system_prompt=args.system_prompt
        )
        if rec is not None:
            out_records.append(rec)

    print(f"[qwen] Built {len(out_records)}/{total} records.")

    if args.json_out:
        write_json(out_records, args.json_out, minify=args.minify)
    if args.jsonl_out:
        write_jsonl(out_records, args.jsonl_out)
    if not (args.json_out or args.jsonl_out):
        print("[qwen] Nothing to write; please specify --json-out and/or --jsonl-out.", file=sys.stderr)


def build_argparser():
    p = argparse.ArgumentParser(description="CSV -> JSON/JSONL exporter (+ optional Qwen dataset builder).")

    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--delimiter", default=";", help="CSV delimiter (default=';')")
    p.add_argument("--encoding", default=None, help="CSV encoding (default: auto-try utf-8, utf-8-sig, cp1252)")
    p.add_argument("--mode", choices=["raw", "qwen"], default="raw", help="Export mode: raw or qwen (default=raw)")

    p.add_argument("--json-out", default=None, help="Path to write JSON array")
    p.add_argument("--jsonl-out", default=None, help="Path to write JSONL (one json per line)")
    p.add_argument("--minify", action="store_true", help="Minify JSON output (no pretty indent)")

    # Qwen-specific options
    p.add_argument("--image-col", default="image_path", help="Column name for image path/URL (default='image_path')")
    p.add_argument("--caption-col", default="caption_text", help="Column name for caption text (default='caption_text')")
    p.add_argument("--base-image-dir", default=None, help="Base folder for images (joins when image path is relative)")
    p.add_argument("--download-from-url", action="store_true", help="Download images if image path is a URL")
    p.add_argument("--require-caption", action="store_true", help="Skip rows without caption")
    p.add_argument("--require-image", action="store_true", help="Skip rows without image path")
    p.add_argument("--system-prompt", default="Bạn là AI trích xuất facet sản phẩm từ văn bản tiếng Việt.", help="System prompt for Qwen messages")

    return p


def main():
    args = build_argparser().parse_args()

    if not os.path.exists(args.input):
        print(f"[error] Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = read_csv_safely(args.input, delimiter=args.delimiter, encoding=args.encoding)

    if args.mode == "raw":
        run_raw_mode(df, args)
    else:
        # Ensure needed columns exist (warn if not)
        for col in [args.image_col, args.caption_col]:
            if col not in df.columns:
                print(f"[warn] Column '{col}' not found in CSV. Existing columns: {list(df.columns)}", file=sys.stderr)
        run_qwen_mode(df, args)


if __name__ == "__main__":
    main()
