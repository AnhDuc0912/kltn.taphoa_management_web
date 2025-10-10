"""
Simple Gemini caption generator - trả về đầy đủ metadata theo schema sku_captions
"""
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def get_mime(path: str) -> str:
    """Guess MIME type from file extension"""
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"

# Prompt yêu cầu JSON đầy đủ metadata
PROMPT_FULL_JSON = """Bạn là trợ lý phân tích sản phẩm. Hãy phân tích ảnh và TRẢ VỀ JSON duy nhất (không thêm markdown hay text ngoài JSON).

Schema JSON:
{
  "caption_vi": "Mô tả ngắn gọn 1-2 câu bằng tiếng Việt",
  "keywords": ["từ khóa 1", "từ khóa 2"],
  "colors": ["màu 1", "màu 2"],
  "shapes": ["hình dạng 1"],
  "materials": ["vật liệu 1"],
  "packaging": ["loại bao bì"],
  "taste": ["hương vị nếu là thực phẩm"],
  "texture": ["kết cấu nếu có"],
  "brand_guess": "tên thương hiệu (hoặc null)",
  "variant_guess": "biến thể sản phẩm (hoặc null)",
  "size_guess": "kích thước/dung tích (hoặc null)",
  "category_guess": "danh mục sản phẩm",
  "facet_scores": [
    {"facet": "tên_facet", "score": 0.85}
  ]
}

YÊU CẦU:
- Không bịa dữ liệu; nếu không chắc thì để null hoặc mảng rỗng []
- Tất cả text bằng tiếng Việt, chữ thường
- facet_scores: mảng các {facet, score} với score từ 0..1
"""

def generate_full_caption_data(
    image_path: str,
    ocr_text: Optional[str] = None,
    sku_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate full caption metadata using Gemini.
    Returns a dict matching sku_captions schema (without DB-only fields like id/created_at).
    """
    if genai is None:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")

    api_key = os.getenv("API_KEY_GEMINI")
    if not api_key:
        raise RuntimeError("API_KEY_GEMINI not set in .env")

    genai.configure(api_key=api_key)

    model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    ]

    model = genai.GenerativeModel(model_name)

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime = get_mime(image_path)

    parts = [PROMPT_FULL_JSON]
    if ocr_text:
        parts.append(f"OCR text từ ảnh: {ocr_text}")
    parts.append({"mime_type": mime, "data": img_bytes})

    retry = 3
    for attempt in range(1, retry + 1):
        try:
            res = model.generate_content(
                parts,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                    "max_output_tokens": 512
                },
                safety_settings=safety_settings,
                request_options={"timeout": 60},
            )

            candidates = getattr(res, "candidates", [])
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
                if finish_reason == 2:
                    logging.warning("Blocked by safety (finish_reason=2) on attempt %d", attempt)
                    raise RuntimeError("Gemini blocked by safety")

            text = res.text.strip() if hasattr(res, "text") else ""
            if not text:
                logging.warning("Empty response on attempt %d", attempt)
                raise RuntimeError("Empty response")

            # Parse JSON
            data = json.loads(text)

            # Normalize to match schema
            result = {
                "sku_id": sku_id,
                "image_path": str(Path(image_path).name),
                "lang": "vi",
                "style": "search",  # default style
                "caption_text": data.get("caption_vi", ""),
                "model_name": model_name,
                "prompt_version": "v1.0",
                "needs_review": True,
                "keywords": data.get("keywords", []),
                "colors": data.get("colors", []),
                "shapes": data.get("shapes", []),
                "materials": data.get("materials", []),
                "packaging": data.get("packaging", []),
                "taste": data.get("taste", []),
                "texture": data.get("texture", []),
                "brand_guess": data.get("brand_guess"),
                "variant_guess": data.get("variant_guess"),
                "size_guess": data.get("size_guess"),
                "category_guess": data.get("category_guess"),
                "facet_scores": data.get("facet_scores", [])
            }
            return result

        except Exception as e:
            logging.warning("Attempt %d/%d failed: %s", attempt, retry, e)
            time.sleep(0.5 * attempt)

    # Fallback
    logging.error("All attempts failed. Returning minimal data.")
    return {
        "sku_id": sku_id,
        "image_path": str(Path(image_path).name),
        "lang": "vi",
        "style": "search",
        "caption_text": Path(image_path).stem.replace("_", " "),
        "model_name": model_name,
        "prompt_version": "v1.0",
        "needs_review": True,
        "keywords": [],
        "colors": [],
        "shapes": [],
        "materials": [],
        "packaging": [],
        "taste": [],
        "texture": [],
        "brand_guess": None,
        "variant_guess": None,
        "size_guess": None,
        "category_guess": None,
        "facet_scores": []
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate full caption metadata with Gemini")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--ocr", help="OCR text from image (optional)")
    parser.add_argument("--sku-id", type=int, help="SKU ID (optional)")
    args = parser.parse_args()

    img_path = args.image
    if not Path(img_path).exists():
        print(f"❌ Image not found: {img_path}")
        return

    print(f"🖼️  Image: {Path(img_path).name}")
    print("Generating full caption data with Gemini...")

    t0 = time.time()
    data = generate_full_caption_data(img_path, ocr_text=args.ocr, sku_id=args.sku_id)
    elapsed = time.time() - t0

    print(f"\n✅ Generated in {elapsed:.2f}s")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()