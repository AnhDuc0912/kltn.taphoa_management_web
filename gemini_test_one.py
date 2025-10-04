"""
gemini_test_one.py

Use Gemini to generate Vietnamese captions and keywords for given images and prompts.

Usage: set environment variable API_KEY_GEMINI in .env, then run this script.
"""
import os
import re
import json
import time
import logging
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from google.api_core import exceptions as google_api_exceptions
except Exception:
    google_api_exceptions = None

load_dotenv()

# ========= Gemini Config =========
PROMPT_FACETS_JSON = """Bạn là trợ lý thị giác. Hãy phân tích ảnh sản phẩm tiêu dùng và TRẢ VỀ CHUẴN JSON duy nhất.
YÊU CẦU:
- Không thêm bình luận ngoài JSON.
- Nếu không chắc về một trường ⇒ để null hoặc mảng rỗng [] (không bịa).
- caption_vi: 1-2 câu, khách quan, tiếng Việt, mô tả màu sắc, chất liệu, bao bì, kích thước.
- colors/shapes/materials/packaging/taste/texture: mảng từ khóa tiếng Việt, chữ thường.
- brand_guess/variant_guess/size_guess/category_guess: chuỗi ngắn, tiếng Việt (hoặc null).
- facet_scores: các key tự do (ví dụ "color_yellow", "shape_triangle") giá trị 0..1.
- Bỏ qua hoàn toàn nội dung nhạy cảm (rượu bia, thuốc lá, hoặc yếu tố không phù hợp).

MẪU JSON:
{
  "caption_vi": "Chai nhựa màu trong suốt chứa nước suối, dung tích 500ml.",
  "colors": ["vàng", "đỏ"],
  "shapes": ["tam giác", "tròn"],
  "materials": ["nhựa", "kim loại"],
  "packaging": ["túi", "lon", "chai"],
  "taste": ["cay", "phô mai"],
  "texture": ["giòn", "xốp"],
  "brand_guess": "Heineken",
  "variant_guess": "chili cheese",
  "size_guess": "330ml",
  "category_guess": "snack",
  "facet_scores": {
    "color_yellow": 0.92,
    "shape_triangle": 0.77
  }
}"""

def _json_only(text: str) -> str:
    if not text: return ""
    m = re.search(r"\{.*\}\s*$", text, flags=re.S)
    return m.group(0) if m else text.strip()

def _guess_mime_from_ext(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime: return mime
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"): return "image/jpeg"
    if ext == ".png": return "image/png"
    if ext == ".webp": return "image/webp"
    return "application/octet-stream"

def generate_with_gemini(img_path: str, prompt: str, max_new_tokens: int = 64, ocr_text: Optional[str] = None) -> Dict:
    if genai is None:
        raise RuntimeError("google-generativeai chưa cài. Cài bằng: pip install google-generativeai")

    api_key = os.environ.get("API_KEY_GEMINI")
    if not api_key:
        raise RuntimeError("API_KEY_GEMINI không được thiết lập trong .env.")

    genai.configure(api_key=api_key)

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro-latest"]
    retry = int(os.environ.get("GENAI_RETRY", "5"))
    request_timeout = int(os.environ.get("GENAI_REQ_TIMEOUT", "120"))
    last_err = None

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=(
                    "Chỉ trả về JSON hợp lệ mô tả sản phẩm tiêu dùng khách quan (màu sắc, chất liệu, bao bì, kích thước). "
                    "Bỏ qua hoàn toàn nội dung nhạy cảm như rượu bia, thuốc lá, hoặc yếu tố không phù hợp. "
                    "Nếu hình ảnh không thể mô tả do hạn chế an toàn, trả về JSON với caption_vi rỗng và các trường khác rỗng hoặc null."
                ),
            )
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            mime = _guess_mime_from_ext(img_path)

            parts: List = [PROMPT_FACETS_JSON]
            if prompt:
                parts.append(f"Prompt người dùng: {prompt}")
            if ocr_text:
                parts.append(f"OCR nhìn thấy trên bao bì: {ocr_text}")
            parts.append({"mime_type": mime, "data": img_bytes})

            for attempt in range(1, retry + 1):
                try:
                    res = model.generate_content(
                        parts,
                        generation_config={
                            "response_mime_type": "application/json",
                            "temperature": 0.2,
                            "max_output_tokens": max_new_tokens,
                        },
                        safety_settings={
                            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                        },
                        request_options={"timeout": request_timeout},
                    )
                    # Kiểm tra và log chi tiết response
                    if hasattr(res, "candidates") and res.candidates:
                        print(f"Safety ratings for {model_name}:", res.candidates[0].safety_ratings)
                        if hasattr(res.candidates[0], "finish_message"):
                            print(f"Finish message: {res.candidates[0].finish_message}")
                        if res.candidates[0].finish_reason == 2:
                            logging.warning(f"Model {model_name} blocked due to SAFETY (finish_reason=2)")
                            return {
                                "caption_vi": "",
                                "colors": [], "shapes": [], "materials": [], "packaging": [], "taste": [], "texture": [],
                                "brand_guess": None, "variant_guess": None, "size_guess": None, "category_guess": None,
                                "facet_scores": {}
                            }
                    txt = (res.text or "").strip()
                    data = json.loads(_json_only(txt))

                    # Chuẩn hóa data
                    data.setdefault("caption_vi", "")
                    for k in ["colors", "shapes", "materials", "packaging", "taste", "texture"]:
                        if not isinstance(data.get(k), list):
                            data[k] = []
                    for k in ["brand_guess", "variant_guess", "size_guess", "category_guess"]:
                        v = data.get(k)
                        if v is not None and not isinstance(v, str):
                            data[k] = str(v)
                    if not isinstance(data.get("facet_scores"), dict):
                        data["facet_scores"] = {}
                    return data
                except Exception as e:
                    last_err = e
                    logging.exception(f"Model {model_name} attempt {attempt} failed: {e}")
                    time.sleep(0.6 * attempt)
            logging.warning(f"Model {model_name} failed after {retry} attempts")
        except Exception as e:
            logging.warning(f"Failed to instantiate model {model_name}: {e}")
            last_err = e
            continue

    raise RuntimeError(f"All models failed after {len(models_to_try)} attempts: {last_err}")

def build_keywords(data: Dict) -> List[str]:
    out: List[str] = []
    def add_many(items):
        for x in (items or []):
            if not x: continue
            t = str(x).strip().lower()
            if t and t not in out:
                out.append(t)
    add_many(data.get("colors"))
    add_many(data.get("shapes"))
    add_many(data.get("materials"))
    add_many(data.get("packaging"))
    add_many(data.get("taste"))
    add_many(data.get("texture"))
    for k in ["brand_guess", "variant_guess", "size_guess", "category_guess"]:
        v = (data.get(k) or "").strip()
        if v:
            add_many([v])
    fs = data.get("facet_scores") or {}
    add_many([str(k).replace("_", " ").strip().lower() for k in fs.keys()])
    return out

def predict_for_images(image_paths: List[str], prompts: List[str], max_new_tokens: int = 64):
    results = []
    failed_images = []  # Lưu danh sách ảnh bị chặn
    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            print(f"❌ Image not found: {img_path}")
            continue
        for prompt in prompts:
            start = time.time()
            try:
                data = generate_with_gemini(str(p), prompt or "Mô tả sản phẩm tiêu dùng trong ảnh bằng tiếng Việt", max_new_tokens=max_new_tokens)
                caption = data.get("caption_vi", "Không tạo được caption")
                keywords = build_keywords(data)
                used = f"gemini ({os.environ.get('MODEL_NAME', 'gemini-2.5-flash')})"
                elapsed = time.time() - start
                print(f"✅ [{used}] {p.name} | Prompt: '{prompt or '(<empty> Vietnamese)'}' -> {elapsed:.1f}s")
                print(f"   Caption (vi): {caption}")
                print(f"   Keywords: {', '.join(keywords)}\n")
                results.append({
                    "image": str(p),
                    "prompt": prompt,
                    "caption": caption,
                    "backend": used,
                    "time_s": elapsed,
                    "keywords": keywords,
                    "facets": data
                })
                # Nếu bị chặn (caption rỗng), ghi vào failed_images
                if not caption:
                    failed_images.append({
                        "image": str(p),
                        "prompt": prompt,
                        "reason": "SAFETY (finish_reason=2)",
                        "model": os.environ.get("MODEL_NAME", "gemini-2.5-flash")
                    })
            except Exception as e:
                print(f"❌ Failed to generate for {p.name} with prompt '{prompt}': {type(e).__name__}: {e}")
                logging.warning(f"Skipped image {p.name} due to error: {e}")
                failed_images.append({
                    "image": str(p),
                    "prompt": prompt,
                    "reason": str(e),
                    "model": os.environ.get("MODEL_NAME", "gemini-2.5-flash")
                })
                results.append({
                    "image": str(p),
                    "prompt": prompt,
                    "caption": "",
                    "backend": used,
                    "time_s": elapsed,
                    "keywords": [],
                    "facets": {"caption_vi": "", "colors": [], "shapes": [], "materials": [], "packaging": [], "taste": [], "texture": [], "brand_guess": None, "variant_guess": None, "size_guess": None, "category_guess": None, "facet_scores": {}}
                })
                continue
    # Lưu danh sách ảnh lỗi vào file
    if failed_images:
        with open("failed_images.json", "w", encoding="utf-8") as f:
            json.dump(failed_images, f, ensure_ascii=False, indent=2)
        print(f"Đã ghi {len(failed_images)} ảnh lỗi vào failed_images.json")
    return results

if __name__ == "__main__":
    # Example default inputs; adjust paths and prompts as needed.
    images = [
        os.environ.get("TEST_IMAGE", r"E:\api_hango\flask_pgvector_shop\flask_pgvector_shop\uploads\24_43289c3340154dd0a75e9cbc4f5a7ee5.jpg")
    ]
    prompts = [
        "Mô tả sản phẩm tiêu dùng trong ảnh bằng 1-2 câu tiếng Việt, chỉ tập trung vào màu sắc, chất liệu, bao bì, và kích thước (nếu rõ ràng). Không đề cập đến rượu bia, thuốc lá, hoặc bất kỳ nội dung nhạy cảm nào. Ví dụ: 'Chai nhựa màu trong suốt chứa nước suối, dung tích 500ml.'"
    ]
    max_tokens = int(os.environ.get("CAPTION_MAX_TOKENS", "80"))

    print("Starting caption prediction (Vietnamese) with Gemini...")
    predict_for_images(images, prompts, max_new_tokens=max_tokens)