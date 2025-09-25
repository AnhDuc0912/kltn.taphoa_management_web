import os
import base64
import io
import time
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import logging
import traceback
from logging.handlers import RotatingFileHandler

# Khởi tạo logger
logger = logging.getLogger("qwen2vl_autogen")

# Thiết lập logging
if not logger.handlers:
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fh = RotatingFileHandler(os.path.join(logs_dir, "qwen2vl_autogen.log"), maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Cấu hình
BASE_MODEL = os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-7B-Instruct")
ADAPTER_DIR = os.getenv("QWEN_VL_ADAPTER", r"e:\api_hango\flask_pgvector_shop\flask_pgvector_shop\out-qwen2vl-lora")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Tải processor và mô hình
print("Đang tải processor và mô hình gốc...", flush=True)
processor = None
processor_source_used = BASE_MODEL
try:
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.tokenizer = tokenizer
    logger.info("Đã tải processor và tokenizer từ mô hình gốc %s", BASE_MODEL)
except Exception as e:
    logger.exception("Không thể tải processor/tokenizer từ mô hình gốc %s: %s", BASE_MODEL, e)
    raise

# Tải mô hình Qwen2-VL
print("Đang tải mô hình gốc (có thể tải trọng số lớn)...", flush=True)
base = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
)

# Tạm bỏ LoRA để kiểm tra, uncomment nếu cần
# print("Đang gắn adapter LoRA (PEFT)...", flush=True)
# model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map="auto" if DEVICE == "cuda" else None)
model = base
model.eval()

def image_to_data_uri(pil_img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def _build_inputs_with_processor(pil_img: Image.Image, text: str):
    """
    Gọi processor(text=..., images=...) một cách an toàn.
    """
    global processor, processor_source_used
    try:
        inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt")
        logger.info("Token hóa input_ids: %s", processor.tokenizer.decode(inputs["input_ids"][0]))
        logger.info("Shape đặc trưng hình ảnh: %s", inputs.get("pixel_values", "N/A").shape if "pixel_values" in inputs else "N/A")
        return inputs
    except TypeError as e:
        logger.warning("Processor từ %s không chấp nhận hình ảnh: %s. Thử processor dự phòng", processor_source_used, e)
        try:
            processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            processor.tokenizer = tokenizer
            processor_source_used = BASE_MODEL
            logger.info("Đã tải processor dự phòng từ %s", BASE_MODEL)
            inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt")
            logger.info("Token hóa input_ids: %s", processor.tokenizer.decode(inputs["input_ids"][0]))
            return inputs
        except Exception as e2:
            logger.error("Processor dự phòng thất bại: %s", e2)
            raise

def generate_caption_from_image(pil_img: Image.Image, prompt: str, max_new_tokens=128, temperature=0.2) -> str:
    try:
        # Kiểm tra và chuẩn hóa hình ảnh
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize((448, 448), Image.Resampling.LANCZOS)
        logger.info("Đã resize hình ảnh về %s, mode=%s", pil_img.size, pil_img.mode)

        # Format messages đúng cho Qwen2-VL
        messages = [
            {"role": "system", "content": "Bạn là một AI hữu ích, trả lời ngắn gọn bằng tiếng Việt."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt + " Trả lời duy nhất caption ngắn (1-2 câu) bằng tiếng Việt."}
            ]}
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        logger.info("Prompt đã chuẩn bị: %s", text)

        # Tạo inputs
        inputs = _build_inputs_with_processor(pil_img, text)

        # Di chuyển tensor đến device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logger.info("Tóm tắt inputs trước khi tạo: %s", 
                    {k: (v.shape if isinstance(v, torch.Tensor) else str(type(v))) for k, v in inputs.items()})

        # Generate
        try:
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature)
        except Exception as ge:
            msg = str(ge)
            logger.error("Generate thất bại: %s\nTraceback:\n%s", ge, traceback.format_exc())
            raise

        # Giải mã đầu ra
        generated_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        text = processor.decode(generated_ids, skip_special_tokens=True).strip()

        logger.info("Tạo caption thành công | len=%d", len(text or ""))
        return text
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Lỗi trong generate_caption_from_image: %s\nPrompt: %r\nImage.size=%s\nTraceback:\n%s", 
                     e, (prompt or "")[:200], getattr(pil_img, "size", None), tb)
        raise

def split_segments(text: str) -> List[str]:
    import re
    segs = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return segs if segs else [text.strip()]

def embed_text(text: str) -> List[float]:
    vec = EMBED_MODEL.encode(text, normalize_embeddings=True)
    return vec.tolist()

def process_images(paths: List[str], prompt_search: str = "Viết caption ngắn 1-2 câu, khách quan, tiếng Việt."):
    for i, p in enumerate(paths, start=1):
        try:
            print(f"[{i}/{len(paths)}] Đang tải {p}", flush=True)
            pil = Image.open(p).convert("RGB")
            caption = generate_caption_from_image(pil, prompt_search)
            print(f"Caption gốc: {caption}", flush=True)
            segs = split_segments(caption)
            for j, s in enumerate(segs, start=1):
                print(f"  segment {j}/{len(segs)}: {s}", flush=True)
                vec = embed_text(s)
                print(f"    embedding len={len(vec)}", flush=True)
            time.sleep(0.5)
        except Exception as ex:
            print("Lỗi:", ex, flush=True)

if __name__ == "__main__":
    import sys, glob
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = glob.glob(os.path.join(os.getcwd(), "Uploads", "*.png"))[:50]
    process_images(paths)