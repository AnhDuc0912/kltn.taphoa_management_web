import os
import base64
import io
import time
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration, GenerationConfig
from typing import List
from sentence_transformers import SentenceTransformer
from peft import PeftModel
import logging, traceback
from logging.handlers import RotatingFileHandler

# ================= Logging =================
logger = logging.getLogger("qwen2vl_autogen")
if not logger.handlers:
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fh = RotatingFileHandler(os.path.join(logs_dir, "qwen2vl_autogen.log"),
                             maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.WARNING)  # ít spam hơn
    logger.addHandler(fh); logger.addHandler(ch)
    logger.setLevel(logging.INFO); logger.propagate = False

# ================= Config =================
BASE_MODEL  = os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct")  # 2B mặc định
ADAPTER_DIR = os.getenv("QWEN_VL_ADAPTER", r"e:\api_hango\flask_pgvector_shop\flask_pgvector_shop\out-qwen2vl-lora")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT    = os.getenv("QWEN_LOAD_4BIT", "0") == "1"   # bật 4-bit nếu cần
USE_LORA    = os.getenv("QWEN_USE_LORA", "0") == "1"    # gắn PEFT adapter nếu cần

# Sentence-Transformers cho embedding
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ================= Processor & Tokenizer =================
print("Đang tải processor/tokenizer...", flush=True)
processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False
)
tokenizer  = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False
)
tokenizer.padding_side = "left"  # ổn định khi batch
processor.tokenizer = tokenizer
logger.info("Processor/tokenizer sẵn từ %s", BASE_MODEL)

# ================= Model Loading (2B) =================
print("Đang tải mô hình 2B...", flush=True)

# Tùy chọn 4-bit nếu có bitsandbytes
quant_config = None
if USE_4BIT:
    try:
        import bitsandbytes as bnb  # noqa: F401
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(torch.float16 if DEVICE == "cuda" else torch.float32),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Bật 4-bit quantization (nf4).")
    except Exception as e:
        logger.warning("Không bật được 4-bit (thiếu bitsandbytes?). Fallback FP16/FP32. Lỗi: %s", e)
        quant_config = None

torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
device_map  = "auto" if DEVICE == "cuda" else None

base = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    device_map=device_map,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager",    # tránh yêu cầu flash-attn
    quantization_config=quant_config,
)

# Gắn LoRA nếu có (mặc định: tắt)
if USE_LORA and os.path.isdir(ADAPTER_DIR):
    print("Đang gắn adapter LoRA (PEFT)...", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map=device_map)
else:
    model = base

# một số cờ an toàn
if getattr(model.config, "use_cache", None) is not None:
    model.config.use_cache = False
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

# ⚠️ Reset generation_config để loại các khóa không hỗ trợ (hết cảnh báo top_p/top_k)
model.generation_config = GenerationConfig.from_model_config(model.config)

model.eval()
logger.info("Model đã sẵn sàng: %s", BASE_MODEL)

# ================= Helpers =================
def image_to_data_uri(pil_img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def _build_inputs_with_processor(pil_img: Image.Image, chat_text: str):
    """
    Với Qwen2-VL: apply_chat_template(..., tokenize=False) -> rồi processor(text=[...], images=[...]).
    """
    try:
        inputs = processor(text=[chat_text], images=[pil_img], return_tensors="pt")
        # log ngắn để tránh spam
        dec_preview = tokenizer.decode(inputs["input_ids"][0][:64], skip_special_tokens=False)
        logger.info("input_ids preview: %s", dec_preview)
        # pixel_values có thể là Tensor hoặc list[Tensor] tùy phiên bản
        pv = inputs.get("pixel_values", None)
        try:
            shape_info = tuple(pv.shape) if isinstance(pv, torch.Tensor) else (len(pv),) if isinstance(pv, list) else None
            logger.info("pixel_values.shape: %s", shape_info)
        except Exception:
            pass
        return inputs
    except Exception as e:
        logger.error("Build inputs thất bại: %s", e)
        raise

def _to_device(batch, device):
    """Đưa batch (Tensor/list/dict) lên đúng device một cách an toàn."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, list):
        return [_to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch

def generate_caption_from_image(pil_img: Image.Image, prompt: str, max_new_tokens=96, temperature=0.0) -> str:
    try:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # messages đúng chuẩn Qwen2-VL
        messages = [
            {"role": "system", "content": "Bạn là một AI hữu ích, trả lời ngắn gọn bằng tiếng Việt."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt + " Trả lời một caption ngắn 1-2 câu bằng tiếng Việt."}
            ]}
        ]

        # Lấy string prompt (không tokenize ở đây)
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = _build_inputs_with_processor(pil_img, chat_text)

        # đẩy tensors lên device của model (robust)
        device = next(model.parameters()).device
        inputs = _to_device(inputs, device)

        logger.info("Inputs summary: %s", {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__)
                                            for k, v in inputs.items()})

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=(temperature if temperature > 0 else None),
            repetition_penalty=1.05,
        )

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})

        # cắt phần prompt, chỉ lấy phần sinh
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, in_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        logger.info("Caption OK | len=%d", len(text))
        return text
    except Exception as e:
        logger.error("Lỗi generate_caption_from_image: %s\nTraceback:\n%s", e, traceback.format_exc())
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
            print(f"Caption: {caption}", flush=True)
            for j, s in enumerate(split_segments(caption), start=1):
                print(f"  segment {j}: {s}", flush=True)
                vec = embed_text(s)
                print(f"    embedding len={len(vec)}", flush=True)
            time.sleep(0.1)
        except Exception as ex:
            print("Lỗi:", ex, flush=True)

def quick_test(img_path: str):
    pil = Image.open(img_path).convert("RGB")
    cap = generate_caption_from_image(
        pil,
        prompt="Mô tả sản phẩm ngắn gọn, khách quan, tiếng Việt.",
        max_new_tokens=64,
        temperature=0.0
    )
    print(">>> CAPTION:", cap)

# ================= Main =================
if __name__ == "__main__":
    import sys, glob

    # Đầu tiên test 1 ảnh để xác nhận output
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = glob.glob(os.path.join(os.getcwd(), "uploads", "*.png"))[:1]

    if not paths:
        print("Không tìm thấy ảnh test trong ./uploads/*.png")
    else:
        quick_test(paths[0])
        # Khi đã OK, bỏ comment dòng dưới để chạy hàng loạt:
        # process_images(paths)