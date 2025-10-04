import os
import re
import logging
import tempfile
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# env keys
GGUF_MODEL = os.getenv("QWEN_GGUF", "")
MMPROJ = os.getenv("QWEN_MMPROJ", "")
HF_BASE   = os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct")
BACKEND   = os.getenv("QWEN_VL_BACKEND", "").lower()  # 'llamacpp' or '' (transformers)
IMG_MAX_SIDE = int(os.getenv("IMG_MAX_SIDE", "1280"))
LLAMA_THREADS = int(os.getenv("LLAMA_THREADS", "4"))
QWEN_USE_LORA = os.getenv("QWEN_USE_LORA", "0") == "1"
QWEN_LORA_PATH = os.getenv("QWEN_LORA_PATH", "")

# caches
_llama_model = None
_hf_model = None
_hf_processor = None

def _compress(img: Image.Image, max_side: int = IMG_MAX_SIDE) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def _to_file_url(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.name == "nt":
        return "file:///" + abs_path.replace("\\", "/")
    return "file://" + abs_path

def _load_llama():
    global _llama_model
    if _llama_model is not None:
        return _llama_model
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
    except Exception as e:
        raise RuntimeError("llama-cpp-python not installed") from e

    if not GGUF_MODEL or not os.path.isfile(GGUF_MODEL):
        raise RuntimeError(f"GGUF model not found: {GGUF_MODEL}")

    chat_handler = None
    if MMPROJ and os.path.isfile(MMPROJ):
        try:
            chat_handler = Llava15ChatHandler(clip_model_path=MMPROJ)
            logger.info("Loaded mmproj for vision support")
        except Exception:
            logger.warning("Failed to load mmproj; continuing without vision handler")

    _llama_model = Llama(
        model_path=GGUF_MODEL,
        chat_handler=chat_handler,
        n_ctx=4096,
        n_threads=LLAMA_THREADS,
        n_gpu_layers=0,
        verbose=False
    )
    logger.info("LLama (gguf) loaded")
    return _llama_model

def _gen_llama_caption(img_path: str, prompt: str, max_tokens: int = 80) -> str:
    model = _load_llama()
    # if handler exists, use image_url
    if hasattr(model, "chat_handler") and model.chat_handler:
        image_url = _to_file_url(img_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": f"Mô tả ngắn gọn nội dung ảnh này bằng tiếng Việt: {prompt.strip()}"}
            ]
        }]
        try:
            res = model.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.3, stop=["\n\n"])
            text = res["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("Vision chat failed, falling back to text-only: %s", e)
            text = ""
    else:
        ptxt = f"Mô tả sản phẩm: {prompt.strip()}"
        try:
            res = model(ptxt, max_tokens=max_tokens, temperature=0.3, stop=["\n", "User:", "Assistant:"], echo=False)
            text = res["choices"][0]["text"].strip()
        except Exception as e:
            logger.error("llama text generate failed: %s", e)
            text = ""

    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) < 4 or re.match(r'^[0-9\s]+$', text):
        # fallback simple caption
        return f"Sản phẩm {prompt.strip()}" if prompt.strip() else "Sản phẩm trong hình ảnh"
    return text

def _load_hf():
    global _hf_model, _hf_processor
    if _hf_model and _hf_processor:
        return _hf_processor, _hf_model
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        import torch
    except Exception as e:
        raise RuntimeError("transformers or peft not installed or missing components") from e

    logger.info("Loading HF processor and model from %s ...", HF_BASE)
    _hf_processor = AutoProcessor.from_pretrained(HF_BASE, trust_remote_code=True, use_fast=True)
    _hf_model = Qwen2VLForConditionalGeneration.from_pretrained(HF_BASE, trust_remote_code=True, dtype=torch.float32).to("cpu")
    _hf_model.eval()

    # Load LoRA if enabled
    if QWEN_USE_LORA and QWEN_LORA_PATH and Path(QWEN_LORA_PATH).exists():
        logger.info(f"Loading LoRA checkpoint from {QWEN_LORA_PATH}")
        _hf_model = PeftModel.from_pretrained(_hf_model, QWEN_LORA_PATH)
        logger.info("LoRA applied successfully. Model is now PeftModelForCausalLM")
    else:
        logger.info("LoRA not enabled or path not found. Using base model only.")

    return _hf_processor, _hf_model

def _gen_hf_caption(img: Image.Image, prompt: str, max_new_tokens: int = 80) -> str:
    processor, model = _load_hf()
    # Build input for Qwen2-VL
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt or "Mô tả ảnh này bằng tiếng Việt"}
            ]
        }
    ]
    # Apply chat template with tokenize=False to ensure string output
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    logger.info(f"Generated text prompt (type: {type(text_prompt)}): {text_prompt}")

    # Prepare inputs
    inputs = processor(
        text=[text_prompt],  # Wrap in list to avoid 'list' callable error
        images=[img],
        return_tensors="pt"
    ).to("cpu")
    logger.info(f"Input keys: {list(inputs.keys())}")

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode output
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    gen_ids = outputs[0, input_len:] if input_len and outputs.ndim == 2 and outputs.shape[1] > input_len else outputs[0]
    caption = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    caption = re.sub(r"\s+", " ", caption)
    return caption if caption else (f"Sản phẩm {prompt.strip()}" if prompt.strip() else "Sản phẩm trong hình ảnh")

def generate_caption(
    pil_img: Image.Image,
    prompt: Optional[str] = "",
    max_new_tokens: int = 80,
    preferred_backend: Optional[str] = None
) -> str:
    """
    Generate caption using local Qwen checkpoint.
    preferred_backend: 'llamacpp' or 'hf' or None (choose by BACKEND env then fallback)
    """
    backend = preferred_backend or BACKEND
    img = _compress(pil_img, IMG_MAX_SIDE)

    # try chosen backend first, then fallback
    order = [backend] if backend else []
    # append env default BACKEND then both
    if not order:
        order = [os.getenv("QWEN_VL_BACKEND", "").lower(), "llamacpp", "hf"]
    # dedupe and keep only llama/hf
    seen = set(); order2 = []
    for b in order:
        b = (b or "").lower()
        if b in ("llamacpp", "hf") and b not in seen:
            order2.append(b); seen.add(b)

    for b in order2:
        try:
            if b == "llamacpp":
                # save temp file to give llama a path
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img.save(tmp.name, "JPEG", quality=85, optimize=True)
                    tmp_path = tmp.name
                try:
                    return _gen_llama_caption(tmp_path, prompt or "", max_tokens=max_new_tokens)
                finally:
                    try: os.unlink(tmp_path)
                    except: pass
            elif b == "hf":
                return _gen_hf_caption(img, prompt or "", max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.warning("Backend %s failed: %s", b, e, exc_info=False)
            continue

    # ultimate fallback
    return f"Sản phẩm {prompt.strip()}" if prompt and prompt.strip() else "Sản phẩm trong hình ảnh"

if __name__ == "__main__":
    # simple CLI test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qwen2vl_local.py /path/to/image.jpg [prompt]")
        sys.exit(1)
    img_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Mô tả ảnh này bằng tiếng Việt"
    img = Image.open(img_path)
    print("Generating caption...")
    caption = generate_caption(img, prompt, max_new_tokens=80)
    print("Caption:", caption)