# predict_caption_vn.py  (2-pass: caption -> facets JSON)  [GPU-ready, LoRA-ready]
import os, sys, time, json, re, unicodedata, ast
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image

# ===================== Env & safety =====================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ===================== Torch / Device helpers =====================
def _has_torch():
    try:
        import torch
        return True
    except Exception:
        return False

def _pick_device(env_device: Optional[str] = None) -> str:
    if not _has_torch():
        return "cpu"
    import torch, platform
    if os.getenv("DISABLE_CUDA", "").strip() == "1":
        return "cpu"
    if env_device in ("cuda", "cpu"):
        return env_device
    if platform.system() == "Darwin":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _pick_dtype(device: str):
    import torch
    return torch.float16 if device == "cuda" else torch.float32

def _log_env(device: str):
    try:
        if not _has_torch():
            print("[INFO] PyTorch not installed -> running on CPU")
            return
        import torch
        info = {"device": device, "torch": torch.__version__}
        if device == "cuda":
            info.update({
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "sm_capability": torch.cuda.get_device_capability(0),
            })
        print("[CUDA ENV]", info)
    except Exception as e:
        print("[WARN] Failed to log CUDA env:", e)

# ===================== Utils =====================
def _vn_norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 \-\.x/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dedup_list_str(xs: Any, max_items: int = 12) -> List[str]:
    out, seen = [], set()
    if isinstance(xs, (list, tuple)):
        for x in xs:
            if not x: continue
            s = str(x).strip()
            if not s: continue
            key = _vn_norm(s)
            if key and key not in seen:
                seen.add(key)
                out.append(s)
            if len(out) >= max_items:
                break
    elif isinstance(xs, str) and xs.strip():
        out = [xs.strip()]
    return out or ["không xác định"]

def _ensure_str(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    return s or None

def _safe_json_extract(txt: str) -> Dict[str, Any]:
    txt = (txt or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    try:
        stack, best = [], None
        for i, ch in enumerate(txt):
            if ch == "{":
                stack.append(i)
            elif ch == "}" and stack:
                j = stack.pop()
                cand = txt[j:i + 1]
                if best is None or len(cand) > len(best):
                    best = cand
        if best:
            return json.loads(best)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

def _regex_extract_size(s: str) -> Optional[str]:
    pat = r"(\d+(?:[.,]\d+)?\s?(?:g|kg|ml|l|lit|lít|oz|pack|bịch|hộp))"
    m = re.search(pat, (s or "").lower())
    return m.group(1) if m else None

# ===================== Transformers backend =====================
_MODEL_CACHE: Dict[str, Any] = {}

def _cache_key(base_or_lora: str, device: str, dtype) -> str:
    return f"{base_or_lora}::{device}::{str(dtype)}"

def _load_processor_model_vl(base_model: str, device: Optional[str] = None):
    device = _pick_device(device)
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = _pick_dtype(device)
    use_lora = os.getenv("QWEN_USE_LORA", "0") == "1"
    lora_dir = os.getenv("QWEN_LORA_PATH", "").strip()
    has_lora = use_lora and lora_dir and Path(lora_dir).exists()

    proc_src = lora_dir if has_lora else base_model
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True, use_fast=False)

    key = _cache_key(proc_src, device, dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]["processor"], _MODEL_CACHE[key]["model"]

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    if has_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        if os.getenv("QWEN_MERGE_LORA", "0") == "1":
            model = model.merge_and_unload()

    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    _MODEL_CACHE[key] = {"processor": processor, "model": model}
    return processor, model

# -------- PASS 1: Vision -> caption --------
def _gen_caption_from_image(base_model: str, img: Image.Image,
                            max_new_tokens: int = 150, device: str = "cpu") -> str:
    import torch
    processor, model = _load_processor_model_vl(base_model, device=device)
    device = _pick_device(device)
    use_amp = (device == "cuda")

    sys_prompt = "Bạn là AI mô tả sản phẩm bằng tiếng Việt, tập trung vào các đặc điểm nổi bật."
    user_text = ("Mô tả sản phẩm trong ảnh bằng 1-2 câu, bao gồm màu sắc, chất liệu, bao bì, "
                 "kích thước (nếu có), và các đặc điểm chính. Trả lời khách quan, không phóng đại.")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": img.convert("RGB")},
            {"type": "text", "text": user_text}
        ]}
    ]

    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[chat_text], images=[img], padding=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        ctx = torch.cuda.amp.autocast(dtype=_pick_dtype(device)) if use_amp else nullcontext()
        with ctx:
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                repetition_penalty=1.05,
            )

    in_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, in_len:] if out_ids.ndim == 2 and out_ids.shape[1] > in_len else out_ids[0]
    txt = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    bad = {"sản phẩm trong hình", "sản phẩm", "hình ảnh sản phẩm"}
    if _vn_norm(txt) in set(map(_vn_norm, bad)):
        return ""
    return txt

# -------- PASS 2: Text-only -> JSON facets --------
def _gen_facets_from_text(base_model: str, text_input: str,
                          max_new_tokens: int = 256, device: str = "cpu") -> Dict[str, Any]:
    import torch
    processor, model = _load_processor_model_vl(base_model, device=device)

    # 🔹 PROMPT ĐÃ SỬA THEO YÊU CẦU
    system_prompt = (
        "Bạn là AI trích xuất thông tin sản phẩm từ ảnh. "
        "LUÔN trả lời bằng tiếng Việt và ĐÚNG format JSON như sau: "
        '{"caption": "mô tả ngắn gọn sản phẩm (tối đa 2 câu)", '
        '"keywords": ["từ khóa"], "colors": ["màu sắc"], "shapes": ["hình dạng"], '
        '"materials": ["chất liệu"], "packaging": [], "taste": [], "texture": [], '
        '"brand_guess": null, "variant_guess": null, "size_guess": null, '
        '"category_guess": "danh mục"}'
        "Nếu thiếu thông tin, hãy SUY LUẬN hợp lý nhất từ ảnh và caption. "
        "Nếu sản phẩm là thực phẩm ăn liền, luôn có taste và category_guess phù hợp."
    )

    user_prompt = (
        f"Caption: '{(text_input or '').strip()}'\n"
        "OCR: ''\n"
        "Phân tích kỹ mô tả và trả lại đúng JSON theo format ở trên."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[chat_text], padding=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            repetition_penalty=1.05,
        )

    in_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, in_len:] if out_ids.ndim == 2 and out_ids.shape[1] > in_len else out_ids[0]
    txt = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    data = _safe_json_extract(txt)

    if not data.get("keywords"): data["keywords"] = ["sản phẩm"]
    if not data.get("colors"): data["colors"] = ["không xác định"]
    if not data.get("materials"): data["materials"] = ["nhựa"]
    if not data.get("shapes"): data["shapes"] = ["không xác định"]
    if not isinstance(data.get("facet_scores"), dict):
        data["facet_scores"] = {}
    return data

# -------- Public APIs --------
def generate_caption_struct(img_or_path, max_new_tokens: int = 256,
                            base_model: Optional[str] = None,
                            device: Optional[str] = None) -> Dict[str, Any]:
    base = base_model or os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct")
    device = _pick_device(device)
    _log_env(device)

    if isinstance(img_or_path, str):
        p = Path(img_or_path)
        if not p.exists(): raise FileNotFoundError(p)
        img = Image.open(p).convert("RGB")
    else:
        img = img_or_path

    caption = _gen_caption_from_image(base, img, max_new_tokens=min(150, max_new_tokens), device=device)
    if not caption:
        caption = "Sản phẩm trong ảnh"

    facets = _gen_facets_from_text(base, caption, max_new_tokens=max_new_tokens, device=device)

    out = {
        "caption": _ensure_str(facets.get("caption")) or caption,
        "keywords": _dedup_list_str(facets.get("keywords")),
        "colors": _dedup_list_str(facets.get("colors")),
        "shapes": _dedup_list_str(facets.get("shapes")),
        "materials": _dedup_list_str(facets.get("materials")),
        "packaging": _dedup_list_str(facets.get("packaging")),
        "taste": _dedup_list_str(facets.get("taste")),
        "texture": _dedup_list_str(facets.get("texture")),
        "brand_guess": _ensure_str(facets.get("brand_guess")),
        "variant_guess": _ensure_str(facets.get("variant_guess")),
        "size_guess": _ensure_str(facets.get("size_guess")) or _regex_extract_size(caption),
        "category_guess": _ensure_str(facets.get("category_guess")),
        "facet_scores": facets.get("facet_scores", {}),
    }

    if _has_torch() and device == "cuda":
        import torch
        torch.cuda.empty_cache()
    return out

# ============== CLI demo ==============
if __name__ == "__main__":
    img_path = os.getenv("TEST_IMAGE", r"./sample.jpg")
    p = Path(img_path)
    if not p.exists():
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

    dev = _pick_device(os.getenv("DEVICE"))
    _log_env(dev)

    t0 = time.time()
    data = generate_caption_struct(str(p), max_new_tokens=int(os.getenv("CAPTION_MAX_TOKENS", "256")), device=dev)
    dt = time.time() - t0
    print(f"\n=== {p.name} | device={dev} | {dt:.1f}s ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))
