# predict_caption_vn.py  (2-pass: caption -> facets JSON)
import os, sys, time, json, re, unicodedata, ast
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ============== utils ==============
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
            if x is None: 
                continue
            s = str(x).strip()
            if not s:
                continue
            key = _vn_norm(s)
            if key and key not in seen:
                seen.add(key)
                out.append(s.strip())
            if len(out) >= max_items:
                break
    elif isinstance(xs, str) and xs.strip():
        out = [xs.strip()]
    return out or ["không xác định"]  # Đảm bảo không trả về mảng rỗng

def _ensure_str(x: Any) -> Optional[str]:
    if x is None: 
        return None
    s = str(x).strip()
    return s or None

def _safe_json_extract(txt: str) -> Dict[str, Any]:
    txt = txt.strip()
    # 1) JSON trực tiếp
    try:
        return json.loads(txt)
    except Exception:
        pass
    # 2) Lấy khối {...} dài nhất
    try:
        stack, best = [], None
        for i, ch in enumerate(txt):
            if ch == '{':
                stack.append(i)
            elif ch == '}' and stack:
                j = stack.pop()
                cand = txt[j:i+1]
                if best is None or len(cand) > len(best):
                    best = cand
        if best:
            return json.loads(best)
    except Exception:
        pass
    # 3) literal_eval (chấp nhận quote đơn)
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

def _regex_extract_size(s: str) -> Optional[str]:
    pat = r'(\d+(?:[.,]\d+)?\s?(?:g|kg|ml|l|lit|lít|oz|pack|bịch|hộp))'
    m = re.search(pat, s.lower())
    return m.group(1) if m else None

# ============== Transformers backend (CPU ok) ==============
def _load_processor_model_vl(base_model: str, device: str = "cpu"):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)

    # LoRA (tuỳ chọn)
    if os.environ.get("QWEN_USE_LORA", "0") == "1":
        from peft import PeftModel
        lora_path = os.environ.get("QWEN_LORA_PATH")
        if not lora_path or not Path(lora_path).exists():
            raise RuntimeError(f"LoRA path {lora_path} does not exist")
        ck = Path(lora_path) / "checkpoint-18-lora"
        if ck.exists():
            lora_path = str(ck)
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    # an toàn sinh
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return processor, model

def _gen_caption_from_image(base_model: str, img: Image.Image,
                            max_new_tokens: int = 100, device: str = "cpu") -> str:
    """
    PASS 1: chỉ xin caption 1-2 câu, tránh JSON để giảm 'áp lực' định dạng.
    """
    import torch
    processor, model = _load_processor_model_vl(base_model, device=device)

    sys_prompt = "Bạn là AI mô tả sản phẩm bằng tiếng Việt, tập trung vào các đặc điểm nổi bật."
    user_text = "Mô tả sản phẩm trong ảnh bằng 1-2 câu, bao gồm màu sắc, chất liệu, bao bì, kích thước (nếu có), và các đặc điểm chính. Trả lời khách quan, không phóng đại."

    messages = [
        {"role":"system","content": sys_prompt},
        {"role":"user","content":[{"type":"image"},{"type":"text","text": user_text}]}
    ]
    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[chat_text], images=[img.convert("RGB")], padding=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # greedy cho caption ngắn, giảm noise
            temperature=None,
            repetition_penalty=1.05,
        )
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, in_len:] if out_ids.ndim==2 and out_ids.shape[1]>in_len else out_ids[0]
    txt = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # lọc caption quá chung chung
    bad = {"sản phẩm trong hình", "sản phẩm", "hình ảnh sản phẩm"}
    if _vn_norm(txt) in set(map(_vn_norm, bad)):
        return ""
    return txt

def _gen_facets_from_text(base_model: str, text_input: str, max_new_tokens: int = 196, device: str = "cpu") -> Dict[str, Any]:
    """
    PASS 2: text-only → JSON facets. Thêm few-shot đa dạng để tăng tính tổng quát.
    Dùng chính Qwen2-VL ở chế độ text-only (rẻ hơn vision).
    """
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    fewshot = [
        {
            "role": "user",
            "content": [{"type":"text","text":
                "Caption: 'Snack nui chiên giòn vị cay phô mai trong túi nhựa trong suốt màu vàng cam.'\n"
                "OCR: ''\nHãy trích JSON facet như schema quy định."
            }]
        },
        {
            "role": "assistant",
            "content": [{"type":"text","text":
                '{"caption":"Snack nui chiên giòn vị cay phô mai trong túi nhựa trong suốt màu vàng cam.",'
                '"keywords":["snack","nui chiên","phô mai","cay","giòn"],'
                '"colors":["vàng","cam"],"shapes":["hạt"],"materials":["nhựa"],"packaging":["túi trong suốt"],'
                '"taste":["cay","phô mai"],"texture":["giòn"],'
                '"brand_guess":null,"variant_guess":"cay phô mai","size_guess":null,"category_guess":"snack",'
                '"facet_scores":{"colors":[["vàng",0.9],["cam",0.7]],"materials":[["nhựa",0.8]]}}'
            }]
        },
        {
            "role": "user",
            "content": [{"type":"text","text":
                "Caption: 'Nước ngọt Coca-Cola chai nhựa 500ml màu bạc.'\n"
                "OCR: 'Coca-Cola 500ml'\nHãy trích JSON facet như schema quy định."
            }]
        },
        {
            "role": "assistant",
            "content": [{"type":"text","text":
                '{"caption":"Nước ngọt Coca-Cola chai nhựa 500ml màu bạc.",'
                '"keywords":["nước ngọt","Coca-Cola","chai nhựa"],'
                '"colors":["bạc"],"shapes":["chai"],"materials":["nhựa"],"packaging":["chai"],'
                '"taste":["ngọt"],"texture":[],'
                '"brand_guess":"Coca-Cola","variant_guess":null,"size_guess":"500ml","category_guess":"nước giải khát",'
                '"facet_scores":{"colors":[["bạc",0.95]],"materials":[["nhựa",0.9]]}}'
            }]
        },
        {
            "role": "user",
            "content": [{"type":"text","text":
                "Caption: 'Áo thun nam màu đen, chất liệu cotton, in logo nhỏ màu trắng.'\n"
                "OCR: ''\nHãy trích JSON facet như schema quy định."
            }]
        },
        {
            "role": "assistant",
            "content": [{"type":"text","text":
                '{"caption":"Áo thun nam màu đen, chất liệu cotton, in logo nhỏ màu trắng.",'
                '"keywords":["áo thun","áo nam","cotton","logo"],'
                '"colors":["đen","trắng"],"shapes":["áo"],"materials":["cotton"],"packaging":["túi nhựa"],'
                '"taste":[],"texture":["mềm"],'
                '"brand_guess":null,"variant_guess":null,"size_guess":null,"category_guess":"quần áo",'
                '"facet_scores":{"colors":[["đen",0.9],["trắng",0.7]],"materials":[["cotton",0.95]]}}'
            }]
        },
        {
            "role": "user",
            "content": [{"type":"text","text":
                "Caption: 'Máy xay sinh tố màu trắng, dung tích 1.5 lít, chất liệu nhựa cao cấp.'\n"
                "OCR: '1.5L'\nHãy trích JSON facet như schema quy định."
            }]
        },
        {
            "role": "assistant",
            "content": [{"type":"text","text":
                '{"caption":"Máy xay sinh tố màu trắng, dung tích 1.5 lít, chất liệu nhựa cao cấp.",'
                '"keywords":["máy xay","sinh tố","nhựa cao cấp"],'
                '"colors":["trắng"],"shapes":["máy xay"],"materials":["nhựa"],"packaging":["hộp"],'
                '"taste":[],"texture":[],'
                '"brand_guess":null,"variant_guess":null,"size_guess":"1.5 lít","category_guess":"đồ gia dụng",'
                '"facet_scores":{"colors":[["trắng",0.9]],"materials":[["nhựa",0.85]]}}'
            }]
        },
        {
            "role": "user",
            "content": [{"type":"text","text":
                "Caption: 'Son môi màu đỏ đậm, vỏ kim loại ánh bạc, dạng thỏi.'\n"
                "OCR: ''\nHãy trích JSON facet như schema quy định."
            }]
        },
        {
            "role": "assistant",
            "content": [{"type":"text","text":
                '{"caption":"Son môi màu đỏ đậm, vỏ kim loại ánh bạc, dạng thỏi.",'
                '"keywords":["son môi","đỏ đậm","thỏi"],'
                '"colors":["đỏ","bạc"],"shapes":["thỏi"],"materials":["kim loại"],"packaging":["thỏi"],'
                '"taste":[],"texture":["mịn"],'
                '"brand_guess":null,"variant_guess":"đỏ đậm","size_guess":null,"category_guess":"mỹ phẩm",'
                '"facet_scores":{"colors":[["đỏ",0.9],["bạc",0.8]],"materials":[["kim loại",0.9]]}}'
            }]
        }
    ]

    system_prompt = (
        "Bạn là AI trích xuất facet sản phẩm từ văn bản tiếng Việt. "
        "Phân tích caption và OCR để tạo JSON hợp lệ theo schema. "
        "Suy luận các đặc điểm như màu sắc, chất liệu, bao bì, hình dạng, v.v., từ văn bản. "
        "Các trường keywords, colors, materials, shapes BẮT BUỘC phải có ít nhất một giá trị, "
        "sử dụng suy luận hợp lý hoặc giá trị mặc định nếu không có thông tin rõ ràng."
    )
    user_prompt = (
        "Schema JSON bắt buộc:\n"
        '{ "caption": "...", "keywords": [], "colors": [], "shapes": [], "materials": [], '
        '"packaging": [], "taste": [], "texture": [], '
        '"brand_guess": null, "variant_guess": null, "size_guess": null, "category_guess": null, '
        '"facet_scores": {} }\n'
        "Nguồn:\n"
        f"Caption: '{text_input.strip()}'\n"
        "OCR: ''\n"
        "Yêu cầu: Trích xuất facet chính xác từ caption và OCR. "
        "Điền các trường keywords, colors, materials, shapes BẮT BUỘC có ít nhất một giá trị, "
        "dựa trên caption hoặc suy luận hợp lý (ví dụ: nếu không rõ màu thì dùng 'không xác định', "
        "nếu không rõ chất liệu thì dùng 'nhựa' hoặc 'vải' tùy danh mục). "
        "Các trường khác để rỗng hoặc null nếu không có dữ liệu. Trả đúng JSON."
    )

    messages = [{"role":"system","content": system_prompt}] + fewshot + [
        {"role":"user","content":[{"type":"text","text": user_prompt}]}
    ]

    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[chat_text], images=None, padding=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            repetition_penalty=1.05,
        )
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, in_len:] if out_ids.ndim==2 and out_ids.shape[1]>in_len else out_ids[0]
    txt = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    data = _safe_json_extract(txt)

    # Đảm bảo các trường bắt buộc không rỗng
    if not data.get("keywords"):
        data["keywords"] = ["sản phẩm"]  # Mặc định nếu không có keywords
    if not data.get("colors"):
        data["colors"] = ["không xác định"]  # Mặc định nếu không có màu
    if not data.get("materials"):
        data["materials"] = ["nhựa"]  # Mặc định nếu không có chất liệu
    if not data.get("shapes"):
        data["shapes"] = ["không xác định"]  # Mặc định nếu không có hình dạng
    return data

def generate_caption_struct(
    img_or_path,
    max_new_tokens: int = 128,
    base_model: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    2-pass:
      1) Vision -> caption (1-2 câu)
      2) Text-only -> JSON facets từ caption
    """
    base = base_model or os.environ.get("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct")
    device = device or "cpu"

    if isinstance(img_or_path, str):
        p = Path(img_or_path)
        if not p.exists():
            raise FileNotFoundError(p)
        img = Image.open(p).convert("RGB")
    else:
        img = img_or_path

    # PASS 1
    caption = _gen_caption_from_image(base, img, max_new_tokens=min(100, max_new_tokens), device=device)
    if not caption:
        caption = "Sản phẩm trong ảnh"

    # PASS 2
    facets = _gen_facets_from_text(base, caption, max_new_tokens=max_new_tokens, device=device)

    # Hợp nhất + chuẩn hoá
    out = {
        "caption": _ensure_str(facets.get("caption")) or caption,
        "keywords": _dedup_list_str(facets.get("keywords")) or ["sản phẩm"],
        "colors": _dedup_list_str(facets.get("colors")) or ["không xác định"],
        "shapes": _dedup_list_str(facets.get("shapes")) or ["không xác định"],
        "materials": _dedup_list_str(facets.get("materials")) or ["nhựa"],
        "packaging": _dedup_list_str(facets.get("packaging")),
        "taste": _dedup_list_str(facets.get("taste")),
        "texture": _dedup_list_str(facets.get("texture")),
        "brand_guess": _ensure_str(facets.get("brand_guess")),
        "variant_guess": _ensure_str(facets.get("variant_guess")),
        "size_guess": _ensure_str(facets.get("size_guess")) or _regex_extract_size(caption) or None,
        "category_guess": _ensure_str(facets.get("category_guess")),
        "facet_scores": facets.get("facet_scores") if isinstance(facets.get("facet_scores"), dict) else {},
    }
    return out

# Backward-compatible: caption-only
def generate_caption(img_or_path, prompt: str, max_new_tokens: int = 80, backend: Optional[str] = None, hf_base: Optional[str] = None):
    data = generate_caption_struct(img_or_path, max_new_tokens=max_new_tokens, base_model=hf_base)
    return data.get("caption") or "Sản phẩm trong hình ảnh"

# ============== CLI demo ==============
if __name__ == "__main__":
    img_path = os.environ.get("TEST_IMAGE", r"E:\api_hango\flask_pgvector_shop\flask_pgvector_shop\uploads\72_d1b5d89dff0b4c6096bc19ef01eb0ec0.jpg")
    p = Path(img_path)
    if not p.exists():
        print(f"❌ Image not found: {img_path}")
        sys.exit(1)

    t0 = time.time()
    data = generate_caption_struct(str(p), max_new_tokens=int(os.environ.get("CAPTION_MAX_TOKENS","160")))
    dt = time.time() - t0
    print(f"\n=== {p.name} | {dt:.1f}s ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))