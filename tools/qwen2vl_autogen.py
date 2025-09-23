import os
import base64
import io
import time
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# OPTIONAL: DB helpers from your project
# from services.db_utils import q, exec_sql

# CONFIG - chỉnh lại
BASE_MODEL = os.getenv("QWEN_VL_BASE", "path/to/qwen-vl-base")   # <-- set to your Qwen-VL base model
ADAPTER_DIR = os.getenv("QWEN_VL_ADAPTER", r"e:\api_hango\flask_pgvector_shop\flask_pgvector_shop\out-qwen2vl-lora")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight & good

# load tokenizer and base model, then load LoRA adapter
print("Loading tokenizer and base model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR) if os.path.isdir(ADAPTER_DIR) and os.path.exists(os.path.join(ADAPTER_DIR,"tokenizer.json")) else AutoTokenizer.from_pretrained(BASE_MODEL)

# load base model then wrap with PEFT adapter
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    device_map="auto" if DEVICE=="cuda" else None,
)
print("Attaching LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map="auto" if DEVICE=="cuda" else None)
model.eval()

def image_to_data_uri(pil_img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def prepare_prompt_for_caption(data_uri: str, prompt: str) -> str:
    # Generic prompt pattern: some Qwen-VL models accept <img>...</img> in message.
    # Adjust depending on your base model's expected image markup.
    return f"<img>{data_uri}</img>\n{prompt}\nTrả lời duy nhất caption ngắn (1-2 câu) bằng tiếng Việt."

def generate_caption_from_image(pil_img: Image.Image, prompt: str, max_new_tokens=128, temperature=0.2) -> str:
    data_uri = image_to_data_uri(pil_img, fmt="PNG")
    full_prompt = prepare_prompt_for_caption(data_uri, prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # model may echo prompt; try to strip prompt prefix
    if full_prompt in text:
        text = text.split(full_prompt, 1)[1].strip()
    return text.strip()

def split_segments(text: str) -> List[str]:
    import re
    segs = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return segs if segs else [text.strip()]

def embed_text(text: str) -> List[float]:
    vec = EMBED_MODEL.encode(text, normalize_embeddings=True)
    return vec.tolist()

# Example runner for a list of image paths: generate captions, segments, embeddings, optionally save to DB
def process_images(paths: List[str], prompt_search: str = "Viết caption ngắn 1-2 câu, khách quan, tiếng Việt."):
    for i, p in enumerate(paths, start=1):
        try:
            print(f"[{i}/{len(paths)}] Loading {p}", flush=True)
            pil = Image.open(p).convert("RGB")
            caption = generate_caption_from_image(pil, prompt_search)
            print(f"Caption raw: {caption}", flush=True)
            segs = split_segments(caption)
            for j, s in enumerate(segs, start=1):
                print(f"  segment {j}/{len(segs)}: {s}", flush=True)
                vec = embed_text(s)
                print(f"    embedding len={len(vec)}", flush=True)
                # save to DB example (pseudo):
                # exec_sql("INSERT INTO sku_captions(...) VALUES (...) ...", (...))
                # exec_sql("INSERT INTO text_vectors(sku_id, text, vec) VALUES (%s,%s,%s)", (sku_id, s, vec))
            time.sleep(0.5)
        except Exception as ex:
            print("Error:", ex, flush=True)

if __name__ == "__main__":
    # quick test: supply paths list or directory
    import sys, glob
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = glob.glob(os.path.join(os.getcwd(), "uploads", "*.png"))[:50]
    process_images(paths)