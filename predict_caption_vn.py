"""
predict_caption_vn.py

Load the Qwen2-VL checkpoint (HuggingFace transformers with optional LoRA)
and generate Vietnamese captions for given images and prompts.

Usage: set environment variables QWEN_VL_BASE (hf repo id) and optionally QWEN_LORA_PATH,
then run this script. It will use transformers backend with LoRA if QWEN_USE_LORA=1.
"""
import os
from dotenv import load_dotenv
import sys
import time
from pathlib import Path
from typing import Optional, List
from PIL import Image

# Load .env file
load_dotenv()

# Debug: Print environment variables to check
print("🔍 Debug: QWEN_USE_LORA =", os.environ.get("QWEN_USE_LORA"))
print("🔍 Debug: QWEN_LORA_PATH =", os.environ.get("QWEN_LORA_PATH"))
print("🔍 Debug: QWEN_VL_BASE =", os.environ.get("QWEN_VL_BASE"))
print("🔍 Debug: Current working directory =", os.getcwd())

def generate_with_transformers(base_model: str, img: Image.Image, prompt: str, max_new_tokens: int = 64, device: Optional[str] = None) -> str:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import PeftModel

    device = device or "cpu"  # Force CPU as no CUDA is available
    print(f"🔍 Loading processor for base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    print(f"🔍 Loading base model: {base_model} on {device} with dtype=torch.float32")
    model = Qwen2VLForConditionalGeneration.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.float32).to(device)

    # Load LoRA checkpoint if enabled
    if os.environ.get("QWEN_USE_LORA", "0") == "1":
        lora_path = os.environ.get("QWEN_LORA_PATH")
        print(f"🔍 Checking LoRA path: {lora_path}")
        if not lora_path or not Path(lora_path).exists():
            raise RuntimeError(f"LoRA path {lora_path} does not exist or is not set")
        # Use the latest checkpoint (e.g., checkpoint-18-lora)
        checkpoint_path = Path(lora_path) / "checkpoint-18-lora"
        if checkpoint_path.exists():
            lora_path = str(checkpoint_path)
            print(f"🔍 Using specific LoRA checkpoint: {lora_path}")
        print(f"🔍 Loading LoRA checkpoint from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"✅ LoRA checkpoint loaded successfully. Model is now a PeftModel.")
        model.eval()

    # Build input for Qwen2-VL using proper format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Use type instead of key
                {"type": "text", "text": prompt or "Mô tả ảnh này bằng tiếng Việt"}
            ]
        }
    ]
    
    # Apply chat template
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(f"🔍 Generated text prompt type: {type(text_prompt)}")
    print(f"🔍 Generated text prompt: {text_prompt}")

    # Prepare inputs - use lists for batch processing
    inputs = processor(
        text=[text_prompt],  # Make it a list
        images=[img],        # Make it a list
        padding=True,
        return_tensors="pt"
    ).to(device)
    print(f"🔍 Input keys: {list(inputs.keys())}")

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode output - handle batch output
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    generated_ids = outputs[0, input_len:] if input_len and outputs.ndim == 2 and outputs.shape[1] > input_len else outputs[0]
    caption = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(f"🔍 Raw decoded output: {caption}")
    return caption

def predict_for_images(image_paths: List[str], prompts: List[str], max_new_tokens: int = 64):
    base = os.environ.get("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct")
    results = []

    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            print(f"❌ Image not found: {img_path}")
            continue
        img = Image.open(p).convert("RGB")
        for prompt in prompts:
            start = time.time()
            try:
                caption = generate_with_transformers(base, img, prompt, max_new_tokens=max_new_tokens)
                used = f"transformers ({base})"
                elapsed = time.time() - start
                print(f"✅ [{used}] {p.name} | Prompt: '{prompt or '(<empty> Vietnamese)'}' -> {elapsed:.1f}s")
                print(f"   Caption (vi): {caption}\n")
                results.append({"image": str(p), "prompt": prompt, "caption": caption, "backend": used, "time_s": elapsed})
            except Exception as e:
                print(f"❌ Failed to generate for {p.name} with prompt '{prompt}': {type(e).__name__}: {e}")
                continue
    return results

if __name__ == "__main__":
    images = [
        os.environ.get("TEST_IMAGE", r"E:\api_hango\flask_pgvector_shop\flask_pgvector_shop\uploads\73_603137893c2545948b1778d813b03b24.jpg")
    ]
    prompts = [
        "Mô tả ảnh này bằng tiếng Việt",
        "Tên sản phẩm",
        ""  # empty prompt -> fallback description
    ]
    max_tokens = int(os.environ.get("CAPTION_MAX_TOKENS", "80"))

    print("Starting caption prediction (Vietnamese)...")
    print("🔍 QWEN_USE_LORA from env:", os.environ.get("QWEN_USE_LORA"))
    predict_for_images(images, prompts, max_new_tokens=max_tokens)