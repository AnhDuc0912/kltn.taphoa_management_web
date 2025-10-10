"""Simple OpenAI Vision caption generator"""
import os
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import openai
except ImportError:
    openai = None

def generate_simple_caption_openai(image_path: str, prompt: str = "Mô tả ảnh này bằng tiếng Việt, ngắn gọn") -> str:
    if openai is None:
        raise RuntimeError("openai not installed. Run: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Read and encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4-vision-preview
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                    ]
                }
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return Path(image_path).stem.replace("_", " ")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--prompt", "-p", default="Mô tả ảnh này bằng tiếng Việt, ngắn gọn")
    args = parser.parse_args()
    
    t0 = time.time()
    caption = generate_simple_caption_openai(args.image, args.prompt)
    print(f"✅ Caption ({time.time()-t0:.2f}s):\n   {caption}")

if __name__ == "__main__":
    main()