"""
Agent tự động sinh caption bằng Gemini và lưu vào DB qua API
"""
import os
import sys
import time
import json
import logging
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Import hàm generate từ gemini_simple_caption
from gemini_simple_caption import generate_full_caption_data

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gemini_autogen_agent.log", encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)

# API config
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))


def get_pending_images(api_url: str, sku_id: Optional[int] = None, limit: int = 100, model_name: Optional[str] = None) -> List[Dict]:
    """
    Lấy danh sách ảnh chưa có caption từ DB qua API
    Trả về: [{"sku_id": int, "image_id": int, "image_path": str, "ocr_text": str}, ...]
    """
    try:
        endpoint = f"{api_url}/api/captions/pending"
        params = {"limit": limit}
        if sku_id:
            params["sku_id"] = sku_id
        if model_name:
            params["model_name"] = model_name
        
        response = requests.get(endpoint, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("ok"):
            logger.error("API error: %s", data.get("error"))
            return []
        
        return data.get("images", [])
    except Exception as e:
        logger.exception("Failed to get pending images: %s", e)
        return []


def save_caption_via_api(api_url: str, caption_data: Dict[str, Any]) -> bool:
    """
    Lưu caption vào DB qua API /captions/suggest
    Returns: True nếu thành công
    """
    try:
        endpoint = f"{api_url}/captions/suggest"
        
        # Chuẩn bị payload theo format API
        payload = {
            "sku_id": caption_data["sku_id"],
            "image_path": caption_data["image_path"],
            "style": caption_data["style"],
            "caption_text": caption_data["caption_text"],
            "model_name": caption_data["model_name"],
            "prompt_version": caption_data.get("prompt_version", "v1.0"),
            # Thêm metadata nếu API hỗ trợ
            "keywords": caption_data.get("keywords", []),
            "colors": caption_data.get("colors", []),
            "shapes": caption_data.get("shapes", []),
            "materials": caption_data.get("materials", []),
            "packaging": caption_data.get("packaging", []),
            "taste": caption_data.get("taste", []),
            "texture": caption_data.get("texture", []),
            "brand_guess": caption_data.get("brand_guess"),
            "variant_guess": caption_data.get("variant_guess"),
            "size_guess": caption_data.get("size_guess"),
            "category_guess": caption_data.get("category_guess"),
            "facet_scores": caption_data.get("facet_scores", [])
        }
        
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get("ok"):
            logger.info("✅ Saved caption for SKU %d, image %s (caption_id=%s)",
                       caption_data["sku_id"],
                       caption_data["image_path"],
                       result.get("caption_id"))
            return True
        else:
            logger.error("API returned error: %s", result.get("error"))
            return False
            
    except Exception as e:
        logger.exception("Failed to save caption via API: %s", e)
        return False


def process_one_image(
    img_info: Dict,
    upload_dir: str,
    api_url: str,
    dry_run: bool = False
) -> bool:
    """
    Xử lý 1 ảnh: generate caption + save qua API
    Returns: True nếu thành công
    """
    sku_id = img_info["sku_id"]
    image_path = img_info["image_path"]
    ocr_text = img_info.get("ocr_text")
    
    # Build full path
    full_path = image_path if os.path.isabs(image_path) else os.path.join(upload_dir, image_path)
    
    if not Path(full_path).exists():
        logger.warning("⚠️  Image not found: %s", full_path)
        return False
    
    logger.info("🖼️  Processing SKU %d: %s", sku_id, Path(full_path).name)
    
    try:
        # Generate caption metadata
        t0 = time.time()
        caption_data = generate_full_caption_data(
            image_path=full_path,
            ocr_text=ocr_text,
            sku_id=sku_id
        )
        elapsed = time.time() - t0
        
        logger.info("   Generated in %.2fs: %s", elapsed, caption_data["caption_text"][:80])
        
        if dry_run:
            logger.info("   [DRY RUN] Would save: %s", json.dumps(caption_data, ensure_ascii=False, indent=2))
            return True
        
        # Save via API
        success = save_caption_via_api(api_url, caption_data)
        return success
        
    except Exception as e:
        logger.exception("❌ Error processing image %s: %s", image_path, e)
        return False


def run_agent(
    sku_id: Optional[int] = None,
    limit: int = 100,
    batch_size: int = 10,
    delay: float = 1.0,
    dry_run: bool = False,
    model_name: Optional[str] = None
):
    """
    Agent chính: lấy pending images, generate captions, save qua API
    
    Args:
        sku_id: Chỉ xử lý SKU này (None = all)
        limit: Số ảnh tối đa mỗi lần chạy
        batch_size: Xử lý bao nhiêu ảnh rồi nghỉ 1 chút
        delay: Delay giữa các batch (seconds)
        dry_run: True = chỉ generate, không lưu DB
        model_name: Chỉ xử lý ảnh chưa có caption với model này
    """
    api_url = API_BASE_URL
    upload_dir = os.getenv("UPLOAD_DIR", "uploads")
    
    logger.info("=" * 80)
    logger.info("Gemini Caption Autogen Agent Started")
    logger.info("API: %s", api_url)
    logger.info("Upload dir: %s", upload_dir)
    logger.info("SKU filter: %s", sku_id or "ALL")
    logger.info("Model filter: %s", model_name or "ANY")
    logger.info("Limit: %d, Batch size: %d, Delay: %.1fs", limit, batch_size, delay)
    logger.info("Dry run: %s", dry_run)
    logger.info("=" * 80)
    
    # Get pending images
    logger.info("Fetching pending images...")
    pending = get_pending_images(api_url, sku_id=sku_id, limit=limit, model_name=model_name)
    
    if not pending:
        logger.info("No pending images found.")
        return
    
    logger.info("Found %d pending images", len(pending))
    
    # Process in batches
    total = len(pending)
    success = 0
    failed = 0
    
    for idx, img_info in enumerate(pending, start=1):
        logger.info("\n[%d/%d] Processing...", idx, total)
        
        if process_one_image(img_info, upload_dir, api_url, dry_run=dry_run):
            success += 1
        else:
            failed += 1
        
        # Batch delay
        if idx % batch_size == 0 and idx < total:
            logger.info("--- Batch complete, sleeping %.1fs ---", delay)
            time.sleep(delay)
    
    # Summary
    logger.info("=" * 80)
    logger.info("Agent Finished")
    logger.info("Total: %d | Success: %d | Failed: %d", total, success, failed)
    logger.info("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemini Caption Autogen Agent")
    parser.add_argument("--sku-id", type=int, help="Process only this SKU ID")
    parser.add_argument("--limit", type=int, default=100, help="Max images to process")
    parser.add_argument("--batch-size", type=int, default=10, help="Images per batch before delay")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between batches (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Generate only, don't save to DB")
    parser.add_argument("--model-name", type=str, default="gemini-2.0-flash-exp", help="Only process images without captions from this model")
    args = parser.parse_args()
    
    try:
        run_agent(
            sku_id=args.sku_id,
            limit=args.limit,
            batch_size=args.batch_size,
            delay=args.delay,
            dry_run=args.dry_run,
            model_name=args.model_name
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Agent interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error in agent: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()