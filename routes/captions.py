# routes/captions.py
import os, time, json, traceback
from pathlib import Path
from flask import Blueprint, request, redirect, url_for, flash, current_app, jsonify
from services.db_utils import q, exec_sql
from services.moondream_service import md_chat_vision, _mime_from_path

bp = Blueprint("captions_bp", __name__)

# ==============================
# Global config / constants
# ==============================
UPLOAD_DIR = os.path.abspath(os.getenv("UPLOAD_FOLDER", "uploads"))
PROMPT_SEARCH = "Viết caption ngắn 1-2 câu, khách quan, tiếng Việt."
PROMPT_SEO = "Viết mô tả 3-5 câu bằng tiếng Việt cho trang sản phẩm, nhấn mạnh công dụng/đặc điểm."

# ==============================
# CUDA / PyTorch helpers
# ==============================
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    torch = None  # type: ignore

def _torch_device() -> str:
    """
    Chọn device ưu tiên 'cuda' nếu có; cho phép ép/tắt bằng ENV:
      - DISABLE_CUDA=1  -> luôn dùng CPU
      - FORCE_CUDA=1    -> ưu tiên CUDA nếu is_available()
    """
    if not _HAS_TORCH:
        return "cpu"
    if os.getenv("DISABLE_CUDA", "").strip() == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _log_cuda_env(logger):
    """Log trạng thái CUDA/PyTorch để debug nhanh."""
    try:
        if not _HAS_TORCH:
            logger.info("CUDA env: PyTorch not installed -> CPU")
            return
        dev = _torch_device()
        info = {"device": dev, "torch": torch.__version__}
        if dev == "cuda":
            info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            })
        logger.info("CUDA env: %s", info)
    except Exception as e:
        logger.warning("Failed to log CUDA env: %s", e)

@bp.get("/system/cuda_info")
def cuda_info():
    """Health-check: trả info CUDA/PyTorch + thử tạo tensor trên GPU."""
    try:
        dev = _torch_device()
        out = {"device": dev, "pytorch": (torch.__version__ if _HAS_TORCH else None)}
        if _HAS_TORCH and dev == "cuda":
            out.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            })
            x = torch.randn(128, 128, device="cuda")
            out["tensor_cuda_ok"] = x.shape == (128, 128)
        return {"ok": True, **out}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

# ==============================
# Small math helpers
# ==============================
def _vec_literal(vec):
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]" if vec else None

def _l2norm(v):
    import math
    n = math.sqrt(sum(float(x) * float(x) for x in v)) or 1.0
    return [float(x) / n for x in v]

# ==============================
# Text embedding (GPU nếu có)
# ==============================
def embed_text(text: str):
    """
    Trả embedding vector cho text.
    Ưu tiên utils.encode_texts(..., device=...), fallback SentenceTransformer (GPU), cuối cùng random.
    """
    try:
        from utils import encode_texts
        try:
            emb = encode_texts([text], device=_torch_device())  # API mới có device
        except TypeError:
            emb = encode_texts([text])  # API cũ
        v = emb[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)
    except Exception as e:
        current_app.logger.warning("embed_text fallback (utils failed): %s", e)
        try:
            from sentence_transformers import SentenceTransformer
            dev = _torch_device()
            model = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
            v = model.encode(text, normalize_embeddings=True)
            return v.tolist() if hasattr(v, "tolist") else list(v)
        except Exception as ee:
            current_app.logger.warning("embed_text second fallback failed: %s", ee)
            import numpy as np
            return np.random.rand(512).tolist()

# ==============================
# AUTOGEN CAPTIONS (batch cho 1 SKU)
# ==============================
@bp.post("/admin/captions/autogen/<int:sku_id>", endpoint="captions_autogen")
def captions_autogen(sku_id):
    limit = int(request.form.get("limit", request.args.get("limit", 200)))
    offset = int(request.form.get("offset", request.args.get("offset", 0)))

    # Lazy import để tránh load model khi không dùng
    try:
        from tools.qwen2vl_autogen import generate_caption_struct
        from PIL import Image  # noqa: F401 (đề phòng module cần)
        from io import BytesIO  # noqa: F401
    except Exception as e:
        current_app.logger.exception("Failed to import qwen2vl tool")
        flash("Không thể nạp module qwen autogen: " + str(e), "danger")
        return redirect(url_for("skus.skus"))

    # Chỉ lấy ảnh của sku_id (không ràng buộc "chưa có caption" để cho phép refresh)
    rows = q("""
        SELECT si.id, si.sku_id, si.image_path, si.ocr_text
        FROM sku_images si
        WHERE si.sku_id = %s
        ORDER BY si.is_primary DESC, si.id
        LIMIT %s OFFSET %s
    """, (sku_id, limit, offset))

    if not rows:
        current_app.logger.info(f"No images found for sku_id {sku_id}")
        flash(f"Không tìm thấy ảnh nào cho SKU {sku_id}.", "warning")
        return redirect(url_for("skus.skus"))

    done = fail = 0
    model_name = os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct").split("/")[-1]
    upload_dir = current_app.config.get("UPLOAD_DIR", UPLOAD_DIR)

    # Log CUDA 1 lần
    _log_cuda_env(current_app.logger)
    _qwen_device = _torch_device()
    # Optionally, cho tool biết qua ENV
    os.environ.setdefault("QWEN_DEVICE", _qwen_device)

    for (img_id, curr_sku_id, img_path, ocr_text) in rows:
        try:
            # Xây dựng đường dẫn ảnh
            fpath = img_path if os.path.isabs(img_path) else os.path.join(upload_dir, img_path)
            p = Path(fpath)
            current_app.logger.info(f"Processing image: {p}")

            if not p.exists():
                current_app.logger.warning(f"Image not found, skipping: {fpath}")
                fail += 1
                continue

            # Sinh caption + facets
            try:
                struct_data = generate_caption_struct(str(p), max_new_tokens=256, device=_qwen_device)
                print(struct_data)
            except TypeError:
                struct_data = generate_caption_struct(str(p), max_new_tokens=200)

            # Lấy các field
            caption = struct_data.get("caption", "Sản phẩm trong ảnh")
            keywords = struct_data.get("keywords", [])
            colors = struct_data.get("colors", [])
            shapes = struct_data.get("shapes", [])
            materials = struct_data.get("materials", [])
            packaging = struct_data.get("packaging", [])
            taste = struct_data.get("taste", [])
            texture = struct_data.get("texture", [])
            brand_guess = struct_data.get("brand_guess")
            variant_guess = struct_data.get("variant_guess")
            size_guess = struct_data.get("size_guess")
            category_guess = struct_data.get("category_guess")
            facet_scores = json.dumps(struct_data.get("facet_scores", {}), ensure_ascii=False)

            # Tạo embedding cho caption + keywords
            keywords_str = " ".join(keywords) if keywords else ""
            combined_text = f"{caption} {keywords_str}".strip()
            caption_vec = embed_text(combined_text) if combined_text else []
            caption_vec_lit = _vec_literal(_l2norm(caption_vec)) if caption_vec else None

            # Đọc bytes ảnh cho md_chat_vision
            try:
                with open(fpath, "rb") as fh:
                    img_bytes = fh.read()
            except Exception as e:
                current_app.logger.warning(f"Failed to read image bytes {fpath}: {str(e)}")
                fail += 1
                continue

            # SEO caption (Moondream)
            try:
                seo_caption = md_chat_vision(
                    img_bytes,
                    _mime_from_path(fpath),
                    PROMPT_SEO + (f"\nOCR: {ocr_text}" if ocr_text else "")
                )
            except Exception as e:
                current_app.logger.warning(f"md_chat_vision failed for {fpath}: {str(e)}")
                seo_caption = caption  # fallback

            # Lưu caption 'search'
            exec_sql("""
                INSERT INTO sku_captions(
                    sku_id, image_path, lang, style, caption_text, model_name, prompt_version, needs_review,
                    keywords, colors, shapes, materials, packaging, taste, texture,
                    brand_guess, variant_guess, size_guess, category_guess, facet_scores, caption_vec
                )
                VALUES (%s, %s, 'vi', 'search', %s, %s, 'v1.0', TRUE,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version)
                DO UPDATE SET 
                    caption_text = EXCLUDED.caption_text,
                    needs_review = TRUE,
                    keywords = EXCLUDED.keywords,
                    colors = EXCLUDED.colors,
                    shapes = EXCLUDED.shapes,
                    materials = EXCLUDED.materials,
                    packaging = EXCLUDED.packaging,
                    taste = EXCLUDED.taste,
                    texture = EXCLUDED.texture,
                    brand_guess = EXCLUDED.brand_guess,
                    variant_guess = EXCLUDED.variant_guess,
                    size_guess = EXCLUDED.size_guess,
                    category_guess = EXCLUDED.category_guess,
                    facet_scores = EXCLUDED.facet_scores,
                    caption_vec = EXCLUDED.caption_vec
            """, (
                curr_sku_id, img_path, caption, model_name,
                keywords, colors, shapes, materials, packaging, taste, texture,
                brand_guess, variant_guess, size_guess, category_guess, facet_scores, caption_vec_lit
            ))

            # Lưu caption 'seo'
            exec_sql("""
                INSERT INTO sku_captions(sku_id, image_path, lang, style, caption_text, model_name, prompt_version, needs_review)
                VALUES (%s, %s, 'vi', 'seo', %s, %s, 'v1.0', TRUE)
                ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version)
                DO UPDATE SET 
                    caption_text = EXCLUDED.caption_text,
                    needs_review = TRUE
            """, (curr_sku_id, img_path, seo_caption, model_name))

            done += 1
        except Exception as e:
            current_app.logger.exception(f"Error processing image {img_path}: {str(e)}")
            fail += 1
        finally:
            # Giải phóng VRAM dần khi chạy CUDA
            try:
                if _HAS_TORCH and _qwen_device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # refresh materialized view (best-effort)
    try:
        exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    except Exception as e:
        current_app.logger.warning(f"Failed to refresh search corpus: {str(e)}")

    flash(f"Autogen xong cho SKU {sku_id}: thành công {done}, lỗi {fail}",
          "success" if fail == 0 else "warning")
    return redirect(url_for("skus.skus"))

# ==============================
# SUGGEST CAPTION (write via API)
# ==============================
@bp.post("/captions/suggest", endpoint="captions_suggest")
def captions_suggest():
    """API endpoint to save/update caption suggestions với đầy đủ metadata."""
    try:
        d = request.get_json(force=True)
        current_app.logger.info(
            "captions_suggest payload: %s",
            json.dumps(d, ensure_ascii=False)[:1000]
        )

        # Validate required fields
        required_fields = ["sku_id", "image_path", "style", "caption_text", "model_name"]
        for field in required_fields:
            if not d.get(field):
                return jsonify({"ok": False, "error": f"Missing required field: {field}"}), 400

        caption_text = (d["caption_text"] or "").strip()
        if not caption_text:
            return jsonify({"ok": False, "error": "Caption text cannot be empty"}), 400

        # Arrays
        keywords  = d.get("keywords", [])
        colors    = d.get("colors", [])
        shapes    = d.get("shapes", [])
        materials = d.get("materials", [])
        packaging = d.get("packaging", [])
        taste     = d.get("taste", [])
        texture   = d.get("texture", [])

        # Guess fields
        brand_guess   = d.get("brand_guess")
        variant_guess = d.get("variant_guess")
        size_guess    = d.get("size_guess")
        category_guess= d.get("category_guess")

        # facet_scores: jsonb
        facet_scores = d.get("facet_scores", {})
        facet_scores_json = json.dumps(facet_scores, ensure_ascii=False) if facet_scores else None

        # Embedding (GPU nếu có)
        caption_vec_lit = None
        try:
            v = embed_text(caption_text)
            v = _l2norm(v)
            caption_vec_lit = _vec_literal(v)
            current_app.logger.info("Generated caption_vec length=%d", len(v))
        except Exception as e:
            current_app.logger.warning("Failed to generate caption embedding: %s", e)

        sql_params = (
            d["sku_id"], d["image_path"], d["style"], caption_text, caption_vec_lit,
            d["model_name"], d.get("prompt_version", "v1.0"),
            keywords, colors, shapes, materials, packaging, taste, texture,
            brand_guess, variant_guess, size_guess, category_guess, facet_scores_json
        )

        result = exec_sql("""
            INSERT INTO sku_captions (
                sku_id, image_path, lang, style, caption_text, caption_vec,
                model_name, prompt_version, needs_review,
                keywords, colors, shapes, materials, packaging, taste, texture,
                brand_guess, variant_guess, size_guess, category_guess, facet_scores,
                created_at
            )
            VALUES (%s,%s,'vi',%s,%s,%s::vector,%s,%s,TRUE,
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,NOW())
            ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version)
            DO UPDATE SET 
                caption_text = EXCLUDED.caption_text,
                caption_vec = EXCLUDED.caption_vec,
                keywords = EXCLUDED.keywords,
                colors = EXCLUDED.colors,
                shapes = EXCLUDED.shapes,
                materials = EXCLUDED.materials,
                packaging = EXCLUDED.packaging,
                taste = EXCLUDED.taste,
                texture = EXCLUDED.texture,
                brand_guess = EXCLUDED.brand_guess,
                variant_guess = EXCLUDED.variant_guess,
                size_guess = EXCLUDED.size_guess,
                category_guess = EXCLUDED.category_guess,
                facet_scores = EXCLUDED.facet_scores,
                needs_review = TRUE
            RETURNING id
        """, sql_params, returning=True)

        caption_id = result[0] if result else None

        # refresh corpus (best-effort)
        try:
            exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
        except Exception:
            pass

        return jsonify({"ok": True, "caption_id": caption_id, "message": "Caption saved successfully"})

    except Exception as e:
        current_app.logger.exception("Error saving caption suggestion")
        return jsonify({
            "ok": False,
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500

# ==============================
# LABEL CAPTION (human feedback)
# ==============================
@bp.post("/captions/<int:caption_id>/label", endpoint="captions_label")
def captions_label(caption_id):
    """Save human feedback/labels cho caption."""
    try:
        d = request.get_json(force=True)

        cap = q("SELECT id, caption_text FROM sku_captions WHERE id = %s",
                (caption_id,), fetch="one")
        if not cap:
            return {"ok": False, "error": "Caption not found"}, 404

        is_acceptable = bool(d.get("is_acceptable", True))
        corrected_text = (d.get("corrected_text") or "").strip() or None
        notes = (d.get("notes") or "").strip() or None

        exec_sql("""
            INSERT INTO caption_labels (caption_id, is_acceptable, corrected_text, notes, created_at)
            VALUES (%s,%s,%s,%s,NOW())
            ON CONFLICT (caption_id) DO UPDATE SET
                is_acceptable = EXCLUDED.is_acceptable,
                corrected_text = EXCLUDED.corrected_text,
                notes = EXCLUDED.notes,
                updated_at = NOW()
        """, (caption_id, is_acceptable, corrected_text, notes))

        exec_sql("""
            UPDATE sku_captions 
            SET needs_review = CASE WHEN %s = TRUE THEN FALSE ELSE TRUE END,
                updated_at = NOW()
            WHERE id = %s
        """, (is_acceptable, caption_id))

        # Nếu có corrected_text -> tạo phiên bản corrected (ghi đè theo (sku,image,style,model,version))
        if corrected_text and corrected_text != cap.caption_text:
            try:
                # vector cho corrected_text
                corrected_vec_lit = None
                try:
                    v = embed_text(corrected_text)
                    v = _l2norm(v)
                    corrected_vec_lit = _vec_literal(v)
                except Exception as e:
                    current_app.logger.warning("Vector for corrected text failed: %s", e)

                original = q("""
                    SELECT sku_id, image_path, lang, style, model_name, prompt_version
                    FROM sku_captions WHERE id = %s
                """, (caption_id,), fetch="one")

                if original:
                    exec_sql("""
                        INSERT INTO sku_captions (
                            sku_id, image_path, lang, style, caption_text, caption_vec,
                            model_name, prompt_version, needs_review, created_at, updated_at
                        )
                        VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s,FALSE,NOW(),NOW())
                        ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version)
                        DO UPDATE SET 
                            caption_text = EXCLUDED.caption_text,
                            caption_vec = EXCLUDED.caption_vec,
                            needs_review = FALSE,
                            updated_at = NOW()
                    """, (
                        original.sku_id, original.image_path, original.lang, original.style,
                        corrected_text, corrected_vec_lit,
                        f"{original.model_name}-corrected", original.prompt_version
                    ))
                    current_app.logger.info("Created corrected caption version for id=%s", caption_id)

            except Exception as e:
                current_app.logger.exception("Failed to create corrected caption version")
                # Non-fatal

        try:
            exec_sql("REFRESH MATERIALIZED VIEW CONCURRENTLY sku_search_corpus")
        except Exception as e:
            current_app.logger.warning("Failed to refresh search corpus after labeling: %s", e)

        return {"ok": True, "message": "Caption feedback saved successfully"}

    except Exception as e:
        current_app.logger.exception("Error saving caption label")
        return {"ok": False, "error": str(e)}, 500

# ==============================
# GET pending-review captions
# ==============================
@bp.get("/captions/pending-review")
def captions_pending_review():
    """Get captions cần review."""
    try:
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        captions = q("""
            SELECT 
                c.id, c.sku_id, c.image_path, c.style, c.caption_text, 
                c.model_name, c.created_at, c.needs_review,
                s.name as sku_name,
                cl.is_acceptable, cl.corrected_text, cl.notes
            FROM sku_captions c
            LEFT JOIN skus s ON s.id = c.sku_id
            LEFT JOIN caption_labels cl ON cl.caption_id = c.id
            WHERE c.needs_review = TRUE
            ORDER BY c.created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))

        total = q("SELECT COUNT(*) FROM sku_captions WHERE needs_review = TRUE", fetch="one")[0]

        return {
            "ok": True,
            "captions": [dict(c._asdict()) for c in captions] if captions else [],
            "total": total, "limit": limit, "offset": offset
        }

    except Exception as e:
        current_app.logger.exception("Error fetching pending review captions")
        return {"ok": False, "error": str(e)}, 500

# ==============================
# TEST single-image generation (no DB write)
# ==============================
@bp.post("/admin/captions/test_qwen/<int:sku_id>")
def test_qwen_caption(sku_id):
    """
    Test endpoint tạo caption cho 1 ảnh (search + seo) và trả JSON (không lưu DB).
    """
    try:
        from tools.qwen2vl_autogen import generate_caption
        from PIL import Image
    except Exception as e:
        return {"ok": False, "error": f"Cannot load qwen module: {str(e)}"}, 500

    # Log CUDA
    _log_cuda_env(current_app.logger)
    dev = _torch_device()
    os.environ.setdefault("QWEN_DEVICE", dev)

    row = q("""
        SELECT si.id, si.sku_id, si.image_path, si.ocr_text, s.name
        FROM sku_images si
        JOIN skus s ON s.id = si.sku_id
        WHERE si.sku_id = %s
        ORDER BY si.is_primary DESC, si.id
        LIMIT 1
    """, (sku_id,), fetch="one")

    if not row:
        return {"ok": False, "error": f"No images found for SKU {sku_id}"}, 404

    img_id, curr_sku_id, image_path, ocr_text, sku_name = row
    upload_dir = current_app.config.get("UPLOAD_DIR",
                r"E:\api_hango\flask_pgvector_shop\flask_pgvector_shop\uploads")

    try:
        fpath = image_path if os.path.isabs(image_path) else os.path.join(upload_dir, image_path)
        if not os.path.exists(fpath):
            return {"ok": False, "error": f"Image file not found: {fpath}"}, 404

        img = Image.open(fpath).convert("RGB")

        prompt_search = PROMPT_SEARCH + (f"\nVăn bản OCR: {ocr_text}" if ocr_text else "")
        prompt_seo    = PROMPT_SEO    + (f"\nVăn bản OCR: {ocr_text}" if ocr_text else "")

        t0 = time.time()
        # gọi với device nếu tool hỗ trợ
        try:
            search_caption = generate_caption(img, prompt_search, device=dev)
        except TypeError:
            search_caption = generate_caption(img, prompt_search)
        t1 = time.time()

        try:
            seo_caption = generate_caption(img, prompt_seo, device=dev)
        except TypeError:
            seo_caption = generate_caption(img, prompt_seo)
        t2 = time.time()

        return {
            "ok": True,
            "sku_id": sku_id,
            "sku_name": sku_name,
            "image_path": image_path,
            "fpath": fpath,
            "image_id": img_id,
            "ocr_text": ocr_text,
            "captions": {
                "search": {"text": search_caption, "prompt": prompt_search, "generation_time": round(t1 - t0, 2)},
                "seo":    {"text": seo_caption,    "prompt": prompt_seo,    "generation_time": round(t2 - t1, 2)},
            },
            "total_time": round(t2 - t0, 2),
            "model_info": {
                "backend": os.getenv("QWEN_VL_BACKEND", "transformers"),
                "model_path": os.getenv("QWEN_GGUF", ""),
                "hf_base": os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct"),
                "device": dev
            }
        }

    except Exception as e:
        current_app.logger.exception("Error testing Qwen caption generation")
        return {"ok": False, "error": str(e)}, 500

# ==============================
# Save test captions to DB
# ==============================
@bp.post("/admin/captions/save_test_caption")
def save_test_caption():
    """
    Lưu caption đã test vào DB.
    Payload: {"sku_id": ..., "image_path": "...", "search_caption": "...", "seo_caption": "..."}
    """
    try:
        data = request.get_json()
        if not data:
            return {"ok": False, "error": "No JSON data provided"}, 400

        required_fields = ["sku_id", "image_path", "search_caption", "seo_caption"]
        for field in required_fields:
            if not data.get(field):
                return {"ok": False, "error": f"Missing field: {field}"}, 400

        sku_id = int(data["sku_id"])
        image_path = data["image_path"]
        search_caption = (data["search_caption"] or "").strip()
        seo_caption = (data["seo_caption"] or "").strip()

        exec_sql("""
            INSERT INTO sku_captions
                (sku_id, image_path, lang, style, caption_text, model_name, prompt_version, needs_review)
            VALUES (%s,%s,'vi','search',%s,%s,'v1',TRUE)
            ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version) DO UPDATE
               SET caption_text = EXCLUDED.caption_text, needs_review = TRUE
        """, (sku_id, image_path, search_caption, "qwen2vl-test"))

        exec_sql("""
            INSERT INTO sku_captions
                (sku_id, image_path, lang, style, caption_text, model_name, prompt_version, needs_review)
            VALUES (%s,%s,'vi','seo',%s,%s,'v1',TRUE)
            ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version) DO UPDATE
               SET caption_text = EXCLUDED.caption_text, needs_review = TRUE
        """, (sku_id, image_path, seo_caption, "qwen2vl-test"))

        try:
            exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
        except Exception:
            pass

        return {"ok": True, "message": f"Saved captions for SKU {sku_id}", "sku_id": sku_id, "captions_saved": 2}

    except Exception as e:
        current_app.logger.exception("Error saving test caption")
        return {"ok": False, "error": str(e)}, 500

# ==============================
# READ: captions by SKU
# ==============================
@bp.get("/captions/by-sku/<int:sku_id>", endpoint="captions_by_sku")
def captions_by_sku(sku_id: int):
    """
    Trả về JSON danh sách caption cho 1 SKU; filter theo style (search|seo|all) & needs_review (1|0|all)
    """
    style = (request.args.get("style") or "all").lower()
    nr = request.args.get("needs_review", "all").lower()

    where = ["c.sku_id = %s"]
    params = [sku_id]

    if style in ("search", "seo"):
        where.append("c.style = %s")
        params.append(style)
    if nr in ("1", "true", "yes"):
        where.append("c.needs_review = TRUE")
    elif nr in ("0", "false", "no"):
        where.append("c.needs_review = FALSE")

    sql = f"""
        SELECT 
            c.id, c.sku_id, c.image_path, c.lang, c.style, c.caption_text,
            c.model_name, c.prompt_version, c.needs_review, c.is_ground_truth,
            c.created_at, c.summary_text, 
            c.keywords, c.colors, c.shapes, c.materials, c.packaging, c.taste, c.texture,
            c.brand_guess, c.variant_guess, c.size_guess, c.category_guess
        FROM sku_captions c
        WHERE {' AND '.join(where)}
        ORDER BY c.image_path, c.style, c.created_at DESC
    """
    rows = q(sql, tuple(params)) or []
    return {"ok": True, "captions": [dict(r._asdict()) for r in rows], "sku_id": sku_id}

# ==============================
# FORM actions: accept / reject / ground truth / delete
# ==============================
def _caption_exists(caption_id: int):
    return q("SELECT id, sku_id, style, image_path FROM sku_captions WHERE id=%s",
             (caption_id,), fetch="one")

@bp.post("/admin/captions/<int:caption_id>/accept", endpoint="caption_accept")
def caption_accept(caption_id: int):
    cap = _caption_exists(caption_id)
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus.skus"))
    exec_sql("UPDATE sku_captions SET needs_review=FALSE, updated_at=NOW() WHERE id=%s", (caption_id,))
    try:
        exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    except Exception:
        pass
    flash(f"Đã đánh dấu caption #{caption_id} là ACCEPTED", "success")
    return redirect(request.referrer or url_for("skus.skus"))

@bp.post("/admin/captions/<int:caption_id>/reject", endpoint="caption_reject")
def caption_reject(caption_id: int):
    cap = _caption_exists(caption_id)
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus.skus"))
    notes = (request.form.get("notes") or "").strip() or None
    exec_sql("""
        INSERT INTO caption_labels(caption_id, is_acceptable, corrected_text, notes, created_at)
        VALUES (%s, FALSE, NULL, %s, NOW())
        ON CONFLICT (caption_id) DO UPDATE SET is_acceptable=FALSE, notes=%s, updated_at=NOW()
    """, (caption_id, notes, notes))
    exec_sql("UPDATE sku_captions SET needs_review=TRUE, updated_at=NOW() WHERE id=%s", (caption_id,))
    flash(f"Đã REJECT caption #{caption_id}", "warning")
    return redirect(request.referrer or url_for("skus.skus"))

@bp.post("/admin/captions/<int:caption_id>/ground", endpoint="caption_ground")
def caption_ground(caption_id: int):
    cap = q("SELECT id, sku_id, image_path, style FROM sku_captions WHERE id=%s",
            (caption_id,), fetch="one")
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus.skus"))

    exec_sql("""
        UPDATE sku_captions
           SET is_ground_truth = CASE WHEN id=%s THEN TRUE ELSE FALSE END,
               updated_at = NOW()
         WHERE sku_id=%s AND image_path=%s AND style=%s
    """, (caption_id, cap.sku_id, cap.image_path, cap.style))
    flash(f"Đã đặt caption #{caption_id} là Ground Truth", "success")
    return redirect(request.referrer or url_for("skus.skus"))

@bp.post("/admin/captions/<int:caption_id>/delete", endpoint="caption_delete")
def caption_delete(caption_id: int):
    cap = _caption_exists(caption_id)
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus.skus"))
    exec_sql("DELETE FROM sku_captions WHERE id=%s", (caption_id,))
    flash(f"Đã xoá caption #{caption_id}", "secondary")
    return redirect(request.referrer or url_for("skus.skus"))

# ==============================
# API: images pending (no search caption yet)
# ==============================
@bp.get("/api/captions/pending", endpoint="api_captions_pending")
def api_captions_pending():
    """
    API: Lấy danh sách ảnh chưa có caption 'search' (có filter theo model_name).
    Query params:
      - sku_id (int, optional)
      - limit (int, default=100)
      - model_name (str, optional)
    """
    try:
        sku_id = request.args.get("sku_id", type=int)
        limit = request.args.get("limit", type=int, default=100)
        model_name = request.args.get("model_name", type=str)

        where_clauses, params = [], []

        if sku_id:
            where_clauses.append("si.sku_id = %s")
            params.append(sku_id)

        if model_name:
            where_clauses.append("""
                NOT EXISTS (
                    SELECT 1
                    FROM sku_captions sc 
                    WHERE sc.sku_id = si.sku_id 
                      AND sc.image_path = si.image_path 
                      AND sc.style = 'search'
                      AND sc.model_name = %s
                )
            """)
            params.append(model_name)
        else:
            where_clauses.append("""
                NOT EXISTS (
                    SELECT 1
                    FROM sku_captions sc 
                    WHERE sc.sku_id = si.sku_id 
                      AND sc.image_path = si.image_path 
                      AND sc.style = 'search'
                )
            """)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        params.append(limit)

        rows = q(f"""
            SELECT si.id as image_id, si.sku_id, si.image_path, si.ocr_text
            FROM sku_images si
            WHERE {where_sql}
            ORDER BY si.sku_id, si.is_primary DESC, si.id
            LIMIT %s
        """, tuple(params))

        images = [{
            "image_id": r[0], "sku_id": r[1], "image_path": r[2], "ocr_text": r[3]
        } for r in (rows or [])]

        return jsonify({"ok": True, "images": images, "count": len(images)})

    except Exception as e:
        current_app.logger.exception("Error in api_captions_pending")
        return jsonify({"ok": False, "error": str(e)}), 500

# ==============================
# Helper for available facet names (based on DB schema/model output)
# ==============================
@bp.get("/skus/<int:sku_id>/facets", endpoint="sku_facets")
def sku_facets(sku_id):
    """
    Trả về facet đã lưu trong sku_captions cho 1 SKU.
    Dùng để hiển thị bảng facet trong giao diện.
    """
    try:
        rows = q("""
            SELECT 
                keywords, colors, shapes, materials, packaging, taste, texture,
                brand_guess, variant_guess, size_guess, category_guess
            FROM sku_captions
            WHERE sku_id = %s
            ORDER BY id DESC 
            LIMIT 1
        """, (sku_id,))

        if not rows:
            return jsonify({
                "ok": False,
                "message": "No facet data in database"
            }), 200

        row = rows[0]

        return jsonify({
            "ok": True,
            "data": {
                "keywords": row["keywords"],
                "colors": row["colors"],
                "shapes": row["shapes"],
                "materials": row["materials"],
                "packaging": row["packaging"],
                "taste": row["taste"],
                "texture": row["texture"],
                "brand_guess": row["brand_guess"],
                "variant_guess": row["variant_guess"],
                "size_guess": row["size_guess"],
                "category_guess": row["category_guess"],
            }
        })

    except Exception as e:
        current_app.logger.exception("Failed to load facets")
        return jsonify({"ok": False, "error": str(e)}), 500
