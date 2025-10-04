# routes/captions.py
import os, time
from flask import Blueprint, request, redirect, url_for, flash, current_app
from services.db_utils import q, exec_sql
from services.moondream_service import md_chat_vision, _mime_from_path
import traceback
from pathlib import Path

bp = Blueprint("captions_bp", __name__)
UPLOAD_DIR = os.path.abspath(os.getenv("UPLOAD_FOLDER", "uploads"))
# Simple prompt used for caption generation
PROMPT_SEARCH = "Viết caption ngắn 1-2 câu, khách quan, tiếng Việt."
PROMPT_SEO = "Viết mô tả 3-5 câu bằng tiếng Việt cho trang sản phẩm, nhấn mạnh công dụng/đặc điểm."

@bp.post("/admin/captions/autogen/<int:sku_id>", endpoint="captions_autogen")
def captions_autogen(sku_id):
    limit = int(request.form.get("limit", request.args.get("limit", 200)))
    offset = int(request.form.get("offset", request.args.get("offset", 0)))

    # Lazy import để tránh tải nặng khi app start
    try:
        from tools.qwen2vl_autogen import generate_caption_struct
        from PIL import Image
        import json
        from io import BytesIO
        from pathlib import Path
    except Exception as e:
        current_app.logger.exception("Failed to import qwen2vl tool")
        flash("Không thể nạp module qwen autogen: " + str(e), "danger")
        return redirect(url_for("skus_bp.skus"))

    # Hàm hỗ trợ để chuẩn hóa vector
    def _vec_literal(vec):
        return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]" if vec else None

    def _l2norm(v):
        import math
        n = math.sqrt(sum(float(x)*float(x) for x in v)) or 1.0
        return [float(x)/n for x in v]

    def embed_text(text):
        """Trả về embedding vector cho text, sử dụng utils.encode_texts nếu có"""
        try:
            from utils import encode_texts
            emb = encode_texts([text])[0]
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as e:
            import numpy as np
            current_app.logger.warning("embed_text fallback: %s", e)
            return np.random.rand(512).tolist()

    # Query chỉ lấy ảnh của sku_id cụ thể chưa có caption
    rows = q("""
        SELECT si.id, si.sku_id, si.image_path, si.ocr_text
        FROM sku_images si
        LEFT JOIN sku_captions sc
               ON sc.sku_id = si.sku_id AND sc.image_path = si.image_path
               AND sc.lang = 'vi' AND sc.style = 'search'
        WHERE si.sku_id = %s
        ORDER BY si.is_primary DESC, si.id
        LIMIT %s OFFSET %s
    """, (sku_id, limit, offset))

    if not rows:
        current_app.logger.info(f"No images found for sku_id {sku_id} without captions")
        flash(f"Không tìm thấy ảnh nào cho SKU {sku_id} cần tạo caption.", "warning")
        return redirect(url_for("skus_bp.skus"))

    done = fail = 0
    model_name = os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct").split("/")[-1]
    upload_dir = current_app.config.get("UPLOAD_DIR", "uploads")

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

            # Sinh caption và facets sử dụng generate_caption_struct
            struct_data = generate_caption_struct(str(p), max_new_tokens=200)

            # Lấy các giá trị từ struct_data
            caption = struct_data.get("caption", "Sản phẩm trong ảnh")
            keywords = struct_data.get("keywords", [])  # Truyền trực tiếp danh sách cho text[]
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

            print(struct_data)
            # Tạo embedding cho caption + keywords
            keywords_str = " ".join(keywords) if keywords else ""
            combined_text = f"{caption} {keywords_str}".strip()
            caption_vec = embed_text(combined_text) if combined_text else []
            caption_vec_lit = _vec_literal(_l2norm(caption_vec)) if caption_vec else None

            # Đọc bytes ảnh cho md_chat_vision
            img_bytes = None
            try:
                with open(fpath, "rb") as fh:
                    img_bytes = fh.read()
            except Exception as e:
                current_app.logger.warning(f"Failed to read image bytes {fpath}: {str(e)}")
                fail += 1
                continue

            # Sinh SEO caption
            try:
                seo_caption = md_chat_vision(img_bytes, _mime_from_path(fpath), PROMPT_SEO + (f"\nOCR: {ocr_text}" if ocr_text else ""))
            except Exception as e:
                current_app.logger.warning(f"md_chat_vision failed for {fpath}: {str(e)}")
                seo_caption = caption  # Fallback: sử dụng caption search

            # Lưu caption 'search' với đầy đủ facets
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

            # Lưu caption 'seo' (chỉ caption_text)
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

    try:
        exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    except Exception as e:
        current_app.logger.warning(f"Failed to refresh search corpus: {str(e)}")

    flash(f"Autogen xong cho SKU {sku_id}: thành công {done}, lỗi {fail}", "success" if fail == 0 else "warning")
    return redirect(url_for("skus_bp.skus"))

@bp.post("/captions/suggest", endpoint="captions_suggest")
def captions_suggest():
    """API endpoint to save/update caption suggestions"""
    try:
        d = request.get_json(force=True)
        
        # Validate required fields
        required_fields = ["sku_id", "image_path", "style", "caption_text", "model_name"]
        for field in required_fields:
            if not d.get(field):
                return {"ok": False, "error": f"Missing required field: {field}"}, 400
        
        # Clean and validate caption text
        caption_text = (d["caption_text"] or "").strip()
        if not caption_text:
            return {"ok": False, "error": "Caption text cannot be empty"}, 400
            
        # Generate caption embedding for search purposes
        caption_vector = None
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Normalize Vietnamese text if available
            try:
                from utils import vn_norm
                normalized_caption = vn_norm(caption_text)
            except ImportError:
                normalized_caption = caption_text.lower().strip()
            
            embedding = model.encode(normalized_caption, normalize_embeddings=True)
            caption_vector = embedding.tolist()
            current_app.logger.info("Generated vector embedding for caption (dim=%d)", len(caption_vector))
        except Exception as e:
            current_app.logger.warning("Failed to generate caption embedding: %s", e)
            # Continue without vector - can be backfilled later
        
        # Insert/update caption with vector
        result = exec_sql("""
            INSERT INTO sku_captions (
                sku_id, image_path, lang, style, caption_text, 
                caption_vector, model_name, prompt_version, needs_review, created_at, updated_at
            )
            VALUES (%s,%s,'vi',%s,%s,%s,%s,%s,TRUE,NOW(),NOW())
            ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version)
            DO UPDATE SET 
                caption_text = EXCLUDED.caption_text,
                caption_vector = EXCLUDED.caption_vector,
                needs_review = TRUE,
                updated_at = NOW()
            RETURNING id
        """, (
            d["sku_id"], 
            d["image_path"], 
            d["style"], 
            caption_text,
            caption_vector,
            d["model_name"], 
            d.get("prompt_version", "v1.0")
        ), returning=True)
        
        caption_id = result[0][0] if result else None
        
        # Auto-refresh search corpus to include new caption
        try:
            exec_sql("REFRESH MATERIALIZED VIEW CONCURRENTLY sku_search_corpus")
            current_app.logger.info("Auto-refreshed search corpus after caption save")
        except Exception as e:
            current_app.logger.warning("Failed to refresh search corpus: %s", e)
            # Non-fatal - corpus can be refreshed manually
        
        return {
            "ok": True, 
            "caption_id": caption_id,
            "message": "Caption saved successfully with vector embedding"
        }
        
    except Exception as e:
        current_app.logger.exception("Error saving caption suggestion")
        return {"ok": False, "error": str(e)}, 500

@bp.post("/captions/<int:caption_id>/label", endpoint="captions_label")
def captions_label(caption_id):
    """API endpoint to save human feedback/labels for captions"""
    try:
        d = request.get_json(force=True)
        
        # Validate caption exists
        caption = q("SELECT id, caption_text FROM sku_captions WHERE id = %s", (caption_id,), fetch="one")
        if not caption:
            return {"ok": False, "error": "Caption not found"}, 404
        
        is_acceptable = bool(d.get("is_acceptable", True))
        corrected_text = (d.get("corrected_text") or "").strip() or None
        notes = (d.get("notes") or "").strip() or None
        
        # Save human label/feedback
        exec_sql("""
            INSERT INTO caption_labels (caption_id, is_acceptable, corrected_text, notes, created_at)
            VALUES (%s,%s,%s,%s,NOW())
            ON CONFLICT (caption_id) DO UPDATE SET
                is_acceptable = EXCLUDED.is_acceptable,
                corrected_text = EXCLUDED.corrected_text,
                notes = EXCLUDED.notes,
                updated_at = NOW()
        """, (caption_id, is_acceptable, corrected_text, notes))
        
        # Update caption needs_review status based on feedback
        exec_sql("""
            UPDATE sku_captions 
            SET needs_review = CASE 
                WHEN %s = TRUE THEN FALSE  -- acceptable = no more review needed
                ELSE TRUE                  -- not acceptable = still needs review
            END,
            updated_at = NOW()
            WHERE id = %s
        """, (is_acceptable, caption_id))
        
        # If corrected text provided, optionally create new caption version
        if corrected_text and corrected_text != caption.caption_text:
            try:
                # Generate vector for corrected text
                corrected_vector = None
                try:
                    from sentence_transformers import SentenceTransformer
                    from utils import vn_norm
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    normalized = vn_norm(corrected_text)
                    embedding = model.encode(normalized, normalize_embeddings=True)
                    corrected_vector = embedding.tolist()
                except Exception as e:
                    current_app.logger.warning("Failed to generate vector for corrected text: %s", e)
                
                # Get original caption info for creating corrected version
                original = q("""
                    SELECT sku_id, image_path, lang, style, model_name, prompt_version
                    FROM sku_captions WHERE id = %s
                """, (caption_id,), fetch="one")
                
                if original:
                    exec_sql("""
                        INSERT INTO sku_captions (
                            sku_id, image_path, lang, style, caption_text, caption_vector,
                            model_name, prompt_version, needs_review, created_at, updated_at
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FALSE,NOW(),NOW())
                        ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version)
                        DO UPDATE SET 
                            caption_text = EXCLUDED.caption_text,
                            caption_vector = EXCLUDED.caption_vector,
                            needs_review = FALSE,
                            updated_at = NOW()
                    """, (
                        original.sku_id, original.image_path, original.lang, original.style,
                        corrected_text, corrected_vector,
                        f"{original.model_name}-corrected", original.prompt_version
                    ))
                    current_app.logger.info("Created corrected caption version for caption_id=%s", caption_id)
                    
            except Exception as e:
                current_app.logger.exception("Failed to create corrected caption version")
                # Non-fatal - original label still saved
        
        # Refresh search corpus to include updated captions
        try:
            exec_sql("REFRESH MATERIALIZED VIEW CONCURRENTLY sku_search_corpus")
        except Exception as e:
            current_app.logger.warning("Failed to refresh search corpus after labeling: %s", e)
        
        return {
            "ok": True, 
            "message": "Caption feedback saved successfully"
        }
        
    except Exception as e:
        current_app.logger.exception("Error saving caption label")
        return {"ok": False, "error": str(e)}, 500

@bp.get("/captions/pending-review")
def captions_pending_review():
    """Get captions that need human review"""
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
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        current_app.logger.exception("Error fetching pending review captions")
        return {"ok": False, "error": str(e)}, 500

@bp.post("/admin/captions/test_qwen/<int:sku_id>")
def test_qwen_caption(sku_id):
    """
    Test endpoint để tạo caption cho 1 ảnh của SKU và trả về JSON
    Không lưu vào database, chỉ test generation
    """
    try:
        from tools.qwen2vl_autogen import generate_caption
        from PIL import Image
    except Exception as e:
        return {"ok": False, "error": f"Cannot load qwen module: {str(e)}"}, 500

    # Get primary image for this SKU
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
    # use configured upload dir if present, otherwise a safe raw Windows path
    upload_dir = current_app.config.get("UPLOAD_DIR",
                r"E:\api_hango\flask_pgvector_shop\flask_pgvector_shop\uploads")
    try:
        fpath = image_path if os.path.isabs(image_path) else os.path.join(upload_dir, image_path)
        if not os.path.exists(fpath):
            return {"ok": False, "error": f"Image file not found: {fpath}"}, 404

        img = Image.open(fpath).convert("RGB")
        print(fpath)

        # Generate prompts
        prompt_search = PROMPT_SEARCH + (f"\nVăn bản OCR: {ocr_text}" if ocr_text else "")
        prompt_seo    = PROMPT_SEO    + (f"\nVăn bản OCR: {ocr_text}" if ocr_text else "")

        # Test caption generation
        start_time = time.time()
        
        search_caption = generate_caption(img, prompt_search)
        search_time = time.time() - start_time
        
        seo_start = time.time()
        seo_caption = generate_caption(img, prompt_seo)
        seo_time = time.time() - seo_start

        total_time = time.time() - start_time
        return {
            "ok": True,
            "sku_id": sku_id,
            "sku_name": sku_name,
            "image_path": image_path,
            "fpath": fpath,
            "image_id": img_id,
            "ocr_text": ocr_text,
            "captions": {
                "search": {
                    "text": search_caption,
                    "prompt": prompt_search,
                    "generation_time": round(search_time, 2)
                },
                "seo": {
                    "text": seo_caption,
                    "prompt": prompt_seo,
                    "generation_time": round(seo_time, 2)
                }
            },
            "total_time": round(total_time, 2),
            "model_info": {
                "backend": os.getenv("QWEN_VL_BACKEND", "transformers"),
                "model_path": os.getenv("QWEN_GGUF", ""),
                "hf_base": os.getenv("QWEN_VL_BASE", "Qwen/Qwen2-VL-2B-Instruct")
            }
        }

    except Exception as e:
        current_app.logger.exception("Error testing Qwen caption generation")
        return {"ok": False, "error": str(e)}, 500

@bp.post("/admin/captions/save_test_caption")
def save_test_caption():
    """
    Lưu caption đã test vào database
    Expects JSON: {"sku_id": 123, "image_path": "...", "search_caption": "...", "seo_caption": "...}
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
        search_caption = data["search_caption"].strip()
        seo_caption = data["seo_caption"].strip()

        # Save both captions
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

        # Refresh corpus
        try:
            exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
        except Exception:
            pass

        return {
            "ok": True,
            "message": f"Saved captions for SKU {sku_id}",
            "sku_id": sku_id,
            "captions_saved": 2
        }

    except Exception as e:
        current_app.logger.exception("Error saving test caption")
        return {"ok": False, "error": str(e)}, 500

# === READ: lấy captions theo SKU (group theo ảnh, style) ===
@bp.get("/captions/by-sku/<int:sku_id>", endpoint="captions_by_sku")
def captions_by_sku(sku_id: int):
    """
    Trả về JSON danh sách caption cho 1 SKU:
    - Có thể lọc theo style (=search|seo|all) & needs_review (=1|0|all)
    - Mặc định: all
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
    return {
        "ok": True,
        "captions": [dict(r._asdict()) for r in rows],
        "sku_id": sku_id
    }

# === FORM-ACTIONS: accept / reject / ground truth / delete ===
def _caption_exists(caption_id: int):
    cap = q("SELECT id, sku_id, style, image_path FROM sku_captions WHERE id=%s", (caption_id,), fetch="one")
    return cap

@bp.post("/admin/captions/<int:caption_id>/accept", endpoint="caption_accept")
def caption_accept(caption_id: int):
    cap = _caption_exists(caption_id)
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus_bp.skus"))
    exec_sql("UPDATE sku_captions SET needs_review=FALSE, updated_at=NOW() WHERE id=%s", (caption_id,))
    try: exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    except: pass
    flash(f"Đã đánh dấu caption #{caption_id} là ACCEPTED", "success")
    return redirect(request.referrer or url_for("skus_bp.skus"))

@bp.post("/admin/captions/<int:caption_id>/reject", endpoint="caption_reject")
def caption_reject(caption_id: int):
    cap = _caption_exists(caption_id)
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus_bp.skus"))
    notes = (request.form.get("notes") or "").strip() or None
    # Lưu label + giữ needs_review=TRUE
    exec_sql("""
        INSERT INTO caption_labels(caption_id, is_acceptable, corrected_text, notes, created_at)
        VALUES (%s, FALSE, NULL, %s, NOW())
        ON CONFLICT (caption_id) DO UPDATE SET is_acceptable=FALSE, notes=%s, updated_at=NOW()
    """, (caption_id, notes, notes))
    exec_sql("UPDATE sku_captions SET needs_review=TRUE, updated_at=NOW() WHERE id=%s", (caption_id,))
    flash(f"Đã REJECT caption #{caption_id}", "warning")
    return redirect(request.referrer or url_for("skus_bp.skus"))

@bp.post("/admin/captions/<int:caption_id>/ground", endpoint="caption_ground")
def caption_ground(caption_id: int):
    """
    Đánh dấu caption là Ground Truth cho cặp (sku_id, image_path, style)
    Và huỷ GT của các caption khác cùng (sku, image, style).
    """
    cap = q("""
        SELECT id, sku_id, image_path, style FROM sku_captions WHERE id=%s
    """, (caption_id,), fetch="one")
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus_bp.skus"))

    exec_sql("""
        UPDATE sku_captions
           SET is_ground_truth = CASE WHEN id=%s THEN TRUE ELSE FALSE END,
               updated_at = NOW()
         WHERE sku_id=%s AND image_path=%s AND style=%s
    """, (caption_id, cap.sku_id, cap.image_path, cap.style))
    flash(f"Đã đặt caption #{caption_id} là Ground Truth", "success")
    return redirect(request.referrer or url_for("skus_bp.skus"))

@bp.post("/admin/captions/<int:caption_id>/delete", endpoint="caption_delete")
def caption_delete(caption_id: int):
    cap = _caption_exists(caption_id)
    if not cap:
        flash("Caption không tồn tại", "danger")
        return redirect(request.referrer or url_for("skus_bp.skus"))
    exec_sql("DELETE FROM sku_captions WHERE id=%s", (caption_id,))
    flash(f"Đã xoá caption #{caption_id}", "secondary")
    return redirect(request.referrer or url_for("skus_bp.skus"))
