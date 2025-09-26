# routes/captions.py
import os, time
from flask import Blueprint, request, redirect, url_for, flash, current_app
from services.db_utils import q, exec_sql
from services.moondream_service import md_chat_vision, _mime_from_path
import traceback

bp = Blueprint("captions_bp", __name__)

# Simple prompt used for caption generation
PROMPT_SEARCH = "Viết caption ngắn 1-2 câu, khách quan, tiếng Việt."
PROMPT_SEO = "Viết mô tả 3-5 câu bằng tiếng Việt cho trang sản phẩm, nhấn mạnh công dụng/đặc điểm."

@bp.post("/admin/captions/autogen", endpoint="captions_autogen")
def captions_autogen():
    limit  = int(request.form.get("limit", request.args.get("limit", 200)))
    offset = int(request.form.get("offset", request.args.get("offset", 0)))

    rows = q("""
        SELECT si.id, si.sku_id, si.image_path, si.ocr_text
        FROM sku_images si
        LEFT JOIN sku_captions sc
               ON sc.sku_id = si.sku_id AND sc.image_path = si.image_path
               AND sc.lang = 'vi' AND sc.style = 'search'
        WHERE sc.id IS NULL
        ORDER BY si.sku_id, si.is_primary DESC, si.id
        LIMIT %s OFFSET %s
    """, (limit, offset))

    done = fail = 0
    for (img_id, sku_id, img_path, ocr_text) in rows:
        try:
            fpath = img_path if os.path.isabs(img_path) else os.path.join(current_app.config["UPLOAD_DIR"], img_path)
            with open(fpath, "rb") as f: img_bytes = f.read()
            mime = _mime_from_path(fpath)
            cap  = md_chat_vision(img_bytes, mime, PROMPT_SEARCH + (f"\nOCR: {ocr_text}" if ocr_text else ""))
            desc = md_chat_vision(img_bytes, mime, PROMPT_SEO    + (f"\nOCR: {ocr_text}" if ocr_text else ""))

            exec_sql("""INSERT INTO sku_captions(sku_id,image_path,lang,style,caption_text,model_name,prompt_version,needs_review)
                        VALUES (%s,%s,'vi','search',%s,%s,'v1.0',TRUE)
                        ON CONFLICT (sku_id,image_path,lang,style,model_name,prompt_version)
                        DO UPDATE SET caption_text=EXCLUDED.caption_text, needs_review=TRUE""",
                     (sku_id, img_path, cap, os.getenv("MOONDREAM_MODEL","moondream-2B")))
            exec_sql("""INSERT INTO sku_captions(sku_id,image_path,lang,style,caption_text,model_name,prompt_version,needs_review)
                        VALUES (%s,%s,'vi','seo',%s,%s,'v1.0',TRUE)
                        ON CONFLICT (sku_id,image_path,lang,style,model_name,prompt_version)
                        DO UPDATE SET caption_text=EXCLUDED.caption_text, needs_review=TRUE""",
                     (sku_id, img_path, desc, os.getenv("MOONDREAM_MODEL","moondream-2B")))
            done += 1
        except Exception as e:
            print("[autogen] error:", e); fail += 1

    try: exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    except: pass

    flash(f"Autogen xong: thÃ nh cÃ´ng {done}, lá»—i {fail}", "success" if fail==0 else "warning")
    return redirect(url_for("skus"))

@bp.post("/admin/captions/autogen_qwen")
def captions_autogen_qwen():
    """
    Autogen caption + LƯU NGỮ NGHĨA:
    - Gọi Qwen2-VL sinh caption 'search' + 'seo'
    - Trigger đã đẩy 'search' -> sku_texts.text
    - Embed caption/segment -> UPDATE sku_texts.text_vec (512D)
    - (optional) Embed ảnh -> UPDATE sku_images.image_vec
    """
    limit  = int(request.form.get("limit") or request.args.get("limit") or 100)
    offset = int(request.form.get("offset") or request.args.get("offset") or 0)

    # lazy import để không tải nặng khi app start
    try:
        from tools.qwen2vl_autogen import generate_caption_from_image, split_segments, embed_text, embed_image
        from PIL import Image
    except Exception as e:
        current_app.logger.exception("Failed to import qwen2vl tool")
        flash("Không thể nạp module qwen autogen: " + str(e), "danger")
        return redirect(url_for("skus_bp.skus"))

    def _vec_literal(vec):
        # serialize -> '[0.123,0.456,...]' để cast ::vector
        return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

    def _l2norm(v):
        import math
        n = math.sqrt(sum(float(x)*float(x) for x in v)) or 1.0
        return [float(x)/n for x in v]

    def _vn_norm(s: str) -> str:
        import unicodedata, re
        s = (s or "").strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"[^a-z0-9 \-\.x/]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _upsert_text_and_vec(sku_id: int, raw_text: str, vec: list, model_name="openclip_vit_b32"):
        """
        - Chuẩn hoá 'text' như trigger đang dùng
        - Đảm bảo có dòng trong sku_texts
        - Ghi vector vào text_vec (và text_vec_model nếu có)
        """
        norm = _vn_norm(raw_text)
        if not norm:
            return

        # đảm bảo có hàng text trong sku_texts
        exec_sql("""
            INSERT INTO sku_texts(sku_id, text)
            VALUES (%s, %s)
            ON CONFLICT (sku_id, text) DO NOTHING
        """, (sku_id, norm))

        if vec:
            vec = _l2norm(vec)
            vec_lit = _vec_literal(vec)
            # thử update cả model_name; nếu cột không tồn tại -> fallback chỉ text_vec
            try:
                exec_sql("""
                    UPDATE sku_texts
                       SET text_vec = %s::vector,
                           text_vec_model = %s
                     WHERE sku_id = %s AND text = %s
                """, (vec_lit, model_name, sku_id, norm))
            except Exception:
                exec_sql("""
                    UPDATE sku_texts
                       SET text_vec = %s::vector
                     WHERE sku_id = %s AND text = %s
                """, (vec_lit, sku_id, norm))

    def _update_image_vec(image_path: str, vec: list, model_name="openclip_vit_b32"):
        if not vec:
            return
        vec = _l2norm(vec)
        vec_lit = _vec_literal(vec)
        try:
            exec_sql("""
                UPDATE sku_images
                   SET image_vec = %s::vector, image_vec_model = %s
                 WHERE image_path = %s
            """, (vec_lit, model_name, image_path))
        except Exception:
            exec_sql("""
                UPDATE sku_images
                   SET image_vec = %s::vector
                 WHERE image_path = %s
            """, (vec_lit, image_path))

    rows = q("""
        SELECT si.id, si.sku_id, si.image_path, si.ocr_text
        FROM sku_images si
        LEFT JOIN sku_captions sc
               ON sc.sku_id = si.sku_id
              AND sc.image_path = si.image_path
              AND sc.lang = 'vi'
              AND sc.style = 'search'
        WHERE sc.id IS NULL
        ORDER BY si.sku_id, si.is_primary DESC, si.id
        LIMIT %s OFFSET %s
    """, (limit, offset))

    done, fail = 0, 0
    upload_dir = current_app.config.get("UPLOAD_DIR", "uploads")

    for img_id, sku_id, image_path, ocr_text in rows:
        try:
            fpath = image_path if os.path.isabs(image_path) else os.path.join(upload_dir, image_path)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Không tìm thấy ảnh: {fpath}")

            img = Image.open(fpath).convert("RGB")

            # Prompt (đã tối ưu cho search/seo)
            prompt_search = PROMPT_SEARCH + (f"\nVăn bản OCR: {ocr_text}" if ocr_text else "")
            prompt_seo    = PROMPT_SEO    + (f"\nVăn bản OCR: {ocr_text}" if ocr_text else "")

            # 1) Gọi Qwen sinh caption
            cap  = generate_caption_from_image(img, prompt_search)  # 1 câu, giàu từ khoá
            desc = generate_caption_from_image(img, prompt_seo)     # 1–2 câu

            # 2) Lưu sku_captions (trigger sẽ đẩy 'search' -> sku_texts.text)
            exec_sql("""
                INSERT INTO sku_captions
                    (sku_id, image_path, lang, style, caption_text, model_name, prompt_version, needs_review)
                VALUES (%s,%s,'vi','search',%s,%s,'v1',TRUE)
                ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version) DO UPDATE
                   SET caption_text = EXCLUDED.caption_text, needs_review = TRUE
            """, (sku_id, image_path, cap, "qwen2vl-lora"))

            exec_sql("""
                INSERT INTO sku_captions
                    (sku_id, image_path, lang, style, caption_text, model_name, prompt_version, needs_review)
                VALUES (%s,%s,'vi','seo',%s,%s,'v1',TRUE)
                ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version) DO UPDATE
                   SET caption_text = EXCLUDED.caption_text, needs_review = TRUE
            """, (sku_id, image_path, desc, "qwen2vl-lora"))

            # 3) EMBED & LƯU NGỮ NGHĨA (text_vec + image_vec)
            # 3a) Embed caption 'search' (full câu)
            try:
                v_cap = embed_text(cap)          # -> list[512]
                _upsert_text_and_vec(sku_id, cap, v_cap, model_name="openclip_vit_b32")
            except Exception:
                current_app.logger.warning("Embed caption failed (sku=%s)", sku_id)

            # 3b) Embed từng "đoạn" (nếu bạn muốn thêm nhiều hàng text để phủ token)
            try:
                for seg in split_segments(cap):
                    if seg and seg.strip():
                        v_seg = embed_text(seg)
                        _upsert_text_and_vec(sku_id, seg, v_seg, model_name="openclip_vit_b32")
            except Exception:
                pass  # không bắt buộc

            # 3c) (Tuỳ chọn) Embed ảnh -> image_vec
            try:
                v_img = embed_image(img)        # -> list[512]
                _update_image_vec(image_path, v_img, model_name="openclip_vit_b32")
            except Exception:
                # không chặn pipeline nếu embed ảnh lỗi
                pass

            done += 1
            time.sleep(0.15)  # giảm tải

        except Exception:
            fail += 1
            current_app.logger.exception("Autogen error img_id=%s sku_id=%s", img_id, sku_id)

    # 4) Refresh corpus để BM25/fuzzy bắt kịp caption mới
    try:
        exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    except Exception:
        pass

    flash(f"Autogen QWEN (lưu ngữ nghĩa): OK={done}, FAIL={fail}", "success" if fail==0 else "warning")
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

