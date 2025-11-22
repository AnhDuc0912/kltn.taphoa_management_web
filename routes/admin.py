# routes/admin.py
import os, csv, datetime
from flask import Blueprint, send_file, request, redirect, url_for, flash, jsonify, current_app
from services.db_utils import q, exec_sql
from services.clip_service import embed_text_clip_512
from utils import vn_norm
from sentence_transformers import SentenceTransformer
import logging
import open_clip
import torch
from PIL import Image

bp = Blueprint("admin_bp", __name__)

# Load CLIP model (lazy init)
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None

def get_clip_model():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        model_name = os.getenv("EMBED_MODEL", "ViT-B-32")
        pretrained = os.getenv("EMBED_PRETRAINED", "laion2b_s34b_b79k")
        
        logger.info("Loading CLIP model: %s/%s", model_name, pretrained)
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _clip_tokenizer = open_clip.get_tokenizer(model_name)
        _clip_model.eval()
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(device)
        logger.info("CLIP model loaded on device: %s", device)
    
    return _clip_model, _clip_preprocess, _clip_tokenizer

logger = logging.getLogger("admin")

@bp.post("/admin/search-corpus/refresh")
def refresh_corpus():
    exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    return {"ok": True}

# @bp.post("/admin/text-vector-backfill-clip")
# def text_vec_backfill_clip():
#     """Backfill text embeddings using CLIP text encoder"""
#     try:
#         limit = int(request.form.get("limit") or request.args.get("limit") or 100)
        
#         # Find sku_texts without embeddings
#         rows = q("""
#             SELECT id, sku_id, text
#             FROM sku_texts 
#             WHERE text_vec IS NULL
#             ORDER BY id 
#             LIMIT %s
#         """, (limit,))
        
#         if not rows:
#             flash("Không có text nào cần backfill vector", "info")
#             return redirect(url_for("skus.skus"))

#         model, preprocess, tokenizer = get_clip_model()
#         device = next(model.parameters()).device
#         processed = 0
#         failed = 0

#         for row in rows:
#             try:
#                 # Access tuple by index: (id, sku_id, text)
#                 row_id = row[0]
#                 sku_id = row[1] 
#                 text_to_embed = (row[2] or "").strip()
                
#                 if not text_to_embed:
#                     continue
                
#                 # Normalize Vietnamese text if available
#                 try:
#                     normalized_text = vn_norm(text_to_embed)
#                 except:
#                     normalized_text = text_to_embed.lower().strip()
                
#                 # Generate CLIP text embedding (512 dimensions)
#                 with torch.no_grad():
#                     text_tokens = tokenizer([normalized_text]).to(device)
#                     text_features = model.encode_text(text_tokens)
#                     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#                     embedding = text_features[0].cpu().numpy()
                
#                 # Ensure 512 dimensions for text_vec column
#                 if len(embedding) != 512:
#                     logger.warning("CLIP embedding has %d dimensions, expected 512", len(embedding))
#                     continue
                
#                 vector_list = embedding.tolist()
                
#                 # Update database with CLIP vector
#                 exec_sql("""
#                     UPDATE sku_texts 
#                     SET text_vec = %s::vector
#                     WHERE id = %s
#                 """, (vector_list, row_id))
                
#                 processed += 1
#                 logger.info("Generated CLIP text vector for sku_texts.id=%s (sku_id=%s)", row_id, sku_id)
                
#             except Exception as e:
#                 failed += 1
#                 logger.error("Failed to generate CLIP text vector for sku_texts.id=%s: %s", row[0], e)

#         flash(f"CLIP text vector backfill: {processed} thành công, {failed} lỗi", "success" if failed == 0 else "warning")
        
#         # Auto refresh corpus after backfill
#         try:
#             refresh_corpus_internal()
#             flash("Đã refresh search corpus", "info")
#         except Exception as e:
#             logger.error("Failed to refresh corpus after CLIP backfill: %s", e)
            
#         return redirect(url_for("skus.skus"))
        
#     except Exception as e:
#         logger.exception("CLIP text vector backfill failed")
#         flash(f"Lỗi backfill CLIP: {e}", "danger")
#         return redirect(url_for("skus.skus"))

# @bp.post("/admin/image-vector-backfill-clip")
# def image_vec_backfill_clip():
#     """Backfill image embeddings using CLIP image encoder"""
#     try:
#         limit = int(request.form.get("limit") or request.args.get("limit") or 50)
        
#         # Find images without vectors
#         rows = q("""
#             SELECT id, sku_id, image_path 
#             FROM sku_images 
#             WHERE image_vector IS NULL
#             ORDER BY id 
#             LIMIT %s
#         """, (limit,))
        
#         if not rows:
#             flash("Không có ảnh nào cần backfill vector", "info")
#             return redirect(url_for("skus.skus"))

#         model, preprocess, tokenizer = get_clip_model()
#         device = next(model.parameters()).device
#         processed = 0
#         failed = 0
#         upload_folder = os.getenv("UPLOAD_FOLDER", "uploads")

#         for row in rows:
#             try:
#                 # Access tuple by index: (id, sku_id, image_path)
#                 row_id = row[0]
#                 sku_id = row[1]
#                 image_path = row[2]
                
#                 if not image_path:
#                     continue
                
#                 # Build full path
#                 full_path = os.path.join(upload_folder, image_path.lstrip('/'))
#                 if not os.path.exists(full_path):
#                     logger.warning("Image not found: %s", full_path)
#                     continue
                
#                 # Load and preprocess image
#                 pil_image = Image.open(full_path).convert('RGB')
#                 image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                
#                 # Generate CLIP image embedding
#                 with torch.no_grad():
#                     image_features = model.encode_image(image_tensor)
#                     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#                     embedding = image_features[0].cpu().numpy()
                
#                 vector_list = embedding.tolist()
                
#                 # Update database with vector
#                 exec_sql("""
#                     UPDATE sku_images 
#                     SET image_vector = %s::vector
#                     WHERE id = %s
#                 """, (vector_list, row_id))
                
#                 processed += 1
#                 logger.info("Generated CLIP vector for sku_images.id=%s (sku_id=%s)", row_id, sku_id)
                
#             except Exception as e:
#                 failed += 1
#                 logger.error("Failed to generate CLIP vector for sku_images.id=%s: %s", row[0], e)

#         flash(f"CLIP image vector backfill: {processed} thành công, {failed} lỗi", "success" if failed == 0 else "warning")
        
#         return redirect(url_for("skus.skus"))
        
#     except Exception as e:
#         logger.exception("CLIP image vector backfill failed")
#         flash(f"Lỗi backfill CLIP images: {e}", "danger")
#         return redirect(url_for("skus.skus"))

@bp.post("/admin/text-vector-backfill")
def text_vec_backfill():
    """Backfill text_vec using OpenCLIP text encoder (512d)"""
    try:
        limit = int(request.form.get("limit") or request.args.get("limit") or 100)

        rows = q("""
            SELECT id, sku_id, text
            FROM sku_texts
            WHERE text_vec IS NULL
            ORDER BY id
            LIMIT %s
        """, (limit,))

        if not rows:
            flash("Không có text nào cần backfill vector", "info")
            return redirect(url_for("skus.skus"))

        model, preprocess, tokenizer = get_clip_model()
        device = next(model.parameters()).device
        processed = 0
        failed = 0

        for row in rows:
            try:
                row_id, sku_id, text = row[0], row[1], (row[2] or "").strip()
                if not text:
                    continue

                try:
                    normalized = vn_norm(text)
                except Exception:
                    normalized = text.lower().strip()

                # tokenize + encode correctly for open_clip
                tokens = tokenizer([normalized]).to(device)
                with torch.no_grad():
                    features = model.encode_text(tokens)
                    features = features / features.norm(dim=-1, keepdim=True)
                    emb = features[0].cpu().numpy()

                if emb.shape[0] != 512:
                    logger.warning("Skipping id=%s: embedding dim=%d != 512", row_id, emb.shape[0])
                    failed += 1
                    continue

                exec_sql("""
                    UPDATE sku_texts
                    SET text_vec = %s::vector
                    WHERE id = %s
                """, (emb.tolist(), row_id))

                processed += 1
                logger.info("Backfilled text_vec id=%s (sku_id=%s)", row_id, sku_id)

            except Exception as e:
                failed += 1
                logger.exception("Failed backfill for sku_texts.id=%s: %s", row[0] if row else "?", e)

        flash(f"CLIP text backfill: {processed} success, {failed} failed", "success" if failed == 0 else "warning")

        try:
            refresh_corpus_internal()
            flash("Đã refresh search corpus", "info")
        except Exception as e:
            logger.warning("Refresh corpus after backfill failed: %s", e)

        return redirect(url_for("skus.skus"))

    except Exception as e:
        logger.exception("CLIP text vector backfill failed")
        flash(f"Lỗi backfill CLIP: {e}", "danger")
        return redirect(url_for("skus.skus"))
# ...existing code...
@bp.post("/admin/refresh-corpus")
def refresh_search_corpus():  # tên function khác
    """Refresh materialized view sku_search_corpus"""
    try:
        refresh_corpus_internal()
        flash("Đã refresh search corpus thành công", "success")
        return redirect(url_for("skus.skus"))
    except Exception as e:
        logger.exception("Refresh corpus failed")
        flash(f"Lỗi refresh corpus: {e}", "danger")
        return redirect(url_for("skus.skus"))

def refresh_corpus_internal():
    """Internal function to refresh corpus (can be called from other functions)"""
    try:
        # Check if materialized view exists, create if not
        exists = q("""
            SELECT 1 FROM pg_matviews 
            WHERE matviewname = 'sku_search_corpus'
        """, fetch="one")
        
        if not exists:
            # Create materialized view if it doesn't exist
            exec_sql("""
                CREATE MATERIALIZED VIEW sku_search_corpus AS
                SELECT DISTINCT
                    s.id as sku_id,
                    COALESCE(b.name, '') || ' ' || s.name || ' ' || 
                    COALESCE(s.variant, '') || ' ' || COALESCE(s.size_text, '') || ' ' ||
                    COALESCE(st.text, '') as text_content,
                    to_tsvector('simple', 
                        COALESCE(b.name, '') || ' ' || s.name || ' ' || 
                        COALESCE(s.variant, '') || ' ' || COALESCE(s.size_text, '') || ' ' ||
                        COALESCE(st.text, '')
                    ) as search_vector,
                    si.image_path
                FROM skus s
                LEFT JOIN brands b ON b.id = s.brand_id
                LEFT JOIN sku_texts st ON st.sku_id = s.id
                LEFT JOIN sku_images si ON si.sku_id = s.id AND si.is_primary = true
                WHERE s.is_active = true
            """)
            
            # Create index
            exec_sql("""
                CREATE INDEX IF NOT EXISTS idx_sku_search_corpus_vector 
                ON sku_search_corpus USING gin(search_vector)
            """)
            
            exec_sql("""
                CREATE INDEX IF NOT EXISTS idx_sku_search_corpus_sku_id 
                ON sku_search_corpus (sku_id)
            """)
            
            logger.info("Created sku_search_corpus materialized view")
        else:
            # Refresh existing view
            exec_sql("REFRESH MATERIALIZED VIEW sku_search_corpus")
            logger.info("Refreshed sku_search_corpus materialized view")
            
    except Exception as e:
        logger.error("Error refreshing corpus: %s", e)
        raise

# API endpoints for AJAX calls
# @bp.get("/api/admin/stats")
# def admin_stats():
#     """Get statistics for admin dashboard"""
#     try:
#         stats = {}
        
#         # Count texts without vectors (use correct column name)
#         stats['texts_no_vector'] = q("""
#             SELECT COUNT(*) FROM sku_texts 
#             WHERE text_vec IS NULL
#         """, fetch="one")[0]
        
#         # Count images without vectors  
#         stats['images_no_vector'] = q("""
#             SELECT COUNT(*) FROM sku_images 
#             WHERE image_vector IS NULL
#         """, fetch="one")[0]
        
#         # Total SKUs
#         stats['total_skus'] = q("SELECT COUNT(*) FROM skus", fetch="one")[0]
#         stats['active_skus'] = q("SELECT COUNT(*) FROM skus WHERE is_active = true", fetch="one")[0]
        
#         # Search corpus status
#         corpus_exists = q("""
#             SELECT 1 FROM pg_matviews WHERE matviewname = 'sku_search_corpus'
#         """, fetch="one")
#         stats['corpus_exists'] = bool(corpus_exists)
        
#         if corpus_exists:
#             stats['corpus_entries'] = q("""
#                 SELECT COUNT(*) FROM sku_search_corpus
#             """, fetch="one")[0]
#         else:
#             stats['corpus_entries'] = 0
            
#         return jsonify(stats)
        
#     except Exception as e:
#         logger.exception("Failed to get admin stats")
#         return jsonify({"error": str(e)}), 500

@bp.post("/admin/text-vector-backfill-clip")
def text_vec_backfill_clip():
    """Backfill text embeddings using CLIP text encoder"""
    try:
        limit = int(request.form.get("limit") or request.args.get("limit") or 100)
        
        # Find sku_texts without embeddings
        rows = q("""
            SELECT id, sku_id, text
            FROM sku_texts 
            WHERE text_vec IS NULL
            ORDER BY id 
            LIMIT %s
        """, (limit,))
        
        if not rows:
            flash("Không có text nào cần backfill vector", "info")
            return redirect(url_for("skus.skus"))

        model, preprocess, tokenizer = get_clip_model()
        device = next(model.parameters()).device
        processed = 0
        failed = 0

        for row in rows:
            try:
                # Access tuple by index: (id, sku_id, text)
                row_id = row[0]
                sku_id = row[1] 
                text_to_embed = (row[2] or "").strip()
                
                if not text_to_embed:
                    continue
                
                # Normalize Vietnamese text if available
                try:
                    normalized_text = vn_norm(text_to_embed)
                except:
                    normalized_text = text_to_embed.lower().strip()
                
                # Generate CLIP text embedding (512 dimensions)
                with torch.no_grad():
                    text_tokens = tokenizer([normalized_text]).to(device)
                    text_features = model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    embedding = text_features[0].cpu().numpy()
                
                # Ensure 512 dimensions for text_vec column
                if len(embedding) != 512:
                    logger.warning("CLIP embedding has %d dimensions, expected 512", len(embedding))
                    continue
                
                vector_list = embedding.tolist()
                
                # Update database with CLIP vector
                exec_sql("""
                    UPDATE sku_texts 
                    SET text_vec = %s::vector
                    WHERE id = %s
                """, (vector_list, row_id))
                
                processed += 1
                logger.info("Generated CLIP text vector for sku_texts.id=%s (sku_id=%s)", row_id, sku_id)
                
            except Exception as e:
                failed += 1
                logger.error("Failed to generate CLIP text vector for sku_texts.id=%s: %s", row[0], e)

        flash(f"CLIP text vector backfill: {processed} thành công, {failed} lỗi", "success" if failed == 0 else "warning")
        
        # Auto refresh corpus after backfill
        try:
            refresh_corpus_internal()
            flash("Đã refresh search corpus", "info")
        except Exception as e:
            logger.error("Failed to refresh corpus after CLIP backfill: %s", e)
            
        return redirect(url_for("skus.skus"))
        
    except Exception as e:
        logger.exception("CLIP text vector backfill failed")
        flash(f"Lỗi backfill CLIP: {e}", "danger")
        return redirect(url_for("skus.skus"))

@bp.post("/admin/image-vector-backfill-clip")
def image_vec_backfill_clip():
    """Backfill image embeddings using CLIP image encoder"""
    try:
        limit = int(request.form.get("limit") or request.args.get("limit") or 50)
        
        # Find images without vectors
        rows = q("""
            SELECT id, sku_id, image_path 
            FROM sku_images 
            WHERE image_vector IS NULL
            ORDER BY id 
            LIMIT %s
        """, (limit,))
        
        if not rows:
            flash("Không có ảnh nào cần backfill vector", "info")
            return redirect(url_for("skus.skus"))

        model, preprocess, tokenizer = get_clip_model()
        device = next(model.parameters()).device
        processed = 0
        failed = 0
        upload_folder = os.getenv("UPLOAD_FOLDER", "uploads")

        for row in rows:
            try:
                # Access tuple by index: (id, sku_id, image_path)
                row_id = row[0]
                sku_id = row[1]
                image_path = row[2]
                
                if not image_path:
                    continue
                
                # Build full path
                full_path = os.path.join(upload_folder, image_path.lstrip('/'))
                if not os.path.exists(full_path):
                    logger.warning("Image not found: %s", full_path)
                    continue
                
                # Load and preprocess image
                pil_image = Image.open(full_path).convert('RGB')
                image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                
                # Generate CLIP image embedding
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embedding = image_features[0].cpu().numpy()
                
                vector_list = embedding.tolist()
                
                # Update database with vector
                exec_sql("""
                    UPDATE sku_images 
                    SET image_vector = %s::vector
                    WHERE id = %s
                """, (vector_list, row_id))
                
                processed += 1
                logger.info("Generated CLIP vector for sku_images.id=%s (sku_id=%s)", row_id, sku_id)
                
            except Exception as e:
                failed += 1
                logger.error("Failed to generate CLIP vector for sku_images.id=%s: %s", row[0], e)

        flash(f"CLIP image vector backfill: {processed} thành công, {failed} lỗi", "success" if failed == 0 else "warning")
        
        return redirect(url_for("skus.skus"))
        
    except Exception as e:
        logger.exception("CLIP image vector backfill failed")
        flash(f"Lỗi backfill CLIP images: {e}", "danger")
        return redirect(url_for("skus.skus"))

@bp.post("/admin/text-vector-backfill")
def text_vec_backfill_alt():
    """Backfill text embeddings for sku_texts that don't have vectors"""
    try:
        limit = int(request.form.get("limit") or request.args.get("limit") or 100)
        
        # Find sku_texts without embeddings (use correct column names)
        rows = q("""
            SELECT id, sku_id, text
            FROM sku_texts 
            WHERE text_vec IS NULL
            ORDER BY id 
            LIMIT %s
        """, (limit,))
        
        if not rows:
            flash("Không có text nào cần backfill vector", "info")
            return redirect(url_for("skus.skus"))

        model = get_embed_model()
        processed = 0
        failed = 0

        for row in rows:
            try:
                # Access tuple by index: (id, sku_id, text)
                row_id = row[0]
                sku_id = row[1]
                text_to_embed = (row[2] or "").strip()
                
                if not text_to_embed:
                    continue
                
                # Normalize text using vn_norm
                try:
                    normalized_text = vn_norm(text_to_embed)
                except ImportError:
                    normalized_text = text_to_embed.lower().strip()
                
                # Generate embedding
                embedding = model.encode(normalized_text, normalize_embeddings=True)
                vector_list = embedding.tolist()
                
                # Update database with vector (use correct column name: text_vec)
                exec_sql("""
                    UPDATE sku_texts 
                    SET text_vec = %s::vector
                    WHERE id = %s
                """, (vector_list, row_id))
                
                processed += 1
                logger.info("Generated vector for sku_texts.id=%s (sku_id=%s)", row_id, sku_id)
                
            except Exception as e:
                failed += 1
                logger.error("Failed to generate vector for sku_texts.id=%s: %s", row[0], e)

        flash(f"Text vector backfill: {processed} thành công, {failed} lỗi", "success" if failed == 0 else "warning")
        
        # Auto refresh corpus after backfill
        try:
            refresh_corpus_internal()
            flash("Đã refresh search corpus", "info")
        except Exception as e:
            logger.error("Failed to refresh corpus after backfill: %s", e)
            
        return redirect(url_for("skus.skus"))
        
    except Exception as e:
        logger.exception("Text vector backfill failed")
        flash(f"Lỗi backfill: {e}", "danger")
        return redirect(url_for("skus.skus"))

# @bp.post("/admin/refresh-corpus")
# def refresh_search_corpus():  # tên function khác
#     """Refresh materialized view """
#     try:
#         refresh_corpus_internal()
#         flash("Đã refresh search corpus thành công", "success")
#         return redirect(url_for("skus.skus"))
#     except Exception as e:
#         logger.exception("Refresh corpus failed")
#         flash(f"Lỗi refresh corpus: {e}", "danger")
#         return redirect(url_for("skus.skus"))

def refresh_corpus_internal():
    """Internal function to refresh corpus (can be called from other functions)"""
    try:
        # Check if materialized view exists, create if not
        exists = q("""
            SELECT 1 FROM pg_matviews 
            WHERE matviewname = 'sku_search_corpus'
        """, fetch="one")
        
        if not exists:
            # Create materialized view if it doesn't exist
            exec_sql("""
                CREATE MATERIALIZED VIEW sku_search_corpus AS
                SELECT DISTINCT
                    s.id as sku_id,
                    COALESCE(b.name, '') || ' ' || s.name || ' ' || 
                    COALESCE(s.variant, '') || ' ' || COALESCE(s.size_text, '') || ' ' ||
                    COALESCE(st.text, '') as text_content,
                    to_tsvector('simple', 
                        COALESCE(b.name, '') || ' ' || s.name || ' ' || 
                        COALESCE(s.variant, '') || ' ' || COALESCE(s.size_text, '') || ' ' ||
                        COALESCE(st.text, '')
                    ) as search_vector,
                    si.image_path
                FROM skus s
                LEFT JOIN brands b ON b.id = s.brand_id
                LEFT JOIN sku_texts st ON st.sku_id = s.id
                LEFT JOIN sku_images si ON si.sku_id = s.id AND si.is_primary = true
                WHERE s.is_active = true
            """)
            
            # Create index
            exec_sql("""
                CREATE INDEX IF NOT EXISTS idx_sku_search_corpus_vector 
                ON sku_search_corpus USING gin(search_vector)
            """)
            
            exec_sql("""
                CREATE INDEX IF NOT EXISTS idx_sku_search_corpus_sku_id 
                ON sku_search_corpus (sku_id)
            """)
            
            logger.info("Created sku_search_corpus materialized view")
        else:
            # Refresh existing view
            exec_sql("REFRESH MATERIALIZED VIEW sku_search_corpus")
            logger.info("Refreshed sku_search_corpus materialized view")
            
    except Exception as e:
        logger.error("Error refreshing corpus: %s", e)
        raise

# API endpoints for AJAX calls
@bp.get("/api/admin/stats")
def admin_stats():
    """Get statistics for admin dashboard"""
    try:
        stats = {}
        
        # Count texts without vectors (use correct column name)
        stats['texts_no_vector'] = q("""
            SELECT COUNT(*) FROM sku_texts 
            WHERE text_vec IS NULL
        """, fetch="one")[0]
        
        # Count images without vectors  
        stats['images_no_vector'] = q("""
            SELECT COUNT(*) FROM sku_images 
            WHERE image_vector IS NULL
        """, fetch="one")[0]
        
        # Total SKUs
        stats['total_skus'] = q("SELECT COUNT(*) FROM skus", fetch="one")[0]
        stats['active_skus'] = q("SELECT COUNT(*) FROM skus WHERE is_active = true", fetch="one")[0]
        
        # Search corpus status
        corpus_exists = q("""
            SELECT 1 FROM pg_matviews WHERE matviewname = 'sku_search_corpus'
        """, fetch="one")
        stats['corpus_exists'] = bool(corpus_exists)
        
        if corpus_exists:
            stats['corpus_entries'] = q("""
                SELECT COUNT(*) FROM sku_search_corpus
            """, fetch="one")[0]
        else:
            stats['corpus_entries'] = 0
            
        return jsonify(stats)
        
    except Exception as e:
        logger.exception("Failed to get admin stats")
        return jsonify({"error": str(e)}), 500

