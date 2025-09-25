# routes/admin.py
import os, csv, datetime
from flask import Blueprint, send_file, request, redirect, url_for, flash, jsonify, current_app
from services.db_utils import q, exec_sql
from services.clip_service import embed_text_clip_512
from utils import vn_norm
from sentence_transformers import SentenceTransformer
import logging

bp = Blueprint("admin_bp", __name__)

# Initialize embedding model (lazy load)
_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

logger = logging.getLogger("admin")

@bp.post("/admin/search-corpus/refresh", endpoint="refresh_corpus")
def refresh_corpus():
    exec_sql("SELECT refresh_sku_search_corpus()", returning=True)
    return {"ok": True}

@bp.post("/admin/text-vector-backfill")
def text_vec_backfill():
    """Backfill text embeddings for sku_texts that don't have vectors"""
    try:
        limit = int(request.form.get("limit") or request.args.get("limit") or 100)
        
        # Find sku_texts without embeddings
        rows = q("""
            SELECT id, sku_id, text, normalized_text 
            FROM sku_texts 
            WHERE text_vector IS NULL OR text_vector = '{}'::vector
            ORDER BY id 
            LIMIT %s
        """, (limit,))
        
        if not rows:
            flash("Không có text nào cần backfill vector", "info")
            return redirect(url_for("skus_bp.skus"))

        model = get_embed_model()
        processed = 0
        failed = 0

        for row in rows:
            try:
                text_to_embed = row.normalized_text or row.text or ""
                if not text_to_embed.strip():
                    continue
                
                # Generate embedding
                embedding = model.encode(text_to_embed, normalize_embeddings=True)
                vector_list = embedding.tolist()
                
                # Update database with vector
                exec_sql("""
                    UPDATE sku_texts 
                    SET text_vector = %s::vector, updated_at = NOW()
                    WHERE id = %s
                """, (vector_list, row.id))
                
                processed += 1
                logger.info("Generated vector for sku_texts.id=%s (sku_id=%s)", row.id, row.sku_id)
                
            except Exception as e:
                failed += 1
                logger.error("Failed to generate vector for sku_texts.id=%s: %s", row.id, e)

        flash(f"Text vector backfill: {processed} thành công, {failed} lỗi", "success" if failed == 0 else "warning")
        
        # Auto refresh corpus after backfill
        try:
            refresh_corpus_internal()
            flash("Đã refresh search corpus", "info")
        except Exception as e:
            logger.error("Failed to refresh corpus after backfill: %s", e)
            
        return redirect(url_for("skus_bp.skus"))
        
    except Exception as e:
        logger.exception("Text vector backfill failed")
        flash(f"Lỗi backfill: {e}", "danger")
        return redirect(url_for("skus_bp.skus"))

@bp.post("/admin/refresh-corpus")
def refresh_search_corpus():  # tên function khác
    """Refresh materialized view sku_search_corpus"""
    try:
        refresh_corpus_internal()
        flash("Đã refresh search corpus thành công", "success")
        return redirect(url_for("skus_bp.skus"))
    except Exception as e:
        logger.exception("Refresh corpus failed")
        flash(f"Lỗi refresh corpus: {e}", "danger")
        return redirect(url_for("skus_bp.skus"))

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
        
        # Count texts without vectors
        stats['texts_no_vector'] = q("""
            SELECT COUNT(*) FROM sku_texts 
            WHERE text_vector IS NULL OR text_vector = '{}'::vector
        """, fetch="one")[0]
        
        # Count images without vectors  
        stats['images_no_vector'] = q("""
            SELECT COUNT(*) FROM sku_images 
            WHERE image_vector IS NULL OR image_vector = '{}'::vector
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

