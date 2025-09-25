-- ============================
-- schema.sql  (PostgreSQL + pgvector)
-- Quản lý sản phẩm & logging để fine-tune sau
-- Vector dim = 512 (phù hợp ViT-B/32)
-- ============================

-- 0) Extensions
CREATE EXTENSION IF NOT EXISTS vector;     -- cần cho pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- tiện cho ILIKE/ tìm text
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- nếu muốn UUID

-- 1) Bảng danh mục & thương hiệu
CREATE TABLE IF NOT EXISTS brands (
  id        BIGSERIAL PRIMARY KEY,
  name      TEXT NOT NULL UNIQUE,
  synonyms  TEXT[] DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS categories (
  id         BIGSERIAL PRIMARY KEY,
  parent_id  BIGINT REFERENCES categories(id) ON DELETE SET NULL,
  name       TEXT NOT NULL,
  slug       TEXT UNIQUE
);

-- 2) Sản phẩm (SKU)
CREATE TABLE IF NOT EXISTS skus (
  id           BIGSERIAL PRIMARY KEY,
  brand_id     BIGINT REFERENCES brands(id),
  category_id  BIGINT REFERENCES categories(id),
  name         TEXT NOT NULL,          -- "Mì Hảo Hảo"
  variant      TEXT,                   -- "Tôm chua cay"
  size_text    TEXT,                   -- "75g" / "330ml"
  barcode      TEXT UNIQUE,            -- có thể NULL, UNIQUE đảm bảo không trùng
  attrs        JSONB DEFAULT '{}'::jsonb,
  is_active    BOOLEAN DEFAULT TRUE,
  created_at   timestamptz DEFAULT now(),
  updated_at   timestamptz DEFAULT now(),
  CONSTRAINT skus_brand_name_var_size_uq
    UNIQUE (brand_id, name, variant, size_text)
);

-- Index phục vụ quản trị
CREATE INDEX IF NOT EXISTS idx_skus_brand     ON skus(brand_id);
CREATE INDEX IF NOT EXISTS idx_skus_category  ON skus(category_id);
CREATE INDEX IF NOT EXISTS idx_skus_active    ON skus(is_active);

-- Trigger cập nhật updated_at
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_skus_updated_at ON skus;
CREATE TRIGGER trg_skus_updated_at
BEFORE UPDATE ON skus
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- 3) Text chuẩn hoá cho SKU (phục vụ search & embedding sau)
-- Lưu "chuẩn hoá" (không dấu, chuẩn cách viết) ở cột text; vector để NULL lúc đầu
CREATE TABLE IF NOT EXISTS sku_texts (
  id        BIGSERIAL PRIMARY KEY,
  sku_id    BIGINT REFERENCES skus(id) ON DELETE CASCADE,
  text      TEXT NOT NULL,            -- ví dụ: "mi hao hao tom chua cay 75g"
  text_vec  vector(512),              -- backfill sau khi encode
  CONSTRAINT sku_texts_uq UNIQUE (sku_id, text)
);

-- Tăng tốc ILIKE khi chưa có vector search
CREATE INDEX IF NOT EXISTS idx_sku_texts_trgm ON sku_texts USING gin (text gin_trgm_ops);

-- 4) Ảnh SKU (catalog/thực tế)
CREATE TABLE IF NOT EXISTS sku_images (
  id          BIGSERIAL PRIMARY KEY,
  sku_id      BIGINT REFERENCES skus(id) ON DELETE CASCADE,
  image_path  TEXT NOT NULL,          -- ví dụ: "/app/uploads/haohao_75g_1.jpg"
  image_vec   vector(512),            -- backfill sau
  ocr_text    TEXT,
  is_primary  BOOLEAN DEFAULT FALSE,
  created_at  timestamptz DEFAULT now()
);

-- Mỗi SKU chỉ một ảnh chính
CREATE UNIQUE INDEX IF NOT EXISTS uq_one_primary_image_per_sku
  ON sku_images(sku_id)
  WHERE is_primary IS TRUE;

-- 5) Chỉ mục ANN cho vector (bật khi đã backfill)
-- YÊU CẦU pgvector >= 0.5 để dùng HNSW
CREATE INDEX IF NOT EXISTS idx_sku_texts_vec_hnsw
  ON sku_texts USING hnsw (text_vec vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_sku_images_vec_hnsw
  ON sku_images USING hnsw (image_vec vector_cosine_ops);

-- Nếu bản pgvector cũ, dùng IVFFLAT (và nhớ ANALYZE):
-- CREATE INDEX idx_sku_texts_vec_ivf  ON sku_texts  USING ivfflat (text_vec  vector_cosine_ops) WITH (lists = 200);
-- CREATE INDEX idx_sku_images_vec_ivf ON sku_images USING ivfflat (image_vec vector_cosine_ops) WITH (lists = 200);
-- ANALYZE sku_texts; ANALYZE sku_images;

-- 6) Bảng log truy vấn để thu thập dữ liệu fine-tune
CREATE TABLE IF NOT EXISTS queries (
  id               BIGSERIAL PRIMARY KEY,
  type             TEXT NOT NULL CHECK (type IN ('text','image')),
  raw_text         TEXT,                 -- chuỗi người dùng gõ (nếu có)
  normalized_text  TEXT,                 -- sau khi vn_norm
  image_path       TEXT,                 -- ảnh query lưu tạm (nếu có)
  created_at       timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_candidates (
  id         BIGSERIAL PRIMARY KEY,
  query_id   BIGINT REFERENCES queries(id) ON DELETE CASCADE,
  sku_id     BIGINT REFERENCES skus(id)    ON DELETE CASCADE,
  rank       INT,
  score      REAL,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_labels (
  id             BIGSERIAL PRIMARY KEY,
  query_id       BIGINT REFERENCES queries(id) ON DELETE CASCADE,
  chosen_sku_id  BIGINT REFERENCES skus(id)    ON DELETE CASCADE,
  is_correct     BOOLEAN DEFAULT TRUE,
  created_at     timestamptz DEFAULT now()
);

-- 7) View tiện cho quản trị/tìm text (khi chưa muốn vector search)
CREATE OR REPLACE VIEW sku_search_view AS
SELECT
  s.id AS sku_id,
  COALESCE(b.name, '') AS brand,
  s.name,
  s.variant,
  s.size_text,
  s.barcode,
  st.text AS normalized_text,
  (SELECT image_path FROM sku_images si WHERE si.sku_id = s.id AND si.is_primary IS TRUE LIMIT 1) AS cover_path,
  s.is_active,
  s.updated_at
FROM skus s
LEFT JOIN brands b ON b.id = s.brand_id
LEFT JOIN sku_texts st ON st.sku_id = s.id;

-- 8) Quyền/ANALYZE (tuỳ môi trường)
-- ANALYZE;  -- giúp planner tối ưu truy vấn

-- 1) Bảng caption sinh ra cho SKU
CREATE TABLE IF NOT EXISTS sku_captions (
  id              BIGSERIAL PRIMARY KEY,
  sku_id          BIGINT NOT NULL REFERENCES skus(id) ON DELETE CASCADE,
  image_path      TEXT,                         -- ảnh dùng để caption (thường là ảnh chính)
  lang            TEXT NOT NULL DEFAULT 'vi',   -- 'vi' / 'en' ...
  style           TEXT NOT NULL,                -- 'neutral','search','seo','alt','marketing'
  caption_text    TEXT NOT NULL,
  model_name      TEXT NOT NULL,                -- 'gemini-1.5-flash'...
  prompt_version  TEXT NOT NULL,                -- ví dụ 'v1.0'
  clip_score      REAL,                         -- điểm CLIP image-text (0..1)
  cov_brand       BOOLEAN,                      -- caption có chứa brand?
  cov_variant     BOOLEAN,                      -- có chứa variant?
  cov_size        BOOLEAN,                      -- có chứa size_text?
  needs_review    BOOLEAN DEFAULT FALSE,        -- gắn cờ cần duyệt tay
  created_at      timestamptz DEFAULT now(),
  CONSTRAINT uq_caption_variant UNIQUE (sku_id, image_path, lang, style, model_name, prompt_version)
);

-- 2) Bảng nhãn đánh giá thủ công (để fine-tune)
CREATE TABLE IF NOT EXISTS caption_labels (
  id            BIGSERIAL PRIMARY KEY,
  caption_id    BIGINT NOT NULL REFERENCES sku_captions(id) ON DELETE CASCADE,
  is_acceptable BOOLEAN NOT NULL,     -- chấp nhận làm ground-truth?
  corrected_text TEXT,                -- nếu sửa tay
  notes         TEXT,
  created_at    timestamptz DEFAULT now()
);

-- View lấy caption “mới nhất & đã được chấp nhận/sửa”
CREATE OR REPLACE VIEW caption_latest AS
SELECT sc.*
FROM sku_captions sc
LEFT JOIN LATERAL (
  SELECT cl.is_acceptable, cl.corrected_text, cl.created_at AS label_at
  FROM caption_labels cl
  WHERE cl.caption_id = sc.id
  ORDER BY cl.created_at DESC
  LIMIT 1
) lab ON true
WHERE COALESCE(lab.is_acceptable, TRUE) = TRUE;

CREATE EXTENSION IF NOT EXISTS unaccent;

-- Hàm chuẩn hoá đơn giản
CREATE OR REPLACE FUNCTION vn_norm_simple(s text) RETURNS text AS $$
  SELECT trim(regexp_replace(lower(unaccent(coalesce(s,''))),
                             '[^a-z0-9 \-\.x/]+',' ','g'));
$$ LANGUAGE sql IMMUTABLE;

-- Thêm dòng thiếu trong sku_texts từ skus
INSERT INTO sku_texts (sku_id, text)
SELECT s.id,
       trim(regexp_replace(vn_norm_simple(
              concat_ws(' ', b.name, s.name, s.variant, s.size_text)
            ), '\s+', ' ', 'g')) AS normalized_text
FROM skus s
LEFT JOIN brands b ON b.id = s.brand_id
WHERE s.is_active IS TRUE
  AND NOT EXISTS (
    SELECT 1 FROM sku_texts st
    WHERE st.sku_id = s.id
          AND st.text = trim(regexp_replace(vn_norm_simple(
                               concat_ws(' ', b.name, s.name, s.variant, s.size_text)
                             ), '\s+', ' ', 'g'))
  );
 
 
 CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE OR REPLACE FUNCTION vn_norm_simple(s text) RETURNS text AS $$
  SELECT trim(regexp_replace(lower(unaccent(coalesce(s,''))),
                             '[^a-z0-9 \-\.x/]+',' ','g'));
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION upsert_sku_text_from_caption() RETURNS trigger AS $$
DECLARE norm text;
BEGIN
  IF NEW.style = 'search' AND NEW.caption_text IS NOT NULL THEN
    norm := trim(regexp_replace(vn_norm_simple(NEW.caption_text), '\s+', ' ', 'g'));
    IF norm <> '' THEN
      INSERT INTO sku_texts (sku_id, text)
      VALUES (NEW.sku_id, norm)
      ON CONFLICT (sku_id, text) DO NOTHING;
    END IF;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_caption_to_text ON sku_captions;
CREATE TRIGGER trg_caption_to_text
AFTER INSERT OR UPDATE OF caption_text, style ON sku_captions
FOR EACH ROW EXECUTE FUNCTION upsert_sku_text_from_caption();


-- ============================
-- PATCH for AI captioning + multi-criteria search
-- Compatible with your current schema
-- ============================

-- 0) Extensions (safe)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- 1) Chuẩn hoá chuỗi (re-use, tạo nếu thiếu)
CREATE OR REPLACE FUNCTION vn_norm_simple(s text) RETURNS text AS $$
  SELECT trim(regexp_replace(lower(unaccent(coalesce(s,''))),
                             '[^a-z0-9 \-\.x/]+',' ','g'));
$$ LANGUAGE sql IMMUTABLE;

-- 2) NÂNG CẤP VECTOR DIMS
-- Giữ cột cũ (vector(512)) để không vỡ dữ liệu. Thêm cột mới đúng kích thước:
--  - text embedding: 1024 (bge-m3 / e5-large)
--  - image embedding: 768 (CLIP ViT-L/14, SigLIP)
ALTER TABLE sku_texts  ADD COLUMN IF NOT EXISTS text_vec_1k  vector(1024);
ALTER TABLE sku_images ADD COLUMN IF NOT EXISTS image_vec_768 vector(768);

-- 3) FACETS & MÀU SẮC
-- Facets tổng hợp ở SKU (từ text + ảnh), chi tiết ở từng ảnh
ALTER TABLE skus
  ADD COLUMN IF NOT EXISTS facets jsonb DEFAULT '{}'::jsonb;

ALTER TABLE sku_images
  ADD COLUMN IF NOT EXISTS colors       text[] DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS color_scores jsonb  DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS detected_facets jsonb DEFAULT '{}'::jsonb;

-- 4) BẢNG GỢI Ý AI (brand/category/facets) THEO ẢNH + MODEL
CREATE TABLE IF NOT EXISTS sku_ai_suggestions (
  id            BIGSERIAL PRIMARY KEY,
  sku_id        BIGINT NOT NULL REFERENCES skus(id) ON DELETE CASCADE,
  image_id      BIGINT REFERENCES sku_images(id) ON DELETE SET NULL,
  model_name    TEXT NOT NULL,         -- "qwen2-vl-7b", "clip-vit-l/14", ...
  prompt_ver    TEXT NOT NULL DEFAULT 'v1.0',
  brand_suggest TEXT,                   -- "coca_cola" (chuẩn hoá theo dict_brands.id nếu có)
  category_suggest TEXT,                -- "fried_pasta_snack" (chuẩn hoá theo dict_categories.id)
  facets_suggest  jsonb DEFAULT '{}'::jsonb,
  confidences     jsonb DEFAULT '{}'::jsonb,  -- {"brand":0.92,"category":0.81,"shape_triangle":0.77,...}
  ocr_text_extracted TEXT,
  created_at    timestamptz DEFAULT now(),
  CONSTRAINT uq_ai_suggest UNIQUE (sku_id, image_id, model_name, prompt_ver)
);

-- 5) INDEXES CHO VECTOR MỚI & JSON
-- Vector ANN
CREATE INDEX IF NOT EXISTS idx_sku_texts_vec1k_hnsw
  ON sku_texts USING hnsw (text_vec_1k vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_sku_images_vec768_hnsw
  ON sku_images USING hnsw (image_vec_768 vector_cosine_ops);

-- JSON facets/màu
CREATE INDEX IF NOT EXISTS idx_skus_facets_gin        ON skus       USING gin (facets);
CREATE INDEX IF NOT EXISTS idx_sku_images_facets_gin  ON sku_images USING gin (detected_facets);
CREATE INDEX IF NOT EXISTS idx_sku_images_colors_gin  ON sku_images USING gin (colors);

-- 6) CAPTION PIPELINE (đã có): bổ sung cờ & tóm tắt để search
ALTER TABLE sku_captions
  ADD COLUMN IF NOT EXISTS summary_text TEXT,            -- tuỳ chọn: tóm tắt 1-2 câu cho listing/search
  ADD COLUMN IF NOT EXISTS is_ground_truth BOOLEAN DEFAULT FALSE; -- set TRUE khi caption_labels chấp nhận

-- 7) VIEW CAPTION MỚI NHẤT ĐÃ CHẤP NHẬN (giữ nguyên nếu bạn đã có)
DROP VIEW IF EXISTS caption_latest;
CREATE OR REPLACE VIEW caption_latest AS
SELECT sc.*
FROM sku_captions sc
LEFT JOIN LATERAL (
  SELECT cl.is_acceptable, cl.corrected_text, cl.created_at AS label_at
  FROM caption_labels cl
  WHERE cl.caption_id = sc.id
  ORDER BY cl.created_at DESC
  LIMIT 1
) lab ON true
WHERE COALESCE(lab.is_acceptable, TRUE) = TRUE;

-- 8) CORPUS TÌM KIẾM HỢP NHẤT (materialized view)
-- Gom: brand + name + variant + size + caption đã duyệt + OCR (ảnh chính) + synonyms brand
-- Có tsvector cho BM25, đồng thời giữ normalized_text để ILIKE / debug
DROP MATERIALIZED VIEW IF EXISTS sku_search_corpus;
CREATE MATERIALIZED VIEW sku_search_corpus AS
WITH primary_img AS (
  SELECT si.sku_id, si.image_path, si.ocr_text
  FROM sku_images si
  WHERE si.is_primary IS TRUE
),
accepted_caption AS (
  SELECT DISTINCT ON (sc.sku_id) sc.sku_id,
         COALESCE(cl.corrected_text, sc.caption_text) AS caption_ok
  FROM sku_captions sc
  LEFT JOIN caption_labels cl ON cl.caption_id = sc.id
  WHERE COALESCE(cl.is_acceptable, TRUE) = TRUE
  ORDER BY sc.sku_id, sc.created_at DESC
)
SELECT
  s.id AS sku_id,
  COALESCE(b.name,'') AS brand,
  s.name, s.variant, s.size_text, s.barcode,
  pi.image_path AS cover_path,
  pi.ocr_text   AS ocr_text,
  ac.caption_ok AS caption_text,
  -- text gộp
  trim(regexp_replace(
    vn_norm_simple(concat_ws(' ',
      b.name,
      array_to_string(b.synonyms, ' '),
      s.name, s.variant, s.size_text,
      ac.caption_ok,
      pi.ocr_text
    )), '\s+', ' ', 'g')) AS normalized_text,
  -- tsvector (simple)
  to_tsvector('simple',
    coalesce(b.name,'') || ' ' ||
    coalesce(array_to_string(b.synonyms,' '),'') || ' ' ||
    coalesce(s.name,'') || ' ' || coalesce(s.variant,'') || ' ' || coalesce(s.size_text,'') || ' ' ||
    coalesce(ac.caption_ok,'') || ' ' || coalesce(pi.ocr_text,'')
  ) AS tsv
FROM skus s
LEFT JOIN brands b ON b.id = s.brand_id
LEFT JOIN primary_img pi ON pi.sku_id = s.id
LEFT JOIN accepted_caption ac ON ac.sku_id = s.id
WHERE s.is_active IS TRUE;

-- Chỉ mục cho corpus
CREATE INDEX IF NOT EXISTS idx_sku_corpus_sku           ON sku_search_corpus(sku_id);
CREATE INDEX IF NOT EXISTS idx_sku_corpus_tsv           ON sku_search_corpus USING gin (tsv);
CREATE INDEX IF NOT EXISTS idx_sku_corpus_norm_trgm     ON sku_search_corpus USING gin (normalized_text gin_trgm_ops);

-- 9) TRIGGER REFRESH CORPUS khi dữ liệu liên quan thay đổi
-- Để đơn giản: tạo hàm mark & refresh theo lịch (cron) hoặc refresh nhỏ khi ghi nhận thay đổi quan trọng.
-- Ở đây làm hàm tiện dụng để bạn gọi thủ công/cron:
CREATE OR REPLACE FUNCTION refresh_sku_search_corpus() RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY sku_search_corpus;
EXCEPTION WHEN undefined_table THEN
  -- nếu chưa tạo CONCURRENTLY lần đầu, fallback
  REFRESH MATERIALIZED VIEW sku_search_corpus;
END;
$$ LANGUAGE plpgsql;

-- 10) ĐẨY CAPTION 'search' VÀO sku_texts (bạn đã có trigger, giữ nguyên).
-- Bổ sung: khi caption_labels chấp nhận, tự set is_ground_truth.
CREATE OR REPLACE FUNCTION set_caption_ground_truth() RETURNS trigger AS $$
BEGIN
  IF NEW.is_acceptable IS TRUE THEN
    UPDATE sku_captions
    SET is_ground_truth = TRUE
    WHERE id = NEW.caption_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_caption_labels_gt ON caption_labels;
CREATE TRIGGER trg_caption_labels_gt
AFTER INSERT OR UPDATE OF is_acceptable ON caption_labels
FOR EACH ROW EXECUTE FUNCTION set_caption_ground_truth();

-- 11) TEXT SEARCH TSVECTOR TRÊN sku_texts (tuỳ chọn, nếu muốn BM25 trực tiếp trên sku_texts)
ALTER TABLE sku_texts ADD COLUMN IF NOT EXISTS tsv tsvector;
CREATE INDEX IF NOT EXISTS idx_sku_texts_tsv ON sku_texts USING gin(tsv);

CREATE OR REPLACE FUNCTION sku_texts_tsv_update() RETURNS trigger AS $$
BEGIN
  NEW.tsv := to_tsvector('simple', coalesce(NEW.text,''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sku_texts_tsv ON sku_texts;
CREATE TRIGGER trg_sku_texts_tsv
BEFORE INSERT OR UPDATE OF text ON sku_texts
FOR EACH ROW EXECUTE FUNCTION sku_texts_tsv_update();

-- 12) CLEANUP đường dẫn ảnh (bạn đã có câu UPDATE; để lại làm script riêng)
-- UPDATE sku_images SET image_path = REPLACE(image_path, '/app/uploads/', '') WHERE image_path LIKE '/app/uploads/%';
-- JSON attrs ở SKU (lọc facets/màu...): 

CREATE INDEX IF NOT EXISTS idx_skus_attrs_gin ON skus USING gin (attrs);

-- Join nhanh sku_texts ↔ skus:
CREATE INDEX IF NOT EXISTS idx_sku_texts_sku   ON sku_texts(sku_id);

-- Join nhanh sku_images ↔ skus (ngoài ảnh primary):
CREATE INDEX IF NOT EXISTS idx_sku_images_sku  ON sku_images(sku_id);

-- Log fine-tune: tăng tốc truy vấn & thống kê nhãn
CREATE INDEX IF NOT EXISTS idx_qc_query   ON query_candidates(query_id);
CREATE INDEX IF NOT EXISTS idx_qc_sku     ON query_candidates(sku_id);
CREATE INDEX IF NOT EXISTS idx_ql_query   ON query_labels(query_id);
CREATE INDEX IF NOT EXISTS idx_ql_sku     ON query_labels(chosen_sku_id);

-- Text vector của truy vấn (512 = OpenCLIP; 1024 nếu bạn dùng bge/e5):
ALTER TABLE queries ADD COLUMN IF NOT EXISTS query_text_vec vector(512);
-- (tuỳ chọn) vector ảnh của truy vấn nếu người dùng upload ảnh:
ALTER TABLE queries ADD COLUMN IF NOT EXISTS query_image_vec vector(512);

-- Chỉ mục ANN cho hai cột trên:
CREATE INDEX IF NOT EXISTS idx_queries_text_vec_hnsw  ON queries USING hnsw (query_text_vec vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_queries_image_vec_hnsw ON queries USING hnsw (query_image_vec vector_cosine_ops);

-- XÓA MV cũ nếu cần tạo lại
DROP MATERIALIZED VIEW IF EXISTS sku_search_corpus;

-- TẠO LẠI MV (giữ nội dung SELECT của bạn — mình rút gọn comment)
CREATE MATERIALIZED VIEW sku_search_corpus AS
WITH primary_img AS (
  SELECT si.sku_id, si.image_path, si.ocr_text
  FROM sku_images si
  WHERE si.is_primary IS TRUE
),
accepted_caption AS (
  SELECT DISTINCT ON (sc.sku_id) sc.sku_id,
         COALESCE(cl.corrected_text, sc.caption_text) AS caption_ok
  FROM sku_captions sc
  LEFT JOIN caption_labels cl ON cl.caption_id = sc.id
  WHERE COALESCE(cl.is_acceptable, TRUE) = TRUE
  ORDER BY sc.sku_id, sc.created_at DESC
)
SELECT
  s.id AS sku_id,
  COALESCE(b.name,'') AS brand,
  s.name, s.variant, s.size_text, s.barcode,
  pi.image_path AS cover_path,
  pi.ocr_text   AS ocr_text,
  ac.caption_ok AS caption_text,

  -- normalized cho fuzzy
  trim(regexp_replace(
    vn_norm_simple(concat_ws(' ',
      b.name, array_to_string(b.synonyms,' '),
      s.name, s.variant, s.size_text,
      ac.caption_ok, pi.ocr_text
    )), '\s+', ' ', 'g')) AS normalized_text,

  -- TSV có trọng số
  setweight(to_tsvector('simple', coalesce(b.name,'') || ' ' || coalesce(s.name,'')), 'A') ||
  setweight(to_tsvector('simple', coalesce(s.variant,'') || ' ' || coalesce(s.size_text,'')), 'B') ||
  setweight(to_tsvector('simple', coalesce(ac.caption_ok,'')), 'C') ||
  setweight(to_tsvector('simple', coalesce(pi.ocr_text,'')), 'D') AS tsv_weighted
FROM skus s
LEFT JOIN brands b       ON b.id = s.brand_id
LEFT JOIN primary_img pi ON pi.sku_id = s.id
LEFT JOIN accepted_caption ac ON ac.sku_id = s.id
WHERE s.is_active IS TRUE;

-- INDEXES (thêm UNIQUE để dùng REFRESH CONCURRENTLY)
CREATE UNIQUE INDEX IF NOT EXISTS uq_sku_corpus_sku   ON sku_search_corpus (sku_id);
CREATE INDEX IF NOT EXISTS idx_sku_corpus_tsv_w       ON sku_search_corpus USING gin (tsv_weighted);
CREATE INDEX IF NOT EXISTS idx_sku_corpus_norm_trgm   ON sku_search_corpus USING gin (normalized_text gin_trgm_ops);

CREATE OR REPLACE FUNCTION search_skus_hybrid(qtext TEXT, topk INT DEFAULT 20)
RETURNS TABLE (
  sku_id BIGINT,
  title TEXT,
  variant TEXT,
  size_text TEXT,
  cover_path TEXT,
  normalized_text TEXT,
  trgm_sim REAL,
  bm25 REAL,
  cos_sim REAL,
  score REAL
)
LANGUAGE sql AS $$
  SET LOCAL pg_trgm.similarity_threshold = 0.10;

  WITH p AS (
    SELECT vn_norm_simple(qtext) AS qn
  ),
  -- tách token thành nhiều hàng (không tạo mảng)
  tok AS (
    SELECT regexp_split_to_table((SELECT qn FROM p), '\s+') AS t
  ),
  base AS (
    SELECT
      s.sku_id, s.brand, s.name, s.variant, s.size_text, s.cover_path,
      s.normalized_text, s.tsv_weighted,
      st.text_vec
    FROM sku_search_corpus s
    LEFT JOIN sku_texts st ON st.sku_id = s.sku_id
    WHERE s.normalized_text % (SELECT qn FROM p)
      -- đảm bảo chứa TẤT CẢ token: dùng bool_and thay vì ILIKE ALL (array)
      AND (
        SELECT bool_and(s.normalized_text ILIKE '%'||t||'%')
        FROM tok
      )
  )
  SELECT
    b.sku_id,
    trim(concat_ws(' ', b.brand, b.name)) AS title,
    b.variant, b.size_text, b.cover_path,
    b.normalized_text,
    similarity(b.normalized_text, (SELECT qn FROM p)) AS trgm_sim,
    ts_rank(b.tsv_weighted, plainto_tsquery('simple', (SELECT qn FROM p))) AS bm25,
    NULL::REAL AS cos_sim,  -- chưa dùng vector ở bản này
    -- điểm tổng hợp khi chưa có vector: 0.6 * trigram + 0.4 * BM25
    (0.6 * similarity(b.normalized_text, (SELECT qn FROM p))) +
    (0.4 * ts_rank(b.tsv_weighted, plainto_tsquery('simple', (SELECT qn FROM p))))
    AS score
  FROM base b
  ORDER BY score DESC NULLS LAST
  LIMIT topk;
$$;
