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

