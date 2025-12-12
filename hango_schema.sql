--
-- PostgreSQL database dump
--

\restrict QGDGFXkRDeaKHoOPsxkiia6ZUYurgHECvhyJfZc7nYNLK2xu6QmtyLe0rnHLbqb

-- Dumped from database version 16.10 (Debian 16.10-1.pgdg12+1)
-- Dumped by pg_dump version 16.10 (Debian 16.10-1.pgdg12+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: pg_database_owner
--

CREATE SCHEMA public;


ALTER SCHEMA public OWNER TO pg_database_owner;

--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: pg_database_owner
--

COMMENT ON SCHEMA public IS 'standard public schema';


--
-- Name: refresh_sku_search_corpus(); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.refresh_sku_search_corpus() RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY sku_search_corpus;
EXCEPTION WHEN undefined_table THEN
  -- nếu chưa tạo CONCURRENTLY lần đầu, fallback
  REFRESH MATERIALIZED VIEW sku_search_corpus;
END;
$$;


ALTER FUNCTION public.refresh_sku_search_corpus() OWNER TO admin;

--
-- Name: search_skus_by_image(public.vector, integer); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.search_skus_by_image(qimg_vec public.vector, topk integer DEFAULT 20) RETURNS TABLE(sku_id bigint, title text, cover_path text, cos_sim real)
    LANGUAGE sql
    AS $$
  -- (tuỳ chọn) SET LOCAL hnsw.ef_search = 80;

  SELECT
    s.id AS sku_id,
    trim(concat_ws(' ', COALESCE(b.name,''), s.name)) AS title,
    (SELECT image_path FROM sku_images si2
      WHERE si2.sku_id = s.id AND si2.is_primary IS TRUE
      LIMIT 1) AS cover_path,
    1 - (si.image_vec <=> qimg_vec) AS cos_sim
  FROM sku_images si
  JOIN skus s    ON s.id = si.sku_id
  LEFT JOIN brands b ON b.id = s.brand_id
  WHERE si.image_vec IS NOT NULL
  ORDER BY si.image_vec <=> qimg_vec
  LIMIT topk;
$$;


ALTER FUNCTION public.search_skus_by_image(qimg_vec public.vector, topk integer) OWNER TO admin;

--
-- Name: search_skus_hybrid(text, integer); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.search_skus_hybrid(qtext text, topk integer DEFAULT 20) RETURNS TABLE(sku_id bigint, title text, variant text, size_text text, cover_path text, normalized_text text, trgm_sim real, bm25 real, cos_sim real, score real)
    LANGUAGE sql
    AS $$
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


ALTER FUNCTION public.search_skus_hybrid(qtext text, topk integer) OWNER TO admin;

--
-- Name: search_skus_hybrid_vec(text, public.vector, integer); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.search_skus_hybrid_vec(qtext text, qvec public.vector, topk integer DEFAULT 20) RETURNS TABLE(sku_id bigint, title text, variant text, size_text text, cover_path text, normalized_text text, trgm_sim real, bm25 real, cos_sim real, score real)
    LANGUAGE sql
    AS $$
  SET LOCAL pg_trgm.similarity_threshold = 0.10;
  -- (tuỳ chọn) SET LOCAL hnsw.ef_search = 80;

  WITH p AS (SELECT vn_norm_simple(qtext) AS qn, qvec AS qv),
  tok AS (SELECT regexp_split_to_table((SELECT qn FROM p), '\s+') AS t),
  base AS (
    SELECT s.sku_id, s.brand, s.name, s.variant, s.size_text, s.cover_path,
           s.normalized_text, s.tsv_weighted, st.text_vec
    FROM sku_search_corpus s
    LEFT JOIN sku_texts st ON st.sku_id = s.sku_id
    WHERE s.normalized_text % (SELECT qn FROM p)
      AND (SELECT bool_and(s.normalized_text ILIKE '%'||t||'%') FROM tok)
      AND st.text_vec IS NOT NULL
  )
  SELECT
    b.sku_id,
    trim(concat_ws(' ', b.brand, b.name)) AS title,
    b.variant, b.size_text, b.cover_path,
    b.normalized_text,
    similarity(b.normalized_text, (SELECT qn FROM p)) AS trgm_sim,
    ts_rank(b.tsv_weighted, plainto_tsquery('simple', (SELECT qn FROM p))) AS bm25,
    (1 - (b.text_vec <=> (SELECT qv FROM p))) AS cos_sim,
    (0.5 * similarity(b.normalized_text, (SELECT qn FROM p))) +
    (0.3 * ts_rank(b.tsv_weighted, plainto_tsquery('simple', (SELECT qn FROM p)))) +
    (0.2 * (1 - (b.text_vec <=> (SELECT qv FROM p)))) AS score
  FROM base b
  ORDER BY score DESC NULLS LAST
  LIMIT topk;
$$;


ALTER FUNCTION public.search_skus_hybrid_vec(qtext text, qvec public.vector, topk integer) OWNER TO admin;

--
-- Name: set_caption_ground_truth(); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.set_caption_ground_truth() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  IF NEW.is_acceptable IS TRUE THEN
    UPDATE sku_captions
    SET is_ground_truth = TRUE
    WHERE id = NEW.caption_id;
  END IF;
  RETURN NEW;
END;
$$;


ALTER FUNCTION public.set_caption_ground_truth() OWNER TO admin;

--
-- Name: set_updated_at(); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.set_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END; $$;


ALTER FUNCTION public.set_updated_at() OWNER TO admin;

--
-- Name: sku_captions_tsv_update(); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.sku_captions_tsv_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.tsv := to_tsvector('simple', coalesce(NEW.caption_text,''));
  RETURN NEW;
END; $$;


ALTER FUNCTION public.sku_captions_tsv_update() OWNER TO admin;

--
-- Name: sku_texts_tsv_update(); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.sku_texts_tsv_update() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.tsv := to_tsvector('simple', coalesce(NEW.text,''));
  RETURN NEW;
END;
$$;


ALTER FUNCTION public.sku_texts_tsv_update() OWNER TO admin;

--
-- Name: upsert_sku_text_from_caption(); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.upsert_sku_text_from_caption() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
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
$$;


ALTER FUNCTION public.upsert_sku_text_from_caption() OWNER TO admin;

--
-- Name: vec_to_float4(public.vector); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.vec_to_float4(v public.vector) RETURNS real[]
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$
  SELECT ARRAY(
    SELECT x::real
    FROM regexp_split_to_table( trim(both '[]' from v::text), ',' ) AS x
  )
$$;


ALTER FUNCTION public.vec_to_float4(v public.vector) OWNER TO admin;

--
-- Name: vn_norm_simple(text); Type: FUNCTION; Schema: public; Owner: admin
--

CREATE FUNCTION public.vn_norm_simple(s text) RETURNS text
    LANGUAGE sql IMMUTABLE
    AS $$
  SELECT trim(regexp_replace(lower(unaccent(coalesce(s,''))),
                             '[^a-z0-9 \-\.x/]+',' ','g'));
$$;


ALTER FUNCTION public.vn_norm_simple(s text) OWNER TO admin;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: brands; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.brands (
    id bigint NOT NULL,
    name text NOT NULL,
    synonyms text[] DEFAULT '{}'::text[]
);


ALTER TABLE public.brands OWNER TO admin;

--
-- Name: brands_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.brands_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.brands_id_seq OWNER TO admin;

--
-- Name: brands_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.brands_id_seq OWNED BY public.brands.id;


--
-- Name: caption_labels; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.caption_labels (
    id bigint NOT NULL,
    caption_id bigint NOT NULL,
    is_acceptable boolean NOT NULL,
    corrected_text text,
    notes text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.caption_labels OWNER TO admin;

--
-- Name: caption_labels_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.caption_labels_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.caption_labels_id_seq OWNER TO admin;

--
-- Name: caption_labels_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.caption_labels_id_seq OWNED BY public.caption_labels.id;


--
-- Name: sku_captions; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.sku_captions (
    id bigint NOT NULL,
    sku_id bigint NOT NULL,
    image_path text,
    lang text DEFAULT 'vi'::text NOT NULL,
    style text NOT NULL,
    caption_text text NOT NULL,
    model_name text NOT NULL,
    prompt_version text NOT NULL,
    clip_score real,
    cov_brand boolean,
    cov_variant boolean,
    cov_size boolean,
    needs_review boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now(),
    summary_text text,
    is_ground_truth boolean DEFAULT false,
    keywords text[] DEFAULT '{}'::text[],
    colors text[] DEFAULT '{}'::text[],
    shapes text[] DEFAULT '{}'::text[],
    materials text[] DEFAULT '{}'::text[],
    packaging text[] DEFAULT '{}'::text[],
    taste text[] DEFAULT '{}'::text[],
    texture text[] DEFAULT '{}'::text[],
    brand_guess text,
    variant_guess text,
    size_guess text,
    category_guess text,
    facet_scores jsonb DEFAULT '{}'::jsonb,
    caption_vec public.vector(512),
    tsv tsvector
);


ALTER TABLE public.sku_captions OWNER TO admin;

--
-- Name: caption_latest; Type: VIEW; Schema: public; Owner: admin
--

CREATE VIEW public.caption_latest AS
 SELECT sc.id,
    sc.sku_id,
    sc.image_path,
    sc.lang,
    sc.style,
    sc.caption_text,
    sc.model_name,
    sc.prompt_version,
    sc.clip_score,
    sc.cov_brand,
    sc.cov_variant,
    sc.cov_size,
    sc.needs_review,
    sc.created_at,
    sc.summary_text,
    sc.is_ground_truth
   FROM (public.sku_captions sc
     LEFT JOIN LATERAL ( SELECT cl.is_acceptable,
            cl.corrected_text,
            cl.created_at AS label_at
           FROM public.caption_labels cl
          WHERE (cl.caption_id = sc.id)
          ORDER BY cl.created_at DESC
         LIMIT 1) lab ON (true))
  WHERE (COALESCE(lab.is_acceptable, true) = true);


ALTER VIEW public.caption_latest OWNER TO admin;

--
-- Name: categories; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.categories (
    id bigint NOT NULL,
    parent_id bigint,
    name text NOT NULL,
    slug text
);


ALTER TABLE public.categories OWNER TO admin;

--
-- Name: categories_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.categories_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.categories_id_seq OWNER TO admin;

--
-- Name: categories_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.categories_id_seq OWNED BY public.categories.id;


--
-- Name: queries; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.queries (
    id bigint NOT NULL,
    type text NOT NULL,
    raw_text text,
    normalized_text text,
    image_path text,
    created_at timestamp with time zone DEFAULT now(),
    query_text_vec public.vector(512),
    query_image_vec public.vector(512),
    CONSTRAINT queries_type_check CHECK ((type = ANY (ARRAY['text'::text, 'image'::text])))
);


ALTER TABLE public.queries OWNER TO admin;

--
-- Name: queries_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.queries_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.queries_id_seq OWNER TO admin;

--
-- Name: queries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.queries_id_seq OWNED BY public.queries.id;


--
-- Name: query_candidates; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.query_candidates (
    id bigint NOT NULL,
    query_id bigint,
    sku_id bigint,
    rank integer,
    score real,
    created_at timestamp with time zone DEFAULT now(),
    was_clicked boolean DEFAULT false,
    dwell_time real,
    add_to_cart boolean DEFAULT false,
    purchased boolean DEFAULT false
);


ALTER TABLE public.query_candidates OWNER TO admin;

--
-- Name: query_candidates_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.query_candidates_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.query_candidates_id_seq OWNER TO admin;

--
-- Name: query_candidates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.query_candidates_id_seq OWNED BY public.query_candidates.id;


--
-- Name: query_labels; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.query_labels (
    id bigint NOT NULL,
    query_id bigint,
    chosen_sku_id bigint,
    is_correct boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.query_labels OWNER TO admin;

--
-- Name: query_labels_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.query_labels_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.query_labels_id_seq OWNER TO admin;

--
-- Name: query_labels_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.query_labels_id_seq OWNED BY public.query_labels.id;


--
-- Name: sku_ai_suggestions; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.sku_ai_suggestions (
    id bigint NOT NULL,
    sku_id bigint NOT NULL,
    image_id bigint,
    model_name text NOT NULL,
    prompt_ver text DEFAULT 'v1.0'::text NOT NULL,
    brand_suggest text,
    category_suggest text,
    facets_suggest jsonb DEFAULT '{}'::jsonb,
    confidences jsonb DEFAULT '{}'::jsonb,
    ocr_text_extracted text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.sku_ai_suggestions OWNER TO admin;

--
-- Name: sku_ai_suggestions_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.sku_ai_suggestions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sku_ai_suggestions_id_seq OWNER TO admin;

--
-- Name: sku_ai_suggestions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.sku_ai_suggestions_id_seq OWNED BY public.sku_ai_suggestions.id;


--
-- Name: sku_captions_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.sku_captions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sku_captions_id_seq OWNER TO admin;

--
-- Name: sku_captions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.sku_captions_id_seq OWNED BY public.sku_captions.id;


--
-- Name: sku_images; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.sku_images (
    id bigint NOT NULL,
    sku_id bigint,
    image_path text NOT NULL,
    image_vec public.vector(512),
    ocr_text text,
    is_primary boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now(),
    image_vec_768 public.vector(768),
    colors text[] DEFAULT '{}'::text[],
    color_scores jsonb DEFAULT '{}'::jsonb,
    detected_facets jsonb DEFAULT '{}'::jsonb
);


ALTER TABLE public.sku_images OWNER TO admin;

--
-- Name: sku_images_alias; Type: VIEW; Schema: public; Owner: admin
--

CREATE VIEW public.sku_images_alias AS
 SELECT id,
    sku_id,
    image_path,
    image_vec AS image_vector,
    image_vec_768,
    ocr_text,
    is_primary,
    created_at,
    colors,
    color_scores,
    detected_facets
   FROM public.sku_images;


ALTER VIEW public.sku_images_alias OWNER TO admin;

--
-- Name: sku_images_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.sku_images_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sku_images_id_seq OWNER TO admin;

--
-- Name: sku_images_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.sku_images_id_seq OWNED BY public.sku_images.id;


--
-- Name: sku_primary_vec; Type: VIEW; Schema: public; Owner: admin
--

CREATE VIEW public.sku_primary_vec AS
 WITH prim AS (
         SELECT si.sku_id,
            si.image_vec
           FROM public.sku_images si
          WHERE ((si.is_primary IS TRUE) AND (si.image_vec IS NOT NULL))
        ), avg_img AS (
         SELECT si.sku_id,
            public.avg(si.image_vec) AS image_vec
           FROM public.sku_images si
          WHERE (si.image_vec IS NOT NULL)
          GROUP BY si.sku_id
        )
 SELECT COALESCE(prim.sku_id, avg_img.sku_id) AS sku_id,
    COALESCE(prim.image_vec, avg_img.image_vec) AS image_vec
   FROM (prim
     FULL JOIN avg_img USING (sku_id));


ALTER VIEW public.sku_primary_vec OWNER TO admin;

--
-- Name: skus; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.skus (
    id bigint NOT NULL,
    brand_id bigint,
    category_id bigint,
    name text NOT NULL,
    variant text,
    size_text text,
    barcode text,
    attrs jsonb DEFAULT '{}'::jsonb,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    image text,
    facets jsonb DEFAULT '{}'::jsonb
);


ALTER TABLE public.skus OWNER TO admin;

--
-- Name: sku_search_corpus; Type: MATERIALIZED VIEW; Schema: public; Owner: admin
--

CREATE MATERIALIZED VIEW public.sku_search_corpus AS
 WITH primary_img AS (
         SELECT si.sku_id,
            si.image_path,
            si.ocr_text
           FROM public.sku_images si
          WHERE (si.is_primary IS TRUE)
        ), accepted_caption AS (
         SELECT DISTINCT ON (sc.sku_id) sc.sku_id,
            COALESCE(cl.corrected_text, sc.caption_text) AS caption_ok
           FROM (public.sku_captions sc
             LEFT JOIN public.caption_labels cl ON ((cl.caption_id = sc.id)))
          WHERE (COALESCE(cl.is_acceptable, true) = true)
          ORDER BY sc.sku_id, sc.created_at DESC
        )
 SELECT s.id AS sku_id,
    COALESCE(b.name, ''::text) AS brand,
    s.name,
    s.variant,
    s.size_text,
    s.barcode,
    pi.image_path AS cover_path,
    pi.ocr_text,
    ac.caption_ok AS caption_text,
    TRIM(BOTH FROM regexp_replace(public.vn_norm_simple(concat_ws(' '::text, b.name, array_to_string(b.synonyms, ' '::text), s.name, s.variant, s.size_text, ac.caption_ok, pi.ocr_text)), '\s+'::text, ' '::text, 'g'::text)) AS normalized_text,
    (((setweight(to_tsvector('simple'::regconfig, ((COALESCE(b.name, ''::text) || ' '::text) || COALESCE(s.name, ''::text))), 'A'::"char") || setweight(to_tsvector('simple'::regconfig, ((COALESCE(s.variant, ''::text) || ' '::text) || COALESCE(s.size_text, ''::text))), 'B'::"char")) || setweight(to_tsvector('simple'::regconfig, COALESCE(ac.caption_ok, ''::text)), 'C'::"char")) || setweight(to_tsvector('simple'::regconfig, COALESCE(pi.ocr_text, ''::text)), 'D'::"char")) AS tsv_weighted
   FROM (((public.skus s
     LEFT JOIN public.brands b ON ((b.id = s.brand_id)))
     LEFT JOIN primary_img pi ON ((pi.sku_id = s.id)))
     LEFT JOIN accepted_caption ac ON ((ac.sku_id = s.id)))
  WHERE (s.is_active IS TRUE)
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.sku_search_corpus OWNER TO admin;

--
-- Name: sku_texts; Type: TABLE; Schema: public; Owner: admin
--

CREATE TABLE public.sku_texts (
    id bigint NOT NULL,
    sku_id bigint,
    text text NOT NULL,
    text_vec public.vector(512),
    text_vec_1k public.vector(1024),
    tsv tsvector
);


ALTER TABLE public.sku_texts OWNER TO admin;

--
-- Name: sku_search_view; Type: VIEW; Schema: public; Owner: admin
--

CREATE VIEW public.sku_search_view AS
 SELECT s.id AS sku_id,
    COALESCE(b.name, ''::text) AS brand,
    s.name,
    s.variant,
    s.size_text,
    s.barcode,
    st.text AS normalized_text,
    ( SELECT si.image_path
           FROM public.sku_images si
          WHERE ((si.sku_id = s.id) AND (si.is_primary IS TRUE))
         LIMIT 1) AS cover_path,
    s.is_active,
    s.updated_at
   FROM ((public.skus s
     LEFT JOIN public.brands b ON ((b.id = s.brand_id)))
     LEFT JOIN public.sku_texts st ON ((st.sku_id = s.id)));


ALTER VIEW public.sku_search_view OWNER TO admin;

--
-- Name: sku_texts_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.sku_texts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sku_texts_id_seq OWNER TO admin;

--
-- Name: sku_texts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.sku_texts_id_seq OWNED BY public.sku_texts.id;


--
-- Name: skus_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.skus_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.skus_id_seq OWNER TO admin;

--
-- Name: skus_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.skus_id_seq OWNED BY public.skus.id;


--
-- Name: brands id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.brands ALTER COLUMN id SET DEFAULT nextval('public.brands_id_seq'::regclass);


--
-- Name: caption_labels id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.caption_labels ALTER COLUMN id SET DEFAULT nextval('public.caption_labels_id_seq'::regclass);


--
-- Name: categories id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.categories ALTER COLUMN id SET DEFAULT nextval('public.categories_id_seq'::regclass);


--
-- Name: queries id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.queries ALTER COLUMN id SET DEFAULT nextval('public.queries_id_seq'::regclass);


--
-- Name: query_candidates id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_candidates ALTER COLUMN id SET DEFAULT nextval('public.query_candidates_id_seq'::regclass);


--
-- Name: query_labels id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_labels ALTER COLUMN id SET DEFAULT nextval('public.query_labels_id_seq'::regclass);


--
-- Name: sku_ai_suggestions id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_ai_suggestions ALTER COLUMN id SET DEFAULT nextval('public.sku_ai_suggestions_id_seq'::regclass);


--
-- Name: sku_captions id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_captions ALTER COLUMN id SET DEFAULT nextval('public.sku_captions_id_seq'::regclass);


--
-- Name: sku_images id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_images ALTER COLUMN id SET DEFAULT nextval('public.sku_images_id_seq'::regclass);


--
-- Name: sku_texts id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_texts ALTER COLUMN id SET DEFAULT nextval('public.sku_texts_id_seq'::regclass);


--
-- Name: skus id; Type: DEFAULT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.skus ALTER COLUMN id SET DEFAULT nextval('public.skus_id_seq'::regclass);


--
-- Name: brands brands_name_key; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.brands
    ADD CONSTRAINT brands_name_key UNIQUE (name);


--
-- Name: brands brands_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.brands
    ADD CONSTRAINT brands_pkey PRIMARY KEY (id);


--
-- Name: caption_labels caption_labels_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.caption_labels
    ADD CONSTRAINT caption_labels_pkey PRIMARY KEY (id);


--
-- Name: categories categories_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.categories
    ADD CONSTRAINT categories_pkey PRIMARY KEY (id);


--
-- Name: categories categories_slug_key; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.categories
    ADD CONSTRAINT categories_slug_key UNIQUE (slug);


--
-- Name: queries queries_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.queries
    ADD CONSTRAINT queries_pkey PRIMARY KEY (id);


--
-- Name: query_candidates query_candidates_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_candidates
    ADD CONSTRAINT query_candidates_pkey PRIMARY KEY (id);


--
-- Name: query_labels query_labels_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_labels
    ADD CONSTRAINT query_labels_pkey PRIMARY KEY (id);


--
-- Name: sku_ai_suggestions sku_ai_suggestions_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_ai_suggestions
    ADD CONSTRAINT sku_ai_suggestions_pkey PRIMARY KEY (id);


--
-- Name: sku_captions sku_captions_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_captions
    ADD CONSTRAINT sku_captions_pkey PRIMARY KEY (id);


--
-- Name: sku_images sku_images_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_images
    ADD CONSTRAINT sku_images_pkey PRIMARY KEY (id);


--
-- Name: sku_texts sku_texts_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_texts
    ADD CONSTRAINT sku_texts_pkey PRIMARY KEY (id);


--
-- Name: sku_texts sku_texts_uq; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_texts
    ADD CONSTRAINT sku_texts_uq UNIQUE (sku_id, text);


--
-- Name: skus skus_barcode_key; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.skus
    ADD CONSTRAINT skus_barcode_key UNIQUE (barcode);


--
-- Name: skus skus_brand_name_var_size_uq; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.skus
    ADD CONSTRAINT skus_brand_name_var_size_uq UNIQUE (brand_id, name, variant, size_text);


--
-- Name: skus skus_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.skus
    ADD CONSTRAINT skus_pkey PRIMARY KEY (id);


--
-- Name: sku_ai_suggestions uq_ai_suggest; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_ai_suggestions
    ADD CONSTRAINT uq_ai_suggest UNIQUE (sku_id, image_id, model_name, prompt_ver);


--
-- Name: sku_captions uq_caption_variant; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_captions
    ADD CONSTRAINT uq_caption_variant UNIQUE (sku_id, image_path, lang, style, model_name, prompt_version);


--
-- Name: idx_qc_query; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_qc_query ON public.query_candidates USING btree (query_id);


--
-- Name: idx_qc_sku; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_qc_sku ON public.query_candidates USING btree (sku_id);


--
-- Name: idx_ql_query; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_ql_query ON public.query_labels USING btree (query_id);


--
-- Name: idx_ql_sku; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_ql_sku ON public.query_labels USING btree (chosen_sku_id);


--
-- Name: idx_queries_image_vec_hnsw; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_queries_image_vec_hnsw ON public.queries USING hnsw (query_image_vec public.vector_cosine_ops);


--
-- Name: idx_queries_text_vec_hnsw; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_queries_text_vec_hnsw ON public.queries USING hnsw (query_text_vec public.vector_cosine_ops);


--
-- Name: idx_sc_caption_trgm; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_caption_trgm ON public.sku_captions USING gin (caption_text public.gin_trgm_ops);


--
-- Name: idx_sc_colors_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_colors_gin ON public.sku_captions USING gin (colors);


--
-- Name: idx_sc_facet_scores; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_facet_scores ON public.sku_captions USING gin (facet_scores);


--
-- Name: idx_sc_keywords_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_keywords_gin ON public.sku_captions USING gin (keywords);


--
-- Name: idx_sc_materials_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_materials_gin ON public.sku_captions USING gin (materials);


--
-- Name: idx_sc_packaging_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_packaging_gin ON public.sku_captions USING gin (packaging);


--
-- Name: idx_sc_shapes_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_shapes_gin ON public.sku_captions USING gin (shapes);


--
-- Name: idx_sc_taste_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_taste_gin ON public.sku_captions USING gin (taste);


--
-- Name: idx_sc_texture_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sc_texture_gin ON public.sku_captions USING gin (texture);


--
-- Name: idx_sku_corpus_norm_trgm; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_corpus_norm_trgm ON public.sku_search_corpus USING gin (normalized_text public.gin_trgm_ops);


--
-- Name: idx_sku_corpus_tsv_w; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_corpus_tsv_w ON public.sku_search_corpus USING gin (tsv_weighted);


--
-- Name: idx_sku_images_colors_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_images_colors_gin ON public.sku_images USING gin (colors);


--
-- Name: idx_sku_images_facets_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_images_facets_gin ON public.sku_images USING gin (detected_facets);


--
-- Name: idx_sku_images_sku; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_images_sku ON public.sku_images USING btree (sku_id);


--
-- Name: idx_sku_images_vec768_hnsw; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_images_vec768_hnsw ON public.sku_images USING hnsw (image_vec_768 public.vector_cosine_ops);


--
-- Name: idx_sku_images_vec_hnsw; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_images_vec_hnsw ON public.sku_images USING hnsw (image_vec public.vector_cosine_ops);


--
-- Name: idx_sku_texts_sku; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_texts_sku ON public.sku_texts USING btree (sku_id);


--
-- Name: idx_sku_texts_trgm; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_texts_trgm ON public.sku_texts USING gin (text public.gin_trgm_ops);


--
-- Name: idx_sku_texts_tsv; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_texts_tsv ON public.sku_texts USING gin (tsv);


--
-- Name: idx_sku_texts_vec1k_hnsw; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_texts_vec1k_hnsw ON public.sku_texts USING hnsw (text_vec_1k public.vector_cosine_ops);


--
-- Name: idx_sku_texts_vec_hnsw; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_sku_texts_vec_hnsw ON public.sku_texts USING hnsw (text_vec public.vector_cosine_ops);


--
-- Name: idx_skus_active; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_skus_active ON public.skus USING btree (is_active);


--
-- Name: idx_skus_attrs_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_skus_attrs_gin ON public.skus USING gin (attrs);


--
-- Name: idx_skus_brand; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_skus_brand ON public.skus USING btree (brand_id);


--
-- Name: idx_skus_category; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_skus_category ON public.skus USING btree (category_id);


--
-- Name: idx_skus_facets_gin; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX idx_skus_facets_gin ON public.skus USING gin (facets);


--
-- Name: uq_one_primary_image_per_sku; Type: INDEX; Schema: public; Owner: admin
--

CREATE UNIQUE INDEX uq_one_primary_image_per_sku ON public.sku_images USING btree (sku_id) WHERE (is_primary IS TRUE);


--
-- Name: uq_sku_corpus_sku; Type: INDEX; Schema: public; Owner: admin
--

CREATE UNIQUE INDEX uq_sku_corpus_sku ON public.sku_search_corpus USING btree (sku_id);


--
-- Name: caption_labels trg_caption_labels_gt; Type: TRIGGER; Schema: public; Owner: admin
--

CREATE TRIGGER trg_caption_labels_gt AFTER INSERT OR UPDATE OF is_acceptable ON public.caption_labels FOR EACH ROW EXECUTE FUNCTION public.set_caption_ground_truth();


--
-- Name: sku_captions trg_caption_to_text; Type: TRIGGER; Schema: public; Owner: admin
--

CREATE TRIGGER trg_caption_to_text AFTER INSERT OR UPDATE OF caption_text, style ON public.sku_captions FOR EACH ROW EXECUTE FUNCTION public.upsert_sku_text_from_caption();


--
-- Name: sku_captions trg_sc_tsv; Type: TRIGGER; Schema: public; Owner: admin
--

CREATE TRIGGER trg_sc_tsv BEFORE INSERT OR UPDATE OF caption_text ON public.sku_captions FOR EACH ROW EXECUTE FUNCTION public.sku_captions_tsv_update();


--
-- Name: sku_texts trg_sku_texts_tsv; Type: TRIGGER; Schema: public; Owner: admin
--

CREATE TRIGGER trg_sku_texts_tsv BEFORE INSERT OR UPDATE OF text ON public.sku_texts FOR EACH ROW EXECUTE FUNCTION public.sku_texts_tsv_update();


--
-- Name: skus trg_skus_updated_at; Type: TRIGGER; Schema: public; Owner: admin
--

CREATE TRIGGER trg_skus_updated_at BEFORE UPDATE ON public.skus FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();


--
-- Name: caption_labels caption_labels_caption_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.caption_labels
    ADD CONSTRAINT caption_labels_caption_id_fkey FOREIGN KEY (caption_id) REFERENCES public.sku_captions(id) ON DELETE CASCADE;


--
-- Name: categories categories_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.categories
    ADD CONSTRAINT categories_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.categories(id) ON DELETE SET NULL;


--
-- Name: query_candidates query_candidates_query_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_candidates
    ADD CONSTRAINT query_candidates_query_id_fkey FOREIGN KEY (query_id) REFERENCES public.queries(id) ON DELETE CASCADE;


--
-- Name: query_candidates query_candidates_sku_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_candidates
    ADD CONSTRAINT query_candidates_sku_id_fkey FOREIGN KEY (sku_id) REFERENCES public.skus(id) ON DELETE CASCADE;


--
-- Name: query_labels query_labels_chosen_sku_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_labels
    ADD CONSTRAINT query_labels_chosen_sku_id_fkey FOREIGN KEY (chosen_sku_id) REFERENCES public.skus(id) ON DELETE CASCADE;


--
-- Name: query_labels query_labels_query_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.query_labels
    ADD CONSTRAINT query_labels_query_id_fkey FOREIGN KEY (query_id) REFERENCES public.queries(id) ON DELETE CASCADE;


--
-- Name: sku_ai_suggestions sku_ai_suggestions_image_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_ai_suggestions
    ADD CONSTRAINT sku_ai_suggestions_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.sku_images(id) ON DELETE SET NULL;


--
-- Name: sku_ai_suggestions sku_ai_suggestions_sku_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_ai_suggestions
    ADD CONSTRAINT sku_ai_suggestions_sku_id_fkey FOREIGN KEY (sku_id) REFERENCES public.skus(id) ON DELETE CASCADE;


--
-- Name: sku_captions sku_captions_sku_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_captions
    ADD CONSTRAINT sku_captions_sku_id_fkey FOREIGN KEY (sku_id) REFERENCES public.skus(id) ON DELETE CASCADE;


--
-- Name: sku_images sku_images_sku_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_images
    ADD CONSTRAINT sku_images_sku_id_fkey FOREIGN KEY (sku_id) REFERENCES public.skus(id) ON DELETE CASCADE;


--
-- Name: sku_texts sku_texts_sku_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.sku_texts
    ADD CONSTRAINT sku_texts_sku_id_fkey FOREIGN KEY (sku_id) REFERENCES public.skus(id) ON DELETE CASCADE;


--
-- Name: skus skus_brand_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.skus
    ADD CONSTRAINT skus_brand_id_fkey FOREIGN KEY (brand_id) REFERENCES public.brands(id);


--
-- Name: skus skus_category_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.skus
    ADD CONSTRAINT skus_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.categories(id);


--
-- PostgreSQL database dump complete
--

\unrestrict QGDGFXkRDeaKHoOPsxkiia6ZUYurgHECvhyJfZc7nYNLK2xu6QmtyLe0rnHLbqb

