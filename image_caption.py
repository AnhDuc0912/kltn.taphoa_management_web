# image_caption.py
import os, csv, time, mimetypes, argparse, glob, logging, traceback
import urllib.request
from urllib.parse import urlparse
from typing import Tuple, Optional
from datetime import datetime

import google.generativeai as genai
from db import get_conn

# ====== Cáº¥u hÃ¬nh ======
MODEL_NAME = "gemini-1.5-flash"
PROMPT_VI = (
    "Hãy mô tả ngắn gọn nội dung hình ảnh dùng làm caption sản phẩm. "
    "Tối đa 1–2 câu, khách quan, tránh quảng cáo quá đà. "
    "Trả lời bằng tiếng Việt."
)

REQUEST_TIMEOUT = 60
RETRY = 3
DELAY_BETWEEN_CALLS = 0.5  # giÃ¢y
UPLOAD_DIR = os.path.abspath(os.getenv("UPLOAD_FOLDER", "uploads"))

# ====== Logging ======
def setup_logging(debug: bool, logfile: Optional[str] = None):
    os.makedirs("logs", exist_ok=True)
    if not logfile:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join("logs", f"image_caption_{ts}.log")

    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s | %(levelname)s | %(message)s"

    logging.basicConfig(level=level, format=fmt,
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(logfile, encoding="utf-8")
                        ])
    logging.info("=== Logger initialized ===")
    logging.info(f"Debug mode: {debug}")
    logging.info(f"Log file  : {logfile}")
    return logfile

# ====== DB helper ======
def q(sql, params=None, fetch="one"):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        return cur.fetchone() if fetch == "one" else cur.fetchall()

def exec_sql(sql, params=None):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        conn.commit()

# ====== Tiá»‡n Ã­ch ======
def latest_csv_in(folder: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    return files[-1] if files else None

def _guess_mime_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".bmp": "image/bmp", ".gif": "image/gif"
    }.get(ext, mimetypes.guess_type(path)[0] or "image/jpeg")

def fetch_bytes_from_image_url(url: str) -> Tuple[bytes, str, str]:
    """
    Æ¯u tiÃªn Ä‘á»c file local náº¿u URL lÃ  127.0.0.1/localhost vÃ  path báº¯t Ä‘áº§u /uploads/.
    Náº¿u URL lÃ  Ä‘Æ°á»ng dáº«n relative hoáº·c báº¯t Ä‘áº§u /uploads/ vÃ  cÃ³ APP_HOST thÃ¬ sáº½ chuyá»ƒn
    thÃ nh HTTP request tá»›i APP_HOST (vÃ­ dá»¥: https://hango.ducdatphat.id.vn/uploads/...)
    Tráº£ vá»: (bytes, mime, source) vá»›i source in {"local","http","http(APP_HOST)"}.
    """
    p = urlparse(url)
    app_host = os.getenv("APP_HOST", "").strip()

    def make_host_url(path: str) -> str:
        if not app_host:
            return path
        if app_host.startswith("http://") or app_host.startswith("https://"):
            base = app_host.rstrip("/")
        else:
            # máº·c Ä‘á»‹nh dÃ¹ng https vá»›i remote host; náº¿u cáº§n http, set APP_HOST includes scheme
            base = "https://" + app_host.rstrip("/")
        return base + "/" + path.lstrip("/")

    # Náº¿u lÃ  local dev host vÃ  uploads, Ä‘á»c file trá»±c tiáº¿p tá»« UPLOAD_DIR (náº¿u tá»“n táº¡i)
    try:
        if p.netloc in ("127.0.0.1:5000", "localhost:5000") and p.path.startswith("/uploads/"):
            fname = os.path.basename(p.path)
            fpath = os.path.join(UPLOAD_DIR, fname)
            logging.debug(f"Resolve local file: {fpath}")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"KhÃ´ng tháº¥y file local: {fpath}")
            with open(fpath, "rb") as f:
                data = f.read()
            return data, _guess_mime_from_ext(fpath), "local"
    except Exception as e:
        logging.debug(f"Local read fallback to HTTP due to: {e}")

    # Náº¿u URL lÃ  relative hoáº·c báº¯t Ä‘áº§u /uploads/ vÃ  cÃ³ APP_HOST -> build URL tá»›i APP_HOST
    if (p.netloc == "" and (p.path.startswith("/uploads/") or p.path.startswith("uploads/"))) and app_host:
        target_url = make_host_url(p.path)
        logging.debug(f"Resolved via APP_HOST -> {target_url}")
    else:
        # náº¿u URL Ä‘áº§y Ä‘á»§ thÃ¬ dÃ¹ng nguyÃªn báº£n
        target_url = url

    # HTTP fallback (cÃ³ retry)
    last_err = None
    for attempt in range(1, RETRY + 1):
        try:
            logging.debug(f"HTTP fetch attempt {attempt}: {target_url}")
            req = urllib.request.Request(target_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                ctype = resp.headers.get_content_type() or _guess_mime_from_ext(target_url)
                src_tag = "http(APP_HOST)" if app_host and target_url.startswith(("http://", "https://")) and app_host in target_url else "http"
                return data, ctype, src_tag
        except Exception as e:
            last_err = e
            logging.debug(f"HTTP fetch error attempt {attempt}: {e}")
            time.sleep(0.6 * attempt)
    raise last_err or RuntimeError("fetch_failed")

def get_api_key() -> str:
    # Há»— trá»£ cáº£ 2 tÃªn biáº¿n mÃ´i trÆ°á»ng
    api_key = os.getenv("API_KEY_GEMINI") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Thiáº¿u biáº¿n mÃ´i trÆ°á»ng API_KEY_GEMINI hoáº·c GOOGLE_API_KEY")
    return api_key

def gemini_caption(image_bytes: bytes, mime_type: str) -> str:
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    for attempt in range(1, RETRY + 1):
        try:
            logging.debug(f"Gemini call attempt {attempt}, mime={mime_type}, bytes={len(image_bytes)}")
            parts = [PROMPT_VI, {"mime_type": mime_type, "data": image_bytes}]
            res = model.generate_content(parts, request_options={"timeout": REQUEST_TIMEOUT})
            text = (res.text or "").strip()
            text = text.replace("\n", " ").strip()
            logging.debug(f"Gemini response: {text}")
            return text
        except Exception as e:
            logging.debug(f"Gemini error attempt {attempt}: {e}")
            if attempt == RETRY:
                raise
            time.sleep(1.2 * attempt)
    return ""

# ====== Cháº¡y chÃ­nh ======
def run(csv_path: Optional[str], limit: Optional[int], debug: bool, logfile: Optional[str]):
    setup_logging(debug, logfile)

    # 1) Chá»n CSV
    if not csv_path:
        csv_path = latest_csv_in(os.path.join(os.path.dirname(__file__), "csv_downloads"))
    if not csv_path:
        logging.error("KhÃ´ng tÃ¬m tháº¥y CSV trong ./csv_downloads. HÃ£y truyá»n Ä‘Æ°á»ng dáº«n CSV.")
        raise SystemExit(1)

    logging.info(f"Äá»c CSV: {csv_path}")
    logging.info(f"UPLOAD_DIR: {UPLOAD_DIR}")

    done, fail = 0, 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if limit is not None and done >= limit:
                break

            # CSV may contain either 'id' (sku_images.id) or 'sku_id' (sku id) + cover_url
            img_id_col = (row.get("id") or "").strip()
            sku_id_col = (row.get("sku_id") or "").strip()
            image_url = (row.get("image_url") or row.get("cover_url") or row.get("url") or "").strip()

            logging.info(f"Row {i} start | sku_id={sku_id_col or 'N/A'} img_id={img_id_col or 'N/A'} url={image_url}")

            try:
                # Determine what to fetch: prefer image_url; if missing and img_id provided, try lookup image_path
                if not image_url and img_id_col:
                    # find image_path for given sku_images.id
                    rec = q("SELECT image_path FROM sku_images WHERE id=%s", (int(img_id_col),), fetch="one")
                    if not rec:
                        raise LookupError(f"KhÃ´ng tÃ¬m tháº¥y sku_images.id={img_id_col}")
                    image_path_guess = rec[0]
                    # build a local url for fetching if possible
                    image_url = f"/{os.getenv('UPLOAD_FOLDER','uploads').strip().strip('/')}/{image_path_guess}"

                if not image_url:
                    raise ValueError("KhÃ´ng cÃ³ image_url trong CSV vÃ  khÃ´ng tÃ¬m Ä‘Æ°á»£c image_path")

                # Táº£i áº£nh (há»— trá»£ APP_HOST / local files)
                img_bytes, mime, src = fetch_bytes_from_image_url(image_url)
                logging.debug(f"Fetched image ({src}) | size={len(img_bytes)} mime={mime}")

                # Sinh mÃ´ táº£ (Gemini)
                caption = gemini_caption(img_bytes, mime)
                if not caption:
                    raise RuntimeError("Gemini tráº£ vá» rá»—ng")
                logging.info(f"Row {i} caption: {caption}")

                # XÃ¡c Ä‘á»‹nh sku_id vÃ  image_path lÆ°u vÃ o DB
                basename = os.path.basename(urlparse(image_url).path)
                sku_id_final = None
                image_path_final = None

                # If CSV provided sku_id, try to find matching sku_images row by basename
                if sku_id_col:
                    rec = q(
                        "SELECT id, image_path, sku_id FROM sku_images WHERE sku_id=%s AND image_path LIKE %s LIMIT 1",
                        (int(sku_id_col), f"%{basename}%"),
                        fetch="one"
                    )
                    if rec:
                        _, image_path_final, sku_id_found = rec
                        sku_id_final = sku_id_found
                    else:
                        # fallback: try any image matching basename
                        rec2 = q(
                            "SELECT id, image_path, sku_id FROM sku_images WHERE image_path LIKE %s LIMIT 1",
                            (f"%{basename}%",),
                            fetch="one"
                        )
                        if rec2:
                            _, image_path_final, sku_id_found = rec2
                            sku_id_final = sku_id_found
                        else:
                            # no matching sku_images row â€” use provided sku_id and store basename as image_path
                            sku_id_final = int(sku_id_col)
                            image_path_final = basename
                else:
                    # If CSV provided sku_images.id use that
                    if img_id_col:
                        rec = q("SELECT image_path, sku_id FROM sku_images WHERE id=%s", (int(img_id_col),), fetch="one")
                        if rec:
                            image_path_final, sku_id_found = rec
                            sku_id_final = sku_id_found
                        else:
                            raise LookupError(f"KhÃ´ng tÃ¬m tháº¥y sku_images.id={img_id_col}")
                    else:
                        # last resort: try to find any sku_images matching basename
                        rec = q("SELECT image_path, sku_id FROM sku_images WHERE image_path LIKE %s LIMIT 1", (f"%{basename}%",), fetch="one")
                        if rec:
                            image_path_final, sku_id_found = rec
                            sku_id_final = sku_id_found
                        else:
                            raise LookupError("KhÃ´ng tÃ¬m tháº¥y sku_images tÆ°Æ¡ng á»©ng vÃ  khÃ´ng cÃ³ sku_id trong CSV")

                if not sku_id_final or not image_path_final:
                    raise RuntimeError("KhÃ´ng xÃ¡c Ä‘á»‹nh Ä‘Æ°á»£c sku_id hoáº·c image_path Ä‘á»ƒ lÆ°u")

                # Upsert vÃ o sku_captions (vi, style=search)
                exec_sql("""
                    INSERT INTO sku_captions
                        (sku_id, image_path, lang, style, caption_text, model_name, prompt_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sku_id, image_path, lang, style, model_name, prompt_version)
                    DO UPDATE SET caption_text = EXCLUDED.caption_text
                """, (sku_id_final, image_path_final, "vi", "search", caption, MODEL_NAME, "v1"))

                logging.info(f"Row {i} upsert OK â†’ sku_captions (sku_id={sku_id_final}, image_path={image_path_final})")
                done += 1

                if DELAY_BETWEEN_CALLS:
                    time.sleep(DELAY_BETWEEN_CALLS)

            except Exception as e:
                fail += 1
                logging.error(f"[FAIL row {i}] sku_id={sku_id_col} img_id={img_id_col} url={image_url} â†’ {e}")
                if debug:
                    logging.error(traceback.format_exc())

    logging.info(f"HoÃ n táº¥t. OK={done}, FAIL={fail}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sinh mÃ´ táº£ áº£nh tá»« CSV báº±ng Gemini vÃ  cáº­p nháº­t báº£ng sku_captions"
    )
    parser.add_argument("--csv",
        help="ÄÆ°á»ng dáº«n CSV (cÃ³ cá»™t id,image_url). Náº¿u bá» trá»‘ng sáº½ láº¥y file má»›i nháº¥t trong ./csv_downloads")
    parser.add_argument("--limit", type=int, default=None, help="Giá»›i háº¡n sá»‘ dÃ²ng xá»­ lÃ½")
    parser.add_argument("--debug", action="store_true", help="In log chi tiáº¿t vÃ  stacktrace khi lá»—i")
    parser.add_argument("--logfile", help="ÄÆ°á»ng dáº«n file log (máº·c Ä‘á»‹nh: logs/image_caption_YYYYmmdd_HHMMSS.log)")
    args = parser.parse_args()
    run(args.csv, args.limit, args.debug, args.logfile)

