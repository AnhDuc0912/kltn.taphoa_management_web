import os
import zipfile
import subprocess
from dotenv import load_dotenv

# Load DB config from .env
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS").strip("'")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

BACKUP_DIR = "backup"
os.makedirs(BACKUP_DIR, exist_ok=True)

# 1. Backup uploads folder to zip
zip_path = os.path.join(BACKUP_DIR, "uploads_backup.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, UPLOAD_FOLDER)
            zipf.write(abs_path, arcname=rel_path)
print(f"ÄÃ£ backup uploads vÃ o {zip_path}")

# 2. Backup database using pg_dump
sql_path = os.path.join(BACKUP_DIR, "db_backup.sql")
pg_dump_cmd = [
    "pg_dump",
    f"-h{DB_HOST}",
    f"-p{DB_PORT}",
    f"-U{DB_USER}",
    "-Fc", # custom format, cÃ³ thá»ƒ Ä‘á»•i thÃ nh -f náº¿u muá»‘n plain SQL
    "-f", sql_path,
    DB_NAME
]
# Set PGPASSWORD env for pg_dump
env = os.environ.copy()
env["PGPASSWORD"] = DB_PASS
try:
    subprocess.run(pg_dump_cmd, check=True, env=env)
    print(f"ÄÃ£ backup database vÃ o {sql_path}")
except Exception as e:
    print(f"Backup DB lá»—i: {e}\nBáº¡n cáº§n cÃ i pg_dump vÃ  thÃªm vÃ o PATH.")

