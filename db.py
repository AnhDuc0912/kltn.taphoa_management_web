import os, psycopg2
from dotenv import load_dotenv

# LuÃ´n load .env trÆ°á»›c
load_dotenv(".env")

# Náº¿u cÃ³ .env.production thÃ¬ override (Æ°u tiÃªn production)
# if os.path.exists(".env.production"):
#     load_dotenv(".env.production", override=True)


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "ducdatphat.id.vn"),
        port=int(os.getenv("DB_PORT", "5001")),
        dbname=os.getenv("DB_NAME", "hango"),
        user=os.getenv("DB_USER", "admin"),
        password=os.getenv("DB_PASS", "Duc091203@"),
    )

