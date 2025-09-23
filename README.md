# Flask + PostgreSQL (pgvector) Product Search/Admin

This starter provides:
- Flask CRUD UI for products
- Vector fields (image/text) with pgvector
- Embedding via OpenCLIP (ViT-B-32, laion2b_s34b_b79k, 512-dim)
- Basic image/text search endpoints

## Prereqs
- PostgreSQL 16 with `pgvector` extension
- Python 3.10+
- (Optional) GPU for faster embeddings. CPU works but slower.

## Quick start
1) Create DB and enable extension:
```sql
CREATE DATABASE shop;
\c shop
CREATE EXTENSION IF NOT EXISTS vector;
\i schema.sql
```

2) Install deps and run:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# configure environment (edit .env)
cp .env.sample .env

# run
python app.py
# or: FLASK_APP=app.py flask run -p 5000
```

Visit http://localhost:5000

## Notes
- Vector dimension is **512** (CLIP ViT-B-32). If you change the embedding model, adjust `VECTOR(512)` in `schema.sql` and the SQLAlchemy `Vector(512)` types in `db.py`.
- For cosine similarity we normalize vectors to unit length before storing.
