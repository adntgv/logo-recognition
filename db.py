"""Database and MinIO helpers."""
import os, json, uuid, io, psycopg2, psycopg2.extras
from datetime import datetime
from minio import Minio
from PIL import Image

DATABASE_URL = os.getenv("DATABASE_URL", "")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "logo-uploads")

_pg = None
_mc = None


def get_pg():
    global _pg
    if _pg is None or _pg.closed:
        _pg = psycopg2.connect(DATABASE_URL)
        _pg.autocommit = True
    return _pg


_mc_bucket_ok = False

def get_minio():
    global _mc, _mc_bucket_ok
    if _mc is None:
        _mc = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                     secret_key=MINIO_SECRET_KEY, secure=False)
    if not _mc_bucket_ok:
        if not _mc.bucket_exists(MINIO_BUCKET):
            _mc.make_bucket(MINIO_BUCKET)
        _mc_bucket_ok = True
    return _mc


def init_db():
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        original_name TEXT,
        uploaded_at TIMESTAMPTZ DEFAULT NOW(),
        file_size_kb INTEGER,
        predictions JSONB,
        top_prediction_id TEXT,
        top_prediction_score FLOAT,
        label_company_id TEXT,
        label_status TEXT DEFAULT 'unlabeled',
        labeled_by TEXT,
        labeled_at TIMESTAMPTZ
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(label_status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_uploads_label ON uploads(label_company_id)")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS admin_users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS unverified_companies (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        frequency INTEGER DEFAULT 1,
        first_seen TIMESTAMPTZ DEFAULT NOW(),
        last_seen TIMESTAMPTZ DEFAULT NOW(),
        added_by TEXT
    )""")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unverified_name ON unverified_companies(LOWER(name))")
    cur.close()


def save_upload(img: Image.Image, original_name: str, predictions: list, file_size_kb: int) -> int:
    """Resize to 512px, upload to MinIO, insert DB row. Returns upload id."""
    # Resize
    w, h = img.size
    if max(w, h) > 512:
        scale = 512 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    size = buf.getbuffer().nbytes

    now = datetime.utcnow()
    fname = f"{now.year}/{now.month:02d}/{uuid.uuid4().hex}.jpg"

    mc = get_minio()
    mc.put_object(MINIO_BUCKET, fname, buf, size, content_type="image/jpeg")

    top = predictions[0] if predictions else {}
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO uploads (filename, original_name, file_size_kb, predictions,
                             top_prediction_id, top_prediction_score)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
    """, (fname, original_name, file_size_kb,
          json.dumps(predictions),
          top.get("company_id"), top.get("final_score")))
    uid = cur.fetchone()[0]
    cur.close()
    return uid


def get_image_bytes(filename: str) -> bytes:
    mc = get_minio()
    resp = mc.get_object(MINIO_BUCKET, filename)
    data = resp.read()
    resp.close()
    resp.release_conn()
    return data
