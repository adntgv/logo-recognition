#!/usr/bin/env python3
"""FastAPI logo recognition service."""
import json, time, io, os, numpy as np, faiss
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from rapidfuzz import fuzz
import open_clip, torch

INDEX_DIR = Path(__file__).parent / "index_artifacts"
DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Logo Recognition PoC")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount static AFTER explicit routes are registered (done at bottom)

# Globals loaded on startup
model = None
preprocess = None
index = None
metadata = []
companies_lookup = {}

# DB/Auth imports (lazy — only used if env vars present)
_db_ready = False

def _init_services():
    global _db_ready
    if _db_ready:
        return
    if os.getenv("DATABASE_URL"):
        try:
            from db import init_db
            init_db()
            # Create default admin
            from auth import hash_password
            from db import get_pg
            conn = get_pg()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO admin_users (username, password_hash) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING",
                ("admin", hash_password("halaldamu2026")))
            cur.close()
            _db_ready = True
            print("Database initialized")
        except Exception as e:
            print(f"DB init warning: {e}")

def _pull_data_from_minio():
    """Try to pull companies.json and index artifacts from MinIO if missing locally."""
    try:
        from sync_companies import pull_from_minio
        # Only pull if local data is missing
        if not (DATA_DIR / "companies.json").exists() or not (INDEX_DIR / "index.faiss").exists():
            print("Local data missing, pulling from MinIO...")
            return pull_from_minio()
        else:
            print("Local data exists, skipping MinIO pull")
            return True
    except Exception as e:
        print(f"MinIO pull skipped: {e}")
        return False


@app.on_event("startup")
def startup():
    global model, preprocess, index, metadata, companies_lookup
    
    # Try pulling data from MinIO if local files are missing
    if os.getenv("MINIO_ENDPOINT"):
        _pull_data_from_minio()
    
    print("Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    
    print("Loading index...")
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metadata.jsonl") as f:
        metadata = [json.loads(l) for l in f]
    
    companies = json.loads((DATA_DIR / "companies.json").read_text())
    for c in companies:
        companies_lookup[c["company_id"]] = c
    print(f"Ready: {index.ntotal} vectors, {len(companies_lookup)} companies")
    
    _init_services()

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/admin")
async def admin_page():
    return FileResponse(str(STATIC_DIR / "admin.html"))


# --- Auth helper ---
def require_admin(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    from auth import verify_token
    user = verify_token(auth[7:])
    if not user:
        raise HTTPException(401, "Invalid token")
    return user


# --- Auth endpoint ---
@app.post("/api/login")
async def api_login(request: Request):
    body = await request.json()
    from auth import authenticate
    token = authenticate(body.get("username", ""), body.get("password", ""))
    if not token:
        raise HTTPException(401, "Bad credentials")
    return {"token": token}


# --- Admin API ---
@app.get("/api/admin/stats")
async def admin_stats(user: str = Depends(require_admin)):
    from db import get_pg
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE label_status='unlabeled') as unlabeled,
            COUNT(*) FILTER (WHERE label_status='confirmed') as confirmed,
            COUNT(*) FILTER (WHERE label_status='corrected') as corrected,
            COUNT(*) FILTER (WHERE label_status='unknown') as unknown,
            COUNT(*) FILTER (WHERE label_status='not_halaldamu') as not_halaldamu
        FROM uploads
    """)
    r = cur.fetchone()
    cur.close()
    labeled = r[2] + r[3]
    accuracy = round(r[2] / labeled * 100, 1) if labeled > 0 else 0
    return {"total": r[0], "unlabeled": r[1], "confirmed": r[2],
            "corrected": r[3], "unknown": r[4], "not_halaldamu": r[5],
            "accuracy": accuracy}


@app.get("/api/admin/uploads")
async def admin_uploads(status: str = "", page: int = 1, limit: int = 20,
                        user: str = Depends(require_admin)):
    from db import get_pg
    conn = get_pg()
    cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
    offset = (page - 1) * limit
    if status and status != "all":
        cur.execute("SELECT id, filename, original_name, uploaded_at, predictions, top_prediction_id, top_prediction_score, label_company_id, label_status FROM uploads WHERE label_status=%s ORDER BY id DESC LIMIT %s OFFSET %s", (status, limit, offset))
    else:
        cur.execute("SELECT id, filename, original_name, uploaded_at, predictions, top_prediction_id, top_prediction_score, label_company_id, label_status FROM uploads ORDER BY id DESC LIMIT %s OFFSET %s", (limit, offset))
    rows = cur.fetchall()
    # Get total
    if status and status != "all":
        cur.execute("SELECT COUNT(*) FROM uploads WHERE label_status=%s", (status,))
    else:
        cur.execute("SELECT COUNT(*) FROM uploads")
    total = cur.fetchone()["count"]
    cur.close()
    # Convert datetimes to str
    for r in rows:
        if r.get("uploaded_at"):
            r["uploaded_at"] = r["uploaded_at"].isoformat()
    return {"items": rows, "total": total, "page": page, "limit": limit}


@app.get("/api/admin/uploads/{upload_id}")
async def admin_upload_detail(upload_id: int, user: str = Depends(require_admin)):
    from db import get_pg
    import psycopg2.extras
    conn = get_pg()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM uploads WHERE id=%s", (upload_id,))
    row = cur.fetchone()
    cur.close()
    if not row:
        raise HTTPException(404)
    if row.get("uploaded_at"):
        row["uploaded_at"] = row["uploaded_at"].isoformat()
    if row.get("labeled_at"):
        row["labeled_at"] = row["labeled_at"].isoformat()
    return row


@app.patch("/api/admin/uploads/{upload_id}")
async def admin_label_upload(upload_id: int, request: Request,
                             user: str = Depends(require_admin)):
    body = await request.json()
    from db import get_pg
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("""
        UPDATE uploads SET label_company_id=%s, label_status=%s,
               labeled_by=%s, labeled_at=NOW()
        WHERE id=%s
    """, (body.get("label_company_id"), body.get("label_status"), user, upload_id))
    cur.close()
    return {"ok": True}


@app.get("/api/admin/companies")
async def admin_companies(q: str = "", user: str = Depends(require_admin)):
    if not q or len(q) < 2:
        return []
    q_lower = q.lower()
    results = []
    for cid, c in companies_lookup.items():
        name = c.get("company_name", "")
        if q_lower in name.lower():
            results.append({"company_id": cid, "company_name": name})
            if len(results) >= 20:
                break
    return results


@app.post("/api/sync")
async def api_sync(user: str = Depends(require_admin)):
    """Hot reload: sync companies from HalalDamu API and rebuild FAISS index."""
    global index, metadata, companies_lookup
    from sync_companies import fetch_and_sync, pull_from_minio
    result = fetch_and_sync()
    # Reload index
    pull_from_minio()
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metadata.jsonl") as f:
        metadata = [json.loads(l) for l in f]
    companies = json.loads((DATA_DIR / "companies.json").read_text())
    companies_lookup.clear()
    for c in companies:
        companies_lookup[c["company_id"]] = c
    result["reloaded_vectors"] = index.ntotal
    result["reloaded_companies"] = len(companies_lookup)
    return result


@app.post("/api/admin/rebuild-index")
async def admin_rebuild_index(user: str = Depends(require_admin)):
    """Rebuild FAISS index incorporating labeled uploads."""
    global index, metadata
    import subprocess
    result = subprocess.run(
        ["python3", "rebuild_index.py"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent)
    )
    # Reload
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metadata.jsonl") as f:
        metadata = [json.loads(l) for l in f]
    return {"ok": True, "vectors": index.ntotal, "output": result.stdout[-500:] if result.stdout else result.stderr[-500:]}


@app.post("/api/admin/unverified")
async def admin_add_unverified(request: Request, user: str = Depends(require_admin)):
    """Add or increment an unverified company name."""
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    from db import get_pg
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO unverified_companies (name, added_by)
        VALUES (%s, %s)
        ON CONFLICT (LOWER(name)) DO UPDATE SET
            frequency = unverified_companies.frequency + 1,
            last_seen = NOW()
        RETURNING id
    """, (name, user))
    uid = cur.fetchone()[0]
    cur.close()
    return {"ok": True, "id": uid}


@app.get("/api/admin/unverified")
async def admin_list_unverified(user: str = Depends(require_admin)):
    from db import get_pg
    import psycopg2.extras
    conn = get_pg()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM unverified_companies ORDER BY frequency DESC, last_seen DESC LIMIT 50")
    rows = cur.fetchall()
    cur.close()
    for r in rows:
        for k in ("first_seen", "last_seen"):
            if r.get(k):
                r[k] = r[k].isoformat()
    return rows


@app.get("/api/uploads/{upload_id}/image")
async def get_upload_image(upload_id: int):
    from db import get_pg, get_image_bytes
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("SELECT filename FROM uploads WHERE id=%s", (upload_id,))
    row = cur.fetchone()
    cur.close()
    if not row:
        raise HTTPException(404)
    data = get_image_bytes(row[0])
    return Response(content=data, media_type="image/jpeg")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def resize_max(img: Image.Image, max_side=1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

def multi_crop(img: Image.Image):
    """Generate multiple crops from image."""
    w, h = img.size
    crops = []
    
    # Center crops at different scales
    for scale in [0.5, 0.7, 0.9]:
        cw, ch = int(w*scale), int(h*scale)
        x, y = (w-cw)//2, (h-ch)//2
        crops.append(img.crop((x, y, x+cw, y+ch)))
    
    # Corner crops
    for scale in [0.5, 0.7]:
        cw, ch = int(w*scale), int(h*scale)
        for (x, y) in [(0,0), (w-cw,0), (0,h-ch), (w-cw,h-ch)]:
            crops.append(img.crop((x, y, x+cw, y+ch)))
    
    # Full image
    crops.append(img)
    return crops

def embed_crops(crops):
    tensors = [preprocess(c).unsqueeze(0) for c in crops]
    batch = torch.cat(tensors, dim=0)
    with torch.no_grad():
        embs = model.encode_image(batch)
        embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.numpy().astype(np.float32)

def ocr_text(img: Image.Image) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""

def text_score(ocr_tokens, company):
    if not ocr_tokens:
        return 0.0
    names = [company["company_name"]] + company.get("aliases", [])
    best = 0.0
    for token in ocr_tokens:
        for name in names:
            score = fuzz.partial_ratio(token.lower(), name.lower()) / 100.0
            best = max(best, score)
    return best

@app.post("/match")
async def match(file: UploadFile = File(...)):
    t_start = time.time()
    
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = resize_max(img)
    t_preprocess = time.time()
    
    # Multi-crop + embed
    crops = multi_crop(img)
    embeddings = embed_crops(crops)
    t_embed = time.time()
    
    # FAISS search per crop
    k = 10
    company_scores = {}
    for emb in embeddings:
        D, I = index.search(emb.reshape(1, -1), k)
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            cid = metadata[idx]["company_id"]
            if cid not in company_scores or score > company_scores[cid]:
                company_scores[cid] = float(score)
    t_search = time.time()
    
    # OCR
    ocr_raw = ocr_text(img)
    tokens = [t for t in ocr_raw.split() if len(t) >= 3]
    t_ocr = time.time()
    
    # Score fusion
    candidates = []
    for cid, visual_score in sorted(company_scores.items(), key=lambda x: -x[1])[:20]:
        c = companies_lookup.get(cid, {})
        ts = text_score(tokens, c) if c else 0.0
        final = 0.85 * visual_score + 0.15 * ts
        candidates.append({
            "company_id": cid,
            "company_name": c.get("company_name", ""),
            "visual_score": round(visual_score, 4),
            "text_score": round(ts, 4),
            "final_score": round(final, 4),
        })
    
    candidates.sort(key=lambda x: -x["final_score"])
    candidates = candidates[:10]
    
    label = "unknown" if not candidates or candidates[0]["final_score"] < 0.3 else candidates[0]["company_name"]
    
    # Save to MinIO + DB (non-blocking, best effort)
    if _db_ready:
        try:
            from db import save_upload
            save_upload(img, file.filename or "unknown.jpg", candidates, len(data) // 1024)
        except Exception as e:
            print(f"Save upload error: {e}")
    
    t_end = time.time()
    
    return {
        "label": label,
        "candidates": candidates,
        "ocr_tokens": tokens[:10],
        "timing_ms": {
            "preprocess": round((t_preprocess - t_start) * 1000),
            "embed": round((t_embed - t_preprocess) * 1000),
            "search": round((t_search - t_embed) * 1000),
            "ocr": round((t_ocr - t_search) * 1000),
            "total": round((t_end - t_start) * 1000),
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
