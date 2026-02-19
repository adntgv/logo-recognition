#!/usr/bin/env python3
"""FastAPI logo recognition service."""
import json, time, io, numpy as np, faiss
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rapidfuzz import fuzz
import open_clip, torch

INDEX_DIR = Path(__file__).parent / "index_artifacts"
DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Logo Recognition PoC")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

# Globals loaded on startup
model = None
preprocess = None
index = None
metadata = []
companies_lookup = {}

@app.on_event("startup")
def startup():
    global model, preprocess, index, metadata, companies_lookup
    
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
