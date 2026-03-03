#!/usr/bin/env python3
"""Sync companies from HalalDamu API → MinIO → FAISS index.

Usage:
  python sync_companies.py              # full sync
  python sync_companies.py --upload     # upload existing local data to MinIO (bootstrap)
"""
import json, io, os, sys, time, requests, numpy as np, faiss
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image, ImageEnhance
from minio import Minio
import open_clip, torch

API_URL = "https://halaldamu.kz/wp-json/map/v1/active-companies?lang=kz&show_all=1"

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-qg4wgcogk8484gwow884s4go:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "logo-uploads")
DATA_PREFIX = "data/"  # prefix in MinIO for companies data

INDEX_DIR = Path(__file__).parent / "index_artifacts"
INDEX_DIR.mkdir(exist_ok=True)
LOCAL_DATA = Path(__file__).parent / "data"


def get_minio():
    mc = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
               secret_key=MINIO_SECRET_KEY, secure=False)
    if not mc.bucket_exists(MINIO_BUCKET):
        mc.make_bucket(MINIO_BUCKET)
    return mc


def upload_local_to_minio():
    """Bootstrap: upload existing local companies.json + logos to MinIO."""
    mc = get_minio()

    # Upload companies.json
    cj = LOCAL_DATA / "companies.json"
    if cj.exists():
        data = cj.read_bytes()
        mc.put_object(MINIO_BUCKET, DATA_PREFIX + "companies.json",
                      io.BytesIO(data), len(data), content_type="application/json")
        print(f"Uploaded companies.json ({len(data)} bytes)")

    # Upload logos
    logos_dir = LOCAL_DATA / "logos"
    if logos_dir.exists():
        count = 0
        for logo_file in logos_dir.rglob("*"):
            if logo_file.is_file():
                rel = logo_file.relative_to(LOCAL_DATA)
                obj_name = DATA_PREFIX + str(rel)
                data = logo_file.read_bytes()
                ct = "image/jpeg" if logo_file.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                mc.put_object(MINIO_BUCKET, obj_name, io.BytesIO(data), len(data), content_type=ct)
                count += 1
                if count % 100 == 0:
                    print(f"  Uploaded {count} logos...")
        print(f"Uploaded {count} logos total")

    # Upload index artifacts
    for fname in ["index.faiss", "metadata.jsonl", "embeddings.npy"]:
        fpath = INDEX_DIR / fname
        if fpath.exists():
            data = fpath.read_bytes()
            mc.put_object(MINIO_BUCKET, DATA_PREFIX + "index_artifacts/" + fname,
                          io.BytesIO(data), len(data))
            print(f"Uploaded index_artifacts/{fname}")


def pull_from_minio():
    """Pull companies.json, logos, and index artifacts from MinIO to local."""
    mc = get_minio()

    # Download companies.json
    try:
        resp = mc.get_object(MINIO_BUCKET, DATA_PREFIX + "companies.json")
        LOCAL_DATA.mkdir(parents=True, exist_ok=True)
        (LOCAL_DATA / "companies.json").write_bytes(resp.read())
        resp.close(); resp.release_conn()
        print("Pulled companies.json from MinIO")
    except Exception as e:
        print(f"No companies.json in MinIO: {e}")
        return False

    # Download logos
    logos_dir = LOCAL_DATA / "logos"
    logos_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for obj in mc.list_objects(MINIO_BUCKET, prefix=DATA_PREFIX + "logos/", recursive=True):
        rel = obj.object_name[len(DATA_PREFIX):]
        local_path = LOCAL_DATA / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            resp = mc.get_object(MINIO_BUCKET, obj.object_name)
            local_path.write_bytes(resp.read())
            resp.close(); resp.release_conn()
            count += 1
    if count:
        print(f"Pulled {count} new logos from MinIO")

    # Download index artifacts
    INDEX_DIR.mkdir(exist_ok=True)
    for fname in ["index.faiss", "metadata.jsonl", "embeddings.npy"]:
        try:
            resp = mc.get_object(MINIO_BUCKET, DATA_PREFIX + "index_artifacts/" + fname)
            (INDEX_DIR / fname).write_bytes(resp.read())
            resp.close(); resp.release_conn()
            print(f"Pulled index_artifacts/{fname}")
        except Exception:
            pass

    return True


def fetch_and_sync():
    """Fetch from HalalDamu API, compare with current, download new logos, rebuild index."""
    mc = get_minio()

    # Load current companies
    try:
        resp = mc.get_object(MINIO_BUCKET, DATA_PREFIX + "companies.json")
        current = json.loads(resp.read())
        resp.close(); resp.release_conn()
    except Exception:
        current = []

    current_ids = {c["company_id"] for c in current}
    current_map = {c["company_id"]: c for c in current}

    # Fetch from API
    print("Fetching from HalalDamu API...")
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    companies_raw = raw.get("companies", raw) if isinstance(raw, dict) else raw
    print(f"API returned {len(companies_raw)} companies")

    new_companies = []
    updated = 0
    for c in companies_raw:
        cid = str(c["id"])
        brand = (c.get("brand_name") or c.get("title") or "").strip()
        title = (c.get("title") or "").strip()

        logo_url = None
        fi = c.get("featured_image")
        if isinstance(fi, dict):
            logo_url = fi.get("full") or fi.get("medium") or fi.get("thumbnail")
        elif isinstance(fi, str) and fi:
            logo_url = fi

        if not logo_url or not brand:
            continue

        ext = Path(urlparse(logo_url).path).suffix or ".png"
        logo_minio_path = DATA_PREFIX + f"logos/{cid}/logo{ext}"
        logo_rel = f"logos/{cid}/logo{ext}"

        # Check if logo exists in MinIO
        logo_exists = False
        try:
            mc.stat_object(MINIO_BUCKET, logo_minio_path)
            logo_exists = True
        except Exception:
            pass

        if not logo_exists:
            try:
                r = requests.get(logo_url, timeout=15)
                r.raise_for_status()
                data = r.content
                ct = "image/jpeg" if ext.lower() in (".jpg", ".jpeg") else "image/png"
                mc.put_object(MINIO_BUCKET, logo_minio_path,
                              io.BytesIO(data), len(data), content_type=ct)
                updated += 1
            except Exception as e:
                print(f"  Failed to download logo for {cid}: {e}")
                continue

        aliases = [title] if title and title != brand else []
        new_companies.append({
            "company_id": cid,
            "company_name": brand,
            "aliases": aliases,
            "logo_paths": [logo_rel]
        })

    # Save updated companies.json to MinIO
    data = json.dumps(new_companies, ensure_ascii=False, indent=2).encode()
    mc.put_object(MINIO_BUCKET, DATA_PREFIX + "companies.json",
                  io.BytesIO(data), len(data), content_type="application/json")

    print(f"Sync complete: {len(new_companies)} companies, {updated} new logos downloaded")

    if updated > 0 or len(new_companies) != len(current):
        print("Rebuilding FAISS index...")
        rebuild_index(mc, new_companies)

    return {"total": len(new_companies), "new_logos": updated,
            "previous": len(current)}


def augment_image(img, preprocess):
    augmented = []
    for angle in [-5, 5]:
        aug = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        augmented.append(preprocess(aug).unsqueeze(0))
    for factor in [0.8, 1.2]:
        aug = ImageEnhance.Brightness(img).enhance(factor)
        augmented.append(preprocess(aug).unsqueeze(0))
    aug = ImageEnhance.Contrast(img).enhance(1.3)
    augmented.append(preprocess(aug).unsqueeze(0))
    return augmented


def rebuild_index(mc, companies):
    """Rebuild FAISS index from logos in MinIO."""
    print("Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()

    all_embeddings = []
    metadata = []

    for i, c in enumerate(companies):
        for lp in c["logo_paths"]:
            obj_name = DATA_PREFIX + lp
            try:
                resp = mc.get_object(MINIO_BUCKET, obj_name)
                img_data = resp.read()
                resp.close(); resp.release_conn()
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                tensor = preprocess(img).unsqueeze(0)
            except Exception as e:
                continue

            tensors = [tensor] + augment_image(img, preprocess)
            with torch.no_grad():
                batch = torch.cat(tensors, dim=0)
                embs = model.encode_image(batch)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                embs = embs.numpy().astype(np.float32)

            for j, emb in enumerate(embs):
                all_embeddings.append(emb)
                metadata.append({
                    "company_id": c["company_id"],
                    "company_name": c["company_name"],
                    "aug": "original" if j == 0 else f"aug_{j}"
                })

        if (i + 1) % 100 == 0:
            print(f"  Embedded {i+1}/{len(companies)}")

    if not all_embeddings:
        print("No embeddings!")
        return

    embeddings = np.stack(all_embeddings).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save locally
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    np.save(str(INDEX_DIR / "embeddings.npy"), embeddings)
    with open(INDEX_DIR / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Upload to MinIO
    for fname in ["index.faiss", "metadata.jsonl", "embeddings.npy"]:
        fdata = (INDEX_DIR / fname).read_bytes()
        mc.put_object(MINIO_BUCKET, DATA_PREFIX + "index_artifacts/" + fname,
                      io.BytesIO(fdata), len(fdata))

    print(f"Index rebuilt: {len(metadata)} vectors")


if __name__ == "__main__":
    if "--upload" in sys.argv:
        upload_local_to_minio()
    else:
        fetch_and_sync()
