#!/usr/bin/env python3
"""Rebuild FAISS index incorporating labeled uploads."""
import json, io, numpy as np, faiss
from pathlib import Path
from PIL import Image
from db import get_pg, get_image_bytes
import open_clip, torch

INDEX_DIR = Path(__file__).parent / "index_artifacts"

def main():
    # Load model
    print("Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()

    # Load original index data
    print("Loading original index...")
    orig_index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    orig_vecs = faiss.rev_swig_ptr(orig_index.get_xb(), orig_index.ntotal * orig_index.d)
    orig_vecs = orig_vecs.reshape(orig_index.ntotal, orig_index.d).copy()
    
    with open(INDEX_DIR / "metadata.jsonl") as f:
        orig_meta = [json.loads(l) for l in f]

    print(f"Original: {len(orig_meta)} vectors")

    # Load labeled uploads
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, label_company_id FROM uploads WHERE label_status IN ('confirmed','corrected') AND label_company_id IS NOT NULL")
    rows = cur.fetchall()
    cur.close()
    print(f"Labeled uploads: {len(rows)}")

    if not rows:
        print("No labeled uploads to add. Done.")
        return

    new_vecs = []
    new_meta = []
    for uid, filename, company_id in rows:
        try:
            data = get_image_bytes(filename)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            new_vecs.append(emb.numpy().flatten())
            new_meta.append({"company_id": company_id, "source": f"upload_{uid}"})
        except Exception as e:
            print(f"  Skip upload {uid}: {e}")

    if new_vecs:
        all_vecs = np.vstack([orig_vecs] + [v.reshape(1, -1) for v in new_vecs]).astype(np.float32)
        all_meta = orig_meta + new_meta
    else:
        all_vecs = orig_vecs
        all_meta = orig_meta

    # Build new index
    print(f"Building index with {len(all_meta)} vectors...")
    new_index = faiss.IndexFlatIP(all_vecs.shape[1])
    new_index.add(all_vecs)

    # Save
    faiss.write_index(new_index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metadata.jsonl", "w") as f:
        for m in all_meta:
            f.write(json.dumps(m) + "\n")

    print(f"Done! New index: {new_index.ntotal} vectors ({len(new_vecs)} added from labels)")

if __name__ == "__main__":
    main()
