#!/usr/bin/env python3
"""Fetch companies from HalalDamu API and download logos."""
import json, os, requests, time
from pathlib import Path
from urllib.parse import urlparse

API_URL = "https://halaldamu.kz/wp-json/map/v1/active-companies?lang=kz&show_all=1"
DATA_DIR = Path(__file__).parent / "data"
LOGOS_DIR = DATA_DIR / "logos"

def main():
    print("Fetching companies...")
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    companies_raw = raw.get("companies", raw) if isinstance(raw, dict) else raw
    
    companies = []
    failed = 0
    
    for i, c in enumerate(companies_raw):
        cid = str(c["id"])
        brand = (c.get("brand_name") or c.get("title") or "").strip()
        title = (c.get("title") or "").strip()
        
        # Get logo URL
        logo_url = None
        fi = c.get("featured_image")
        if isinstance(fi, dict):
            logo_url = fi.get("full") or fi.get("medium") or fi.get("thumbnail")
        elif isinstance(fi, str) and fi:
            logo_url = fi
        
        if not logo_url or not brand:
            failed += 1
            continue
        
        # Download logo
        logo_dir = LOGOS_DIR / cid
        logo_dir.mkdir(parents=True, exist_ok=True)
        
        ext = Path(urlparse(logo_url).path).suffix or ".png"
        logo_path = logo_dir / f"logo{ext}"
        
        if not logo_path.exists():
            try:
                r = requests.get(logo_url, timeout=15)
                r.raise_for_status()
                logo_path.write_bytes(r.content)
            except Exception as e:
                print(f"  [{i}] Failed {cid}: {e}")
                failed += 1
                continue
        
        aliases = [title] if title and title != brand else []
        companies.append({
            "company_id": cid,
            "company_name": brand,
            "aliases": aliases,
            "logo_paths": [str(logo_path.relative_to(DATA_DIR))]
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(companies_raw)}...")
    
    out = DATA_DIR / "companies.json"
    out.write_text(json.dumps(companies, ensure_ascii=False, indent=2))
    print(f"Done: {len(companies)} companies saved, {failed} failed")

if __name__ == "__main__":
    main()
