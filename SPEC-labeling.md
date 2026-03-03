# Logo Recognition — Data Collection & Labeling Admin

## Overview
Add upload persistence and a labeling admin panel to the existing logo recognition PoC.
Every photo submitted via POST /match gets saved. Admin users can label photos to improve the model.

## Infrastructure
- **MinIO**: `minio-api.adntgv.com` (internal docker: `minio-qg4wgcogk8484gwow884s4go:9000`)
  - Root user: `aidyn.torgayev@gmail.com`
  - Root password: `Amx7btMEvKeMytP!`
  - Bucket: `logo-uploads` (create if not exists)
- **PostgreSQL**: Instance `f8sg4sww0wocco80wgw84o8o` (port 5432 inside docker network)
  - User: `admin`, Password: `shared_pg_2026`, DB: `logo_recognition` (already created)
- **App**: `logo-recognition` container on coolify network, port 8090
- **Domain**: `logo.adntgv.com`

## Database Schema (PostgreSQL)

```sql
CREATE TABLE IF NOT EXISTS uploads (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,          -- UUID.jpg in MinIO
    original_name TEXT,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    file_size_kb INTEGER,
    -- Model predictions at upload time
    predictions JSONB,               -- [{company_id, company_name, final_score}, ...]
    top_prediction_id TEXT,          -- company_id of top prediction
    top_prediction_score FLOAT,
    -- Labeling
    label_company_id TEXT,           -- NULL = unlabeled
    label_status TEXT DEFAULT 'unlabeled', -- unlabeled | confirmed | corrected | unknown
    labeled_by TEXT,
    labeled_at TIMESTAMPTZ
);

CREATE INDEX idx_uploads_status ON uploads(label_status);
CREATE INDEX idx_uploads_label ON uploads(label_company_id);

CREATE TABLE IF NOT EXISTS admin_users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);
```

## Changes to app.py

### POST /match (modify existing)
After computing predictions:
1. Resize image to max 512px, save as JPEG quality 85
2. Generate UUID filename
3. Upload to MinIO `logo-uploads/{yyyy}/{mm}/{uuid}.jpg`
4. Insert row into `uploads` table with predictions JSON
5. Return response as before (no breaking changes)

### New endpoints

#### Auth
- `POST /api/login` — username/password → JWT token (simple, HS256, 24h expiry)
- All /api/admin/* endpoints require `Authorization: Bearer <token>`

#### Admin API
- `GET /api/admin/uploads?status=unlabeled&page=1&limit=20` — paginated list
- `GET /api/admin/uploads/:id` — single upload with full predictions
- `PATCH /api/admin/uploads/:id` — `{label_company_id, label_status}` — set label
- `GET /api/admin/stats` — `{total, unlabeled, confirmed, corrected, unknown, accuracy}`
- `GET /api/admin/companies?q=search` — search companies by name (for "other company" flow)

#### Image serving
- `GET /api/uploads/:id/image` — proxy from MinIO (or return presigned URL redirect)

### Admin UI
Single HTML page at `/admin` (served as static file, like current index.html).

**Layout:**
- Top bar: stats (X unlabeled / Y total / Z% model accuracy) + logout
- Filter tabs: All | Unlabeled | Confirmed | Corrected | Unknown
- Grid of cards, each card:
  - Left: image thumbnail (click to enlarge)
  - Right: 
    - Top-5 predictions as buttons with scores (click = confirm that prediction)
    - Search input for "other company" → autocomplete from companies list
    - "Unknown / Not in database" button
    - Timestamp
- Pagination at bottom

**Tech:** Vanilla HTML/CSS/JS (no framework needed for this). Fetch API for requests.

### Admin user creation
- CLI command or script: `python create_admin.py <username> <password>`
- Uses bcrypt for hashing
- Create default: username=`admin`, password=`halaldamu2026`

## Dependencies to add
```
psycopg2-binary
minio
PyJWT
bcrypt
```

## Docker changes
- Add env vars to container:
  - `DATABASE_URL=postgresql://admin:shared_pg_2026@f8sg4sww0wocco80wgw84o8o:5432/logo_recognition`
  - `MINIO_ENDPOINT=minio-qg4wgcogk8484gwow884s4go:9000`
  - `MINIO_ACCESS_KEY=aidyn.torgayev@gmail.com`
  - `MINIO_SECRET_KEY=Amx7btMEvKeMytP!`
  - `MINIO_BUCKET=logo-uploads`
  - `JWT_SECRET=<generate random>`
- Container must be on coolify network to reach MinIO and PostgreSQL

## Index Rebuild Script
`rebuild_index.py`:
1. Load all uploads with `label_status IN ('confirmed', 'corrected')`
2. Download images from MinIO
3. Compute embeddings with OpenCLIP
4. Merge with original logo embeddings
5. Build new FAISS index
6. Save to `index_artifacts/`
7. Print stats

Can be run manually inside container or triggered from admin UI later.

## Important Notes
- Do NOT break existing POST /match API contract — same response format
- MinIO internal endpoint (docker network): `minio-qg4wgcogk8484gwow884s4go:9000` with `secure=False`
- Coolify FQDN: use `http://` only, never `https://`
- The app currently bakes data into the Docker image — keep that for logos, but uploads go to MinIO
- Test that /match still works after changes
