#!/bin/bash
set -e

# On first run, upload local data to MinIO if not already there
if [ -n "$MINIO_ENDPOINT" ]; then
  echo "Checking MinIO data..."
  python3 -c "
from sync_companies import get_minio, DATA_PREFIX, MINIO_BUCKET
mc = get_minio()
try:
    mc.stat_object(MINIO_BUCKET, DATA_PREFIX + 'companies.json')
    print('MinIO data already exists, skipping upload')
except:
    print('Uploading local data to MinIO...')
    from sync_companies import upload_local_to_minio
    upload_local_to_minio()
" 2>&1 || echo "MinIO bootstrap check failed, continuing..."
fi

exec uvicorn app:app --host 0.0.0.0 --port 8090
