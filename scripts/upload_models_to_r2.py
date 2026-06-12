"""Upload all trained .pkl model files to Cloudflare R2.

Run once after training. After this, model_loader.py will pull from R2 if
the local cache is missing.

Usage:
    cd SmogAlert-PK
    python scripts/upload_models_to_r2.py
"""

import os
import sys
from pathlib import Path

# Load .env from repo root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[1] / ".env")

import boto3
from botocore.exceptions import ClientError

MODELS_DIR = Path(__file__).parents[1] / "models"
BUCKET = os.getenv("CLOUDFLARE_R2_BUCKET", "smogalert-pk")
ENDPOINT = os.getenv("CLOUDFLARE_R2_ENDPOINT", "")
ACCESS_KEY = os.getenv("CLOUDFLARE_R2_ACCESS_KEY", "")
SECRET_KEY = os.getenv("CLOUDFLARE_R2_SECRET_KEY", "")


def get_r2_client():
    if not (ENDPOINT and ACCESS_KEY and SECRET_KEY):
        print("ERROR: R2 credentials not set in .env")
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="auto",
    )


def upload_models():
    client = get_r2_client()
    pkl_files = sorted(MODELS_DIR.glob("*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {MODELS_DIR}")
        sys.exit(1)

    print(f"Uploading {len(pkl_files)} model files to R2 bucket '{BUCKET}'...\n")
    ok = 0
    failed = 0

    for path in pkl_files:
        key = f"models/{path.name}"
        try:
            client.upload_file(str(path), BUCKET, key)
            size_kb = path.stat().st_size // 1024
            print(f"  ✓ {path.name:<50} {size_kb:>6} KB  →  {key}")
            ok += 1
        except ClientError as exc:
            print(f"  ✗ {path.name}: {exc}")
            failed += 1

    print(f"\nDone. {ok} uploaded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    upload_models()
