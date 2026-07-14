#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/dist/hf-space-backend"

echo "Preparing optional Hugging Face Space backend fallback package..."
echo "Output: ${OUT_DIR}"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

cp "${ROOT_DIR}/deployment/huggingface/README.md" "${OUT_DIR}/README.md"
cp "${ROOT_DIR}/backend/Dockerfile" "${OUT_DIR}/Dockerfile"
cp "${ROOT_DIR}/backend/.dockerignore" "${OUT_DIR}/.dockerignore"
cp "${ROOT_DIR}/backend/requirements.txt" "${OUT_DIR}/requirements.txt"

rsync -a \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  "${ROOT_DIR}/backend/api" \
  "${OUT_DIR}/"

rsync -a \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='data/' \
  --exclude='testing/' \
  --exclude='training/' \
  --exclude='evaluation/generate_awt_frontend_v2_report.py' \
  --exclude='evaluation/*.docx' \
  --exclude='evaluation/*.html' \
  --exclude='evaluation/*.pdf' \
  --exclude='evaluation/awt_report_assets/' \
  --exclude='evaluation/finbert_intent_v3_eval/' \
  --exclude='evaluation/paper_docx_assets/' \
  --exclude='evaluation/better_than_fin/results/' \
  --exclude='*.pt' \
  --exclude='*.bin' \
  --exclude='*.safetensors' \
  "${ROOT_DIR}/backend/financial_pragmatic_ai" \
  "${OUT_DIR}/"

cat <<EOF

Optional Hugging Face Space backend fallback package is ready:
  ${OUT_DIR}

Next steps:
  1. Create or clone the Hugging Face Space repo.
  2. Copy the contents of ${OUT_DIR} into that Space repo.
  3. Commit and push to Hugging Face.
  4. Validate:
     curl https://sarconarco-financial-pragmatic-ai-backend.hf.space/health
     curl https://sarconarco-financial-pragmatic-ai-backend.hf.space/version

EOF
