#!/usr/bin/env bash
# GPU 서버에서 실행. 공식 DPR 레포의 download_data.py를 사용한다.
set -euo pipefail

DATA_DIR="${1:-./data}"

# ── 1. 공식 DPR 레포 클론 ──────────────────────────────────────────────────────
if [ ! -d "DPR" ]; then
    git clone https://github.com/facebookresearch/DPR.git
fi
cd DPR
pip install -e . -q

# ── 2. Wikipedia passages (psgs_w100.tsv, ~14GB) ──────────────────────────────
echo "[1/3] Downloading Wikipedia passages..."
python data/download_data.py \
    --resource data.wikipedia_split.psgs_w100 \
    --output_dir "${DATA_DIR}/wikipedia"

# ── 3. NQ retriever 학습 데이터 (train/dev, JSON 포맷) ────────────────────────
echo "[2/3] Downloading NQ retriever data..."
python data/download_data.py \
    --resource data.retriever.nq \
    --output_dir "${DATA_DIR}/nq"

# ── 4. NQ 평가용 QA CSV (test) ────────────────────────────────────────────────
echo "[3/3] Downloading NQ QA CSVs..."
python data/download_data.py \
    --resource data.retriever.qas.nq \
    --output_dir "${DATA_DIR}/nq"

echo "Done. Data saved to ${DATA_DIR}"
