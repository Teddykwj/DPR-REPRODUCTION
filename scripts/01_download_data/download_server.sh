#!/usr/bin/env bash
# GPU 서버에서 실행. 공식 DPR 레포의 download_data.py를 사용한다.
# Phase 3(학습)만 필요하면: SKIP_WIKI=1 ./download_server.sh
# Phase 4(임베딩 추출)까지 필요하면: ./download_server.sh (Wikipedia 21M passage, ~14GB 포함)
set -euo pipefail

DATA_DIR="${1:-./data}"
SKIP_WIKI="${SKIP_WIKI:-0}"

# ── 1. 공식 DPR 레포 클론 ──────────────────────────────────────────────────────
if [ ! -d "DPR" ]; then
    git clone https://github.com/facebookresearch/DPR.git
fi
cd DPR
# pip install -e .는 최신 setuptools의 엄격해진 flat-layout 자동 탐지 때문에
# 오래된 이 레포(setup.py가 packages를 명시하지 않음)에서 실패한다.
# 레거시 경로(python setup.py develop)는 그 검증을 타지 않아 우회된다.
python setup.py develop -q

# ── 2. Wikipedia passages (psgs_w100.tsv, ~14GB) ──────────────────────────────
if [ "$SKIP_WIKI" = "1" ]; then
    echo "[1/3] Skipping Wikipedia passages (SKIP_WIKI=1)"
else
    echo "[1/3] Downloading Wikipedia passages..."
    python data/download_data.py \
        --resource data.wikipedia_split.psgs_w100 \
        --output_dir "${DATA_DIR}/wikipedia"
fi

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
