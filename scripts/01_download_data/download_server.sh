#!/usr/bin/env bash
# GPU 서버에서 실행. 공식 DPR 레포의 download_data.py를 사용한다.
# Phase 3(학습)만 필요하면: SKIP_WIKI=1 ./download_server.sh
# Phase 4(임베딩 추출)까지 필요하면: ./download_server.sh (Wikipedia 21M passage, ~14GB 포함)
set -euo pipefail

DATA_DIR="${1:-./data}"
SKIP_WIKI="${SKIP_WIKI:-0}"

# 이후 DPR/ 로 cd하므로, 상대경로가 엉뚱한 곳(DPR/data)을 가리키지 않도록 절대경로로 고정
mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

# ── 1. 공식 DPR 레포 클론 ──────────────────────────────────────────────────────
if [ ! -d "DPR" ]; then
    git clone https://github.com/facebookresearch/DPR.git
fi
cd DPR
# pip install -e . / python setup.py develop 둘 다 최신 setuptools의 엄격해진
# flat-layout 자동 탐지(오래된 이 레포의 setup.py가 packages를 명시 안 함) 때문에 실패한다.
# 패키지 설치 자체를 생략하고, PYTHONPATH로 dpr 모듈만 바로 import 가능하게 만든다.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

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
