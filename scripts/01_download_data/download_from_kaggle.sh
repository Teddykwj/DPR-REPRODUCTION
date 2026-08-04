#!/usr/bin/env bash
# Kaggle에 이미 올려둔 데이터셋(teddykwj/dpr-reproduction-data)을 GPU 서버로 받는다.
# 공식 DPR 레포의 download_data.py(오래된 setup.py로 인한 setuptools 충돌) 대신 사용.
#
# repo 루트에서 실행. Phase 3(학습)만 필요하면 기본값(NQ json만, Wikipedia 스킵)으로 충분하다.
#   bash scripts/01_download_data/download_from_kaggle.sh              # NQ json만 (기본)
#   SKIP_WIKI=0 bash scripts/01_download_data/download_from_kaggle.sh  # Wikipedia(~14GB) 포함 전체
#
# 인증 (둘 중 하나):
#   1) 환경변수 (kaggle.json 업로드 없이 바로 실행 가능):
#      KAGGLE_USERNAME=<username> KAGGLE_KEY=<key> bash scripts/01_download_data/download_from_kaggle.sh
#   2) ~/.kaggle/kaggle.json 파일 (kaggle.com/settings → API → Create New Token)
set -euo pipefail

SKIP_WIKI="${SKIP_WIKI:-1}"
DATASET="teddykwj/dpr-reproduction-data"

if ! command -v kaggle >/dev/null 2>&1; then
    pip install kaggle -q
fi

if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    echo "환경변수(KAGGLE_USERNAME/KAGGLE_KEY)로 인증"
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
    chmod 600 "$HOME/.kaggle/kaggle.json"
else
    echo "에러: Kaggle 인증 정보가 없습니다. 아래 중 하나를 준비하세요."
    echo "  1) 환경변수: KAGGLE_USERNAME=<username> KAGGLE_KEY=<key> bash $0"
    echo "  2) ~/.kaggle/kaggle.json 업로드 (kaggle.com/settings → API → Create New Token)"
    exit 1
fi

if [ "$SKIP_WIKI" = "1" ]; then
    echo "NQ 학습 데이터만 다운로드 (Wikipedia 스킵)..."
    mkdir -p data/nq
    kaggle datasets download -d "$DATASET" -f data/nq/biencoder-nq-train.json -p data/nq --force
    kaggle datasets download -d "$DATASET" -f data/nq/biencoder-nq-dev.json   -p data/nq --force
    # 개별 파일 다운로드는 zip으로 올 수 있어 방어적으로 풀어준다
    (cd data/nq && for f in *.zip; do [ -f "$f" ] && unzip -o "$f" && rm "$f"; done || true)
else
    echo "전체 데이터셋 다운로드 (Wikipedia 포함, ~14GB)..."
    kaggle datasets download -d "$DATASET" -p . --unzip
fi

echo "Done."
find data -maxdepth 2
