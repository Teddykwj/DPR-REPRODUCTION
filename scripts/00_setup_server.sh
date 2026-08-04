#!/usr/bin/env bash
# 새로 렌탈한 GPU 서버에서 최초 1회 실행. repo clone + 의존성 설치 + 학습용 데이터 다운로드.
#
# 사전 준비: Kaggle 인증 (둘 중 하나)
#   1) 환경변수: KAGGLE_USERNAME=<username> KAGGLE_KEY=<key> bash scripts/00_setup_server.sh
#   2) ~/.kaggle/kaggle.json 업로드 (kaggle.com/settings → API → Create New Token)
#
# 사용법 (서버에 SSH 접속 후, repo가 아직 없는 빈 디렉토리에서):
#   curl -sSL https://raw.githubusercontent.com/Teddykwj/dpr-reproduction/main/scripts/00_setup_server.sh | bash
# 이미 clone된 repo 안에서 실행해도 동작한다:
#   bash scripts/00_setup_server.sh
#
# Wikipedia(21M passages, ~14GB)까지 받으려면 Phase 4 진행 시 별도로:
#   SKIP_WIKI=0 bash scripts/01_download_data/download_from_kaggle.sh
set -euo pipefail

REPO_URL="https://github.com/Teddykwj/dpr-reproduction.git"

if [ -f "src/models/biencoder.py" ]; then
    echo "이미 repo 내부에서 실행 중 — clone 생략, 최신 커밋으로 pull"
    git pull
else
    if [ -d "dpr-reproduction/.git" ]; then
        echo "기존 clone 발견 — 최신 커밋으로 pull"
        cd dpr-reproduction
        git pull
    else
        git clone "$REPO_URL" dpr-reproduction
        cd dpr-reproduction
    fi
fi

pip install -r requirements.txt -q

# Kaggle에 이미 올려둔 데이터셋을 받는다 (공식 DPR 레포 download_data.py는 setuptools 문제로 불안정).
# Phase 3(학습)만 필요하므로 Wikipedia는 기본 스킵.
# 사전에 서버의 ~/.kaggle/kaggle.json 이 준비돼 있어야 한다.
bash scripts/01_download_data/download_from_kaggle.sh

echo ""
echo "Setup 완료. 학습 시작: python scripts/03_train/train.py --mini"
