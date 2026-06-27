# DPR Reproduction

Karpukhin et al., 2020 "Dense Passage Retrieval for Open-Domain Question Answering" 재현 프로젝트.

논문: https://arxiv.org/abs/2004.04906  
공식 코드: https://github.com/facebookresearch/DPR

## 디렉토리 구조

```
dpr-reproduction/
├── data/
│   ├── wikipedia/      # Wikipedia dump 전처리 결과 (21M passages)
│   ├── nq/             # Natural Questions 데이터셋
│   └── bm25/           # BM25 hard negative 사전 계산 결과
├── src/
│   ├── data/           # 데이터 전처리 및 Dataset 클래스
│   ├── models/         # Dual-encoder, Reader 구현
│   ├── retriever/      # FAISS 인덱스, BM25 검색
│   └── utils/          # 공통 유틸 (로깅, 평가 지표 등)
├── scripts/            # 전처리 / 학습 / 평가 실행 스크립트
├── configs/            # 하이퍼파라미터 설정 파일
├── notebooks/          # 분석 및 시각화용 Jupyter Notebook
└── outputs/
    ├── checkpoints/    # 학습된 모델 체크포인트
    ├── embeddings/     # Passage 임베딩 벡터
    └── results/        # 검색/QA 평가 결과
```

## 재현 순서

1. **데이터 준비** (`scripts/01_preprocess_wikipedia.py`, `scripts/02_prepare_nq.py`)
2. **BM25 hard negative 계산** (`scripts/03_build_bm25_negatives.py`)
3. **Dual-encoder 학습** (`scripts/04_train_retriever.py`)
4. **Passage 임베딩 추출 + FAISS 인덱스 구축** (`scripts/05_build_index.py`)
5. **Retrieval 평가** (`scripts/06_evaluate_retriever.py`)
6. **Reader 학습 및 평가** (`scripts/07_train_reader.py`)

## 목표 수치 (NQ, Top-20 accuracy)

| Retriever | 논문 | 재현 |
|-----------|------|------|
| BM25      | 59.1% | - |
| DPR Single | 78.4% | - |
