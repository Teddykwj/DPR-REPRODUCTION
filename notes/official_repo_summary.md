# 공식 DPR 레포 요약

> https://github.com/facebookresearch/DPR

---

## 데이터 다운로드

공식 스크립트로 모든 데이터를 받는다. prefix 기반 매칭이라 `data.*`로 전부 받을 수도 있다.

```bash
# Wikipedia passages (21M, ~14GB)
python data/download_data.py --resource data.wikipedia_split.psgs_w100

# NQ retriever 학습 데이터 (train/dev, JSON 포맷)
python data/download_data.py --resource data.retriever.nq

# NQ 평가용 QA CSV (dev/test)
python data/download_data.py --resource data.retriever.qas.nq
```

---

## 데이터 포맷

### Retriever 학습 데이터 (JSON)

```json
[
  {
    "question": "...",
    "answers": ["...", "..."],
    "positive_ctxs": [{"title": "...", "text": "..."}],
    "negative_ctxs": [...],
    "hard_negative_ctxs": [{"title": "...", "text": "..."}]
  }
]
```

- `negative_ctxs`: 랜덤 negative (현재 코드에서 실제로 사용 안 됨)
- `hard_negative_ctxs`: BM25 상위 결과 중 정답 없는 passage

---

## 실행 파이프라인 (NQ 기준)

### 1. Retriever 학습

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
train_dense_encoder.py \
  train=biencoder_nq \
  train_datasets=[nq_train] \
  dev_datasets=[nq_dev] \
  output_dir={checkpoint 저장 경로}
```

- 8 x 32GB GPU, 약 1일 소요 (40 epochs)
- epoch 30부터 Average Rank 검증으로 전환, 최종 ~25 이하 목표
- 보통 마지막 체크포인트가 best

### 2. Passage 임베딩 추출

```bash
python generate_dense_embeddings.py \
  model_file={체크포인트 경로} \
  ctx_src=dpr_wiki \
  shard_id={0부터} num_shards={총 샤드 수} \
  out_file={출력 경로 prefix}
```

- 50대의 2-GPU 서버에서 약 40분 소요
- 단일 GPU라면 shard 나눠서 순차 실행

### 3. Retrieval 평가

```bash
python dense_retriever.py \
  model_file={체크포인트} \
  qa_dataset=nq_test \
  ctx_datatsets=[dpr_wiki] \
  encoded_ctx_files=["{임베딩 파일 glob}"] \
  out_file={결과 json 경로}
```

### 4. Reader 학습

```bash
python train_extractive_reader.py \
  encoder.sequence_length=350 \
  train_files={retriever 결과 json} \
  dev_files={retriever dev 결과 json} \
  output_dir={출력 경로}
```

---

## 신규 모델 (2021년 3월)

기존 NQ 체크포인트로 DPR 인덱스를 만들어 hard negative를 다시 mining → 재학습하는 iterative 방식.

| Top-k | 기존 모델 | 신규 모델 |
|-------|----------|----------|
| 1     | 45.87    | 52.47    |
| 5     | 68.14    | 72.24    |
| 20    | 79.97    | 81.33    |
| 100   | 85.87    | 87.29    |

재현 목표는 **기존 모델 (79.97 Top-20)** 기준으로 잡는다.

```bash
# 신규 모델 학습 데이터 (기존 NQ + adversarial hard negative)
python data/download_data.py --resource data.retriever.nq-adv-hn-train
```

---

## 재현 시 참고 포인트

| 항목 | 내용 |
|------|------|
| 설정 방식 | Hydra 기반 yaml 설정 (`conf/` 디렉토리) |
| 기본 인코더 | HuggingFace BERT-base |
| 학습 설정 파일 | `conf/train/biencoder_nq.yaml` |
| 인덱스 기본값 | exhaustive (flat), HNSW는 옵션 |
| 라이선스 | CC-BY-NC 4.0 |
