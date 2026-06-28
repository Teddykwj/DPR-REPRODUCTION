# DPR 재현 계획

## 목표

논문의 핵심 기여 두 가지를 직접 확인한다:
1. **Retriever**: In-batch negative + BM25 hard negative로 학습한 DPR이 BM25보다 Top-20 accuracy가 높은가?
2. **이해**: 각 설계 결정(negative 종류, 배치 크기, 유사도 함수)이 성능에 어떤 영향을 미치는가?

데이터셋은 **NQ(Natural Questions)** 하나에 집중한다. Reader 구현은 Retriever 재현 후 여력에 따라 결정한다.

---

## GPU 전략

GPU는 비용이 발생하므로 최대한 로컬(CPU)에서 준비하고, GPU가 필요한 작업만 모아서 한 번에 처리한다.

| 작업 | 환경 |
|------|------|
| 데이터 다운로드 / 전처리 | 로컬 CPU |
| BM25 인덱스 구축 + hard negative 추출 | 로컬 CPU |
| 모델 코드 작성 + 단위 테스트 | 로컬 CPU (소규모) |
| Dual-encoder 학습 | **GPU 렌탈** |
| Passage 임베딩 추출 | **GPU 렌탈** |
| FAISS 인덱스 구축 | GPU 렌탈 후 CPU 가능 |
| Top-k accuracy 평가 | CPU |

---

## Phase 1: 데이터 파악 (로컬) + 다운로드 스크립트 준비

실제 데이터 다운로드는 **GPU 서버에서** 한다 (Wikipedia ~14GB, 로컬에 받아도 서버에 다시 올려야 하므로 불필요).
로컬에서는 포맷을 이해하고 스크립트만 준비한다.

### 1-1. 데이터 포맷 파악

공식 DPR 레포 README와 코드를 읽어서 파일 구조를 이해한다.

**Wikipedia passages** (`psgs_w100.tsv`, ~14GB)
```
포맷: id \t text \t title
전처리: DrQA 방식, 100 words 고정 분할, passage 앞에 title 추가
```

**NQ 데이터** (`biencoder-nq-train.json`)
```json
{
  "question": "...",
  "answers": ["..."],
  "positive_ctxs": [{"title": "...", "text": "..."}],
  "negative_ctxs": [...],
  "hard_negative_ctxs": [{"title": "...", "text": "..."}]
}
```

- `negative_ctxs`: 랜덤 negative — 현재 공식 코드에서 실제로 사용하지 않음
- `hard_negative_ctxs`: BM25 상위 결과 중 정답 문자열이 없는 passage → 학습에 사용

**hard negative 흐름**

생성은 DPR 레포 밖에서 수행됨. 공식 배포 JSON에 이미 포함된 상태로 제공되며, 생성 스크립트는 레포에 없음.
직접 생성하려면 Elasticsearch/Pyserini로 BM25 인덱스를 만들어 별도 처리해야 함.

| 단계 | 위치 | 내용 |
|------|------|------|
| 생성 | 레포 외부 (Elasticsearch) | BM25 상위 결과 중 정답 문자열 없는 passage 추출 |
| 읽기 | `dpr/data/biencoder_data.py` L84 | JSON에서 통째로 읽어옴, 개수 제한 없음 |
| 선택 | `dpr/models/biencoder.py` `create_biencoder_input()` | `hard_neg_ctxs[0:num_hard_negatives]` → 1개 슬라이싱 후 배치에 합침 |

`num_hard_negatives=1`은 `conf/train/biencoder_nq.yaml`의 `hard_negatives: 1`에서 설정됨.

### 1-2. 다운로드 스크립트 작성 (`scripts/01_download_data.sh`)

GPU 서버 세팅 시 한 번에 실행할 수 있도록 준비만 해둔다.
공식 레포의 `download_data.py`를 사용한다 (prefix 기반 매칭).

```bash
# 공식 DPR 레포 클론
git clone https://github.com/facebookresearch/DPR.git
cd DPR && pip install .

# Wikipedia passages (~14GB)
python data/download_data.py --resource data.wikipedia_split.psgs_w100

# NQ retriever 학습 데이터 (train/dev, JSON 포맷)
python data/download_data.py --resource data.retriever.nq

# NQ 평가용 QA CSV (dev/test)
python data/download_data.py --resource data.retriever.qas.nq
```

### 1-3. 소규모 실험용 미니 데이터 (로컬용)

NQ json에서 처음 200개만 뽑은 파일을 로컬에 만들어둔다.
→ 모델 코드 작성 후 CPU에서 forward pass, loss 동작 확인에 사용.

---

## Phase 2: 모델 구현 (로컬)

공식 DPR 코드(`dpr/models/biencoder.py`)를 읽으면서 이해하고, 핵심 부분을 직접 구현한다.

### 2-1. Dual-Encoder (`src/models/biencoder.py`)

```python
# 구현할 것
class BertEncoder          # BERT-base 래퍼, CLS 토큰 반환
class BiEncoder            # question encoder + passage encoder
  - forward(q_ids, p_ids) → q_emb, p_emb
  - 두 인코더는 파라미터 공유 안 함
```

공식 코드와 대조할 포인트:
- CLS 토큰 추출 방식
- 두 인코더의 파라미터 초기화가 독립적인지

### 2-2. In-Batch Negative Loss (`src/models/loss.py`)

논문 핵심 수식을 그대로 구현:

```
S = Q @ P^T   → (B, B) 유사도 행렬
target = [0, 1, 2, ..., B-1]  (대각선이 positive)
loss = CrossEntropyLoss(S, target)
```

BM25 hard negative 추가 시:
- 각 질문에 hard negative 1개 → P 행렬에 추가 열로 붙임
- S = Q @ [P_gold | P_hard]^T  → (B, B+B) 또는 (B, B+1)

**공식 코드와 직접 대조**해서 내 구현과 동일한지 확인.

### 2-3. Dataset & DataLoader (`src/data/dataset.py`)

```python
class DPRDataset
  - NQ json 로드
  - 배치 구성: 질문 + positive + hard_negative 1개
  - tokenize (BERT tokenizer, max_length=256 for passage, 64 for question)

class DPRCollator
  - 배치 패딩
  - in-batch negative를 위한 행렬 구성
```

### 2-4. 단위 테스트 (`notebooks/01_model_check.ipynb`)

CPU에서 소규모로 동작 확인:
- BiEncoder forward pass shape 확인
- Loss가 초기에 log(B) 근처에서 시작하는지 (random 초기화 시 예상값)
- Loss가 몇 step 안에 감소하는지

---

## Phase 3: 학습

공식 레포의 `train_dense_encoder.py`를 Hydra 설정과 함께 사용한다.
설정 파일: `conf/train/biencoder_nq.yaml`

### 실행 명령

```bash
# 풀 학습 (8 GPU, 분산 학습)
python -m torch.distributed.launch --nproc_per_node=8 \
  train_dense_encoder.py \
  train=biencoder_nq \
  train_datasets=[nq_train] \
  dev_datasets=[nq_dev] \
  output_dir={체크포인트 저장 경로}
```

### 하이퍼파라미터

| 설정 | 논문/공식 레포 | 미니 실험 |
|------|--------------|-----------|
| Batch size | 128 (GPU당) | 32 |
| Learning rate | 1e-5 | 1e-5 |
| Epochs | 40 | 5~10 |
| Hard negative | BM25 1개 | BM25 1개 |
| Max passage length | 256 tokens | 256 tokens |
| Max question length | 64 tokens | 64 tokens |
| 검증 방식 | epoch 30까지 NLL loss, 이후 Average Rank | NLL loss |

### 학습 순서

1. **미니 실험 (GPU 렌탈 초반)**: NQ 5K + 배치 32 → loss 감소 확인
2. **풀 학습**: NQ 전체 58K + 배치 128 → 40 epochs (~1일 소요)

### 체크포인트

- 공식 레포는 매 validation마다 저장, **보통 마지막 체크포인트가 best**
- epoch ~25 이후부터 Average Rank가 25 이하로 수렴하는지 확인

---

## Phase 4: 인덱스 구축 및 평가

### 4-1. Passage 임베딩 추출

공식 레포의 `generate_dense_embeddings.py` 사용. 샤딩으로 병렬 처리 가능.

```bash
python generate_dense_embeddings.py \
  model_file={체크포인트 경로} \
  ctx_src=dpr_wiki \
  shard_id={0부터} num_shards={총 샤드 수} \
  out_file={출력 경로 prefix} \
  batch_size=128
```

- 예상 크기: 21M × 768 × 4bytes ≈ 64GB
- 단일 GPU면 shard 나눠서 순차 실행

### 4-2. FAISS 인덱스 구축 + Retrieval 평가

공식 레포의 `dense_retriever.py`가 인덱스 구축과 평가를 함께 처리한다.

```bash
python dense_retriever.py \
  model_file={체크포인트} \
  qa_dataset=nq_test \
  ctx_datatsets=[dpr_wiki] \
  encoded_ctx_files=["{임베딩 파일 glob}"] \
  out_file={결과 json 경로}
```

- 기본 인덱스: **exhaustive (flat)** — 정확도 최대
- HNSW 옵션(`indexer=hnsw`): 검색 빠르지만 인덱스 구축 오래 걸리고 RAM 많이 씀
- 재현 목적이므로 **flat 인덱스** 사용

### 4-3. 목표 수치

| | BM25 (논문) | DPR 논문 | 공식 레포 모델 | 재현 |
|-|------------|---------|--------------|------|
| Top-20 | 59.1% | 78.4% | **79.97%** | ? |
| Top-100 | 73.7% | 85.4% | **85.87%** | ? |

> 논문 수치(78.4%)와 공식 레포 제공 체크포인트 수치(79.97%)가 다소 다름.
> 재현 목표는 **논문 기준 78.4%** 이상으로 잡는다.

BM25 기준선은 Pyserini로 직접 측정.

---

## Phase 5: Ablation 실험 (여력 있을 때)

논문 Table 2를 직접 재현해서 설계 결정의 효과를 체감한다.

- [ ] Hard negative 없을 때 vs 있을 때 (BM25 1개)
- [ ] 배치 크기: 32 vs 64 vs 128
- [ ] In-batch negative 없는 표준 학습 vs 있는 학습

---

## Phase 6: Reader (추후 결정)

Retriever Top-20 accuracy 재현 후 시간/여력에 따라 결정.

- BERT-base reader 구현
- Span extraction + passage selection 동시 학습
- Exact Match 측정

---

## 참고 자료

| 자료 | 위치/링크 |
|------|----------|
| 논문 | `2004.04906v3.pdf` |
| 논문 정리 | `DPR_논문정리.md` |
| 공식 레포 요약 | `notes/official_repo_summary.md` |
| 공식 코드 | https://github.com/facebookresearch/DPR |
| Pyserini (BM25) | https://github.com/castorini/pyserini |

---

## 진행 현황

- [x] 디렉토리 구조 설정
- [ ] Phase 1: 데이터 준비
- [ ] Phase 2: 모델 구현
- [ ] Phase 3: 학습
- [ ] Phase 4: 인덱스 구축 및 평가
- [ ] Phase 5: Ablation
- [ ] Phase 6: Reader
