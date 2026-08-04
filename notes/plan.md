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

### 1-2. 다운로드 스크립트 작성 (`scripts/01_download_data/`)

GPU 서버 세팅 시 한 번에 실행할 수 있도록 준비만 해둔다.
공식 레포의 `download_data.py`를 데이터 다운로드 용도로만 사용한다 (prefix 기반 매칭) — 모델 코드와 무관하므로 여기서만 예외적으로 공식 스크립트를 그대로 활용.

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

### 1-3. 소규모 실험용 미니 데이터

별도 파일을 미리 만들어두지 않는다. 대신:
- **모델 동작 확인** (`scripts/02_model_impl/model_check.ipynb`): 하드코딩한 더미 문장 4개로 shape/loss 확인 (NQ 데이터 불필요)
- **미니 학습** (`scripts/03_train/train_kaggle.ipynb`): `MAX_SAMPLES` 값으로 NQ train json을 런타임에 슬라이싱 (`MINI=True` → 5,000개)

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
class NQDataset
  - NQ json 로드
  - __getitem__: 질문 + positive 1개(랜덤) + hard_negative 1개 반환
  - tokenize (BERT tokenizer, max_length=256 for passage, 64 for question)

class DPRCollator
  - 배치 패딩
  - in-batch negative를 위한 행렬 구성
```

### 2-4. 단위 테스트 (`scripts/02_model_impl/model_check.ipynb`)

CPU에서 소규모로 동작 확인:
- BiEncoder forward pass shape 확인
- Loss가 초기에 log(B) 근처에서 시작하는지 (random 초기화 시 예상값)
- Loss가 몇 step 안에 감소하는지

---

## Phase 3: 학습

**Phase 2에서 직접 구현한 `BiEncoder` + `in_batch_negative_loss`를 그대로 사용해 직접 학습 루프를 돈다.**
공식 레포의 `train_dense_encoder.py`(Hydra 기반)는 사용하지 않는다 — Phase 4에서 만들 passage/question 임베딩이 이 체크포인트와 같은 벡터 공간을 공유해야 하므로, 학습부터 인덱스 구축까지 반드시 동일한 모델 코드로 이어져야 한다.

학습 스크립트: `scripts/03_train/train.py` (GPU 렌탈 서버용 CLI 스크립트)
- `python scripts/03_train/train.py --mini` → sanity check (NQ 5K, batch 32, 5 epochs)
- `python scripts/03_train/train.py` → 풀 학습 (NQ 전체, batch 128, 40 epochs)
- `--batch-size`로 GPU 메모리에 맞게 즉석 조정 가능

> Kaggle 무료 GPU(T4/P100, ~14.5GB)에서는 batch 128이 AMP를 적용해도 OOM이 나서(질문+positive+hard negative 3-way forward × BERT-base 2개 구조가 메모리를 많이 씀), VRAM이 더 큰 렌탈 GPU(A100 40GB 등)로 전환. `scripts/03_train/train_kaggle.ipynb`는 Kaggle에서 짧은 sanity check 용도로 남겨둠.

### 학습 루프 구조

```python
model     = BiEncoder()
optimizer = Adam(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(...)

for epoch in range(NUM_EPOCHS):
    for batch in train_loader:  # NQDataset + DPRCollator
        q_emb, p_emb = model(질문, positive)
        h_emb, _     = model(질문, hard_negative)  # passage_encoder 재사용
        loss = in_batch_negative_loss(q_emb, p_emb, h_emb)
        loss.backward(); optimizer.step(); scheduler.step()
    # dev_loader로 검증 loss 측정
```

### 하이퍼파라미터

| 설정 | 논문 기준 (MINI=False) | 미니 실험 (MINI=True) |
|------|--------------|-----------|
| Batch size | 128 | 32 |
| Learning rate | 1e-5 | 1e-5 |
| Epochs | 40 | 5 |
| 학습 샘플 수 | NQ 전체 (~58,880) | 5,000 |
| Hard negative | BM25 1개 | BM25 1개 |
| Max passage length | 256 tokens | 256 tokens |
| Max question length | 64 tokens | 64 tokens |
| 검증 방식 | dev set NLL loss | dev set NLL loss |

> 공식 레포는 epoch 30 이후 Average Rank로 검증하지만, 재현 스코프에서는 NLL loss만으로 단순화한다.

### 학습 순서

1. **미니 실험 (GPU 렌탈 초반)**: `MINI=True` (NQ 5K, 배치 32, 5 epochs) → loss가 log(B) 근처에서 시작해 감소하는지 확인
2. **풀 학습**: `MINI=False` (NQ 전체 58K, 배치 128, 40 epochs, ~1일 소요)

### 체크포인트

- `outputs/checkpoints/best.pt`에 `model.state_dict()` 저장 (question_encoder, passage_encoder 파라미터 모두 포함)
- dev loss가 가장 낮은 epoch마다 덮어써서 best만 유지 (`train_kaggle.ipynb`에 구현 완료)

---

## Phase 4: 인덱스 구축 및 평가

**Phase 3에서 학습한 `BiEncoder` 체크포인트를 그대로 로드해서 사용한다.**
공식 레포의 `generate_dense_embeddings.py` / `dense_retriever.py` 대신 `src/retriever/`에 직접 구현한다 — 직접 학습한 모델의 벡터 공간과 어긋나지 않으려면 임베딩 생성·검색 모두 같은 `question_encoder`/`passage_encoder`를 써야 하기 때문 (공식 스크립트는 공식 레포 자체 모델 클래스/체크포인트 포맷을 전제로 하므로 그대로 못 씀).

### 4-1. Passage 임베딩 추출 (`src/retriever/embed.py`)

```python
model = BiEncoder()
model.load_state_dict(torch.load(checkpoint_path))
passage_encoder = model.passage_encoder  # 이 부분만 사용

# 21M wikipedia passage를 배치 단위로 인코딩 → (21M, 768)
```

- 예상 크기: 21M × 768 × 4bytes ≈ 64GB
- 단일 GPU면 샤딩해서 `outputs/embeddings/`에 여러 파일로 나눠 저장 후 순차 실행

### 4-2. FAISS 인덱스 구축 (`src/retriever/index.py`)

```python
import faiss
index = faiss.IndexFlatIP(768)   # dot product 기반 exhaustive 검색
index.add(passage_embeddings)    # (21M, 768)
```

- **exhaustive (flat)** 인덱스 사용 — 정확도 최대, 재현 목적에 부합
- HNSW는 검색은 빠르지만 근사 검색이라 정확한 재현 비교에는 flat이 더 적합

### 4-3. Retrieval 평가 (`src/retriever/evaluate.py`)

```python
question_encoder = model.question_encoder  # 같은 체크포인트, 다른 절반

q_emb = question_encoder(nq_test_questions)      # (N, 768)
D, I = index.search(q_emb, k=100)                # FAISS top-k 검색
# 정답 문자열이 top-k passage 안에 포함되는 비율 = Top-k accuracy
```

### 4-4. 목표 수치

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
