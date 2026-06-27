# Dense Passage Retrieval for Open-Domain Question Answering

> Karpukhin et al., 2020 | arXiv:2004.04906v3 | Facebook AI, UW, Princeton
> GitHub: https://github.com/facebookresearch/DPR

---

## 1. 한 줄 요약 (Abstract)

BM25 같은 sparse retrieval을 **dense dual-encoder(DPR)** 로 대체했더니, 추가 사전학습 없이도 Top-20 검색 정확도가 BM25 대비 **+9~19%** 향상되고 end-to-end QA에서도 SOTA를 달성했다.

---

## 2. Introduction

### 왜 dense retrieval인가?

전통적 Retriever인 TF-IDF / BM25는 **키워드 매칭** 기반 sparse representation이다.
어휘가 다른 의미적 유사 표현을 포착하지 못한다는 근본적 한계가 있다.

**예시**: "Who is the bad guy in lord of the rings?"
- BM25: "villain Sauron" 포함 passage를 검색하기 어려움 (bad guy ≠ villain)
- DPR: "bad guy" ↔ "villain" 의미적 매칭 가능

Dense representation은 sparse와 **설계상 상보적(complementary by design)** 이다:

| 방식 | 표현 | 강점 | 약점 |
|------|------|------|------|
| BM25 | sparse, high-dim | 희귀 고유명사, 정확한 키워드 | 어휘 불일치 |
| Dense | dense, low-dim | 동의어, 패러프레이즈, 의미 유사성 | 희귀 단어 포착 어려움 |

### 선행 연구 ORQA와 그 한계

DPR 이전에 dense retrieval로 BM25를 처음 넘은 모델이 **ORQA**(Lee et al., 2019)다.
ORQA는 **Inverse Cloze Task (ICT)** 라는 방법으로 사전학습을 수행했다.

#### Inverse Cloze Task (ICT)

labeled 데이터 없이 dense retriever를 사전학습하기 위한 방법.
passage에서 문장 하나를 뽑아 "pseudo 질문"으로, 나머지를 "정답 passage"로 사용한다.

```
[원본 passage]
"The Eiffel Tower is located in Paris, France.
 It was built in 1889 and stands 330 meters tall.
 The tower attracts millions of tourists every year."

→ 문장 하나를 pseudo 질문으로 추출:

[pseudo 질문]  "It was built in 1889 and stands 330 meters tall."
[정답 passage] "The Eiffel Tower is located in Paris...attracts millions..."
```

"이 문장이 어느 passage에서 왔는가?"를 맞추도록 학습.
Wikipedia 전체를 annotation 없이 자동으로 학습 데이터로 활용할 수 있다는 장점이 있다.

**그러나 ORQA는 두 가지 약점이 있다:**

1. **ICT 사전학습 비용이 큼** — computationally intensive
2. **문장 ≠ 질문** — 일반 문장은 실제 질문의 좋은 대리자가 아님 (형태·의도가 다름)
3. **context encoder가 fine-tuning되지 않음** — 질문-정답 쌍으로 passage encoder를 업데이트하지 않아 representation이 최적이 아님

### DPR의 핵심 질문과 기여

> "ICT 같은 추가 사전학습 없이, 질문-passage 쌍만으로 더 좋은 dense retriever를 만들 수 있는가?"

**기여 1**: BERT + dual-encoder를 질문-passage 쌍으로만 fine-tuning해도 BM25를 크게 능가함을 증명  
**기여 2**: 검색 정확도 향상이 end-to-end QA 정확도 향상으로 직결됨을 검증

---

## 3. Background

### Open-Domain QA 문제 정의

특정 문서를 주지 않고, 방대한 corpus(예: 전체 Wikipedia) 전체를 대상으로 질문에 답하는 태스크.

```
질문 (q) → [Retriever] → 후보 passages (k개) → [Reader] → 정답 span
```

- corpus $C = \{p_1, p_2, \ldots, p_M\}$ (M개의 passages)
- Retriever $R: (q, C) \rightarrow C_F$, 여기서 $|C_F| = k \ll M$
- 평가 지표: **Top-k retrieval accuracy** (CF 안에 정답 span이 있는 질문의 비율)

> Retriever가 정답 passage를 못 가져오면 Reader가 아무리 좋아도 답을 못 냄.

---

## 4. Dense Passage Retriever (DPR)

### 4.1 Dual-Encoder 구조

두 개의 **독립적인 BERT(base, uncased)** 인코더:

```
E_Q(q) → d-dim vector   (question encoder)
E_P(p) → d-dim vector   (passage encoder)
```

- d = 768 (`[CLS]` token의 hidden state 사용)
- 두 인코더는 파라미터를 공유하지 않음

### 4.2 유사도 함수

$$\text{sim}(q, p) = E_Q(q)^\top E_P(p)$$

내적(dot product)을 사용.

**왜 cross-attention이 아닌가?**
cross-attention처럼 표현력이 높은 함수는 q와 p를 동시에 입력해야 하므로, 2100만 개 passage를 미리 계산해둘 수 없다. 실시간 검색이 불가능해진다.

유사도 함수는 반드시 **decomposable(분해 가능)** 해야 한다:
- passage 임베딩을 **오프라인에 미리 계산**해두고
- 검색 시에는 질문 임베딩과 **내적 한 번**으로 끝낼 수 있어야 함

**decomposable한 함수들은 사실 모두 L2의 변형:**

| 함수 | 관계 |
|------|------|
| Cosine | 단위벡터일 때 내적과 동일 |
| Mahalanobis distance | 변환된 공간에서의 L2 |
| **Dot product** | 가장 단순한 형태 → 채택 |

어차피 다 비슷하다면 가장 단순한 내적을 쓰고 encoder 학습에 집중하겠다는 전략. ablation 실험에서도 cosine, L2와 성능 차이가 거의 없음을 확인.

### 4.3 Inference

```
[오프라인]  모든 passage → E_P → 임베딩 → FAISS 인덱스 구축
[온라인]    질문 q → E_Q(q) → FAISS에서 top-k 검색 (MIPS)
```

- **FAISS**: Facebook의 고성능 dense vector 유사도 검색 라이브러리
- HNSW 인덱스 (neighbors=512, construction depth=200, search depth=128)

### 4.4 학습

encoder 학습은 본질적으로 **metric learning** 문제다 — 관련 있는 질문-passage 쌍은 벡터 공간에서 가깝게, 무관한 쌍은 멀게 배치되도록 거리 함수(encoder)를 학습하는 것.

#### Loss Function

Negative Log-Likelihood:

$$L(q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-) = -\log \frac{e^{\text{sim}(q_i,\, p_i^+)}}{e^{\text{sim}(q_i,\, p_i^+)} + \sum_{j=1}^{n} e^{\text{sim}(q_i,\, p_{i,j}^-)}}$$

- $p_i^+$: positive passage (정답 포함)
- $p_{i,j}^-$: negative passages

#### Positive vs Negative의 비대칭성

Positive는 QA 데이터셋에서 명시적으로 주어지지만, negative는 나머지 2100만 개 전부가 후보다 — "정답이 없으면 무관한 것으로 간주"하는 암묵적 가정. 어떤 negative를 골라 학습에 쓰느냐가 성능에 결정적 영향을 미친다.

#### Negative Passage 종류

| 타입 | 설명 |
|------|------|
| Random | corpus에서 무작위 sampling |
| BM25 | BM25 상위 결과 중 정답이 없는 passages (hard negative) |
| Gold | 다른 질문의 positive passage (같은 배치 내) |

#### In-Batch Negatives

배치 크기 B일 때, B×B 유사도 행렬을 계산:

```
Q = (B×d) 질문 임베딩 행렬
P = (B×d) passage 임베딩 행렬
S = Q @ P^T  →  (B×B) 유사도 행렬

대각선 (i=j): positive 쌍
나머지 (i≠j): negative 쌍  →  배치 하나에서 B²개 쌍 학습
```

- 같은 배치 안의 다른 질문의 positive passage를 negative로 재사용
- 메모리 효율적이면서 학습 예제 수를 크게 늘림
- 배치 크기 클수록 성능 향상 → 128 사용

**최종 모델**: In-batch gold negatives + BM25 hard negative 1개

#### 학습 하이퍼파라미터

| 설정 | 값 |
|------|-----|
| Batch size | 128 |
| Learning rate | 1e-5 (Adam) |
| Scheduler | Linear warmup + decay |
| Dropout | 0.1 |
| Epochs | 40 (NQ, TriviaQA, SQuAD), 100 (TREC, WQ) |

---

## 5. 실험 설정

### 5.1 Wikipedia 데이터 전처리

- 2018년 12월 English Wikipedia dump
- DrQA 전처리 코드로 정제 (표, infobox, 목록, disambiguation 페이지 제거)
- **100 words** 단위 고정 길이 분할 → **21,015,324 passages**
- 각 passage 앞에 `[title] [SEP]` 추가

### 5.2 QA 데이터셋 (5개)

| Dataset | Train (원본→실사용) | Dev | Test | 특징 |
|---------|-------------------|-----|------|------|
| Natural Questions (NQ) | 79,168 → 58,880 | 8,757 | 3,610 | 실제 Google 검색 쿼리 |
| TriviaQA | 78,785 → 60,413 | 8,837 | 11,313 | 웹에서 수집한 trivia |
| WebQuestions (WQ) | 3,417 → 2,474 | 361 | 2,032 | Freebase entity 답변 |
| CuratedTREC (TREC) | 1,353 → 1,125 | 133 | 694 | TREC QA track |
| SQuAD v1.1 | 78,713 → 70,096 | 8,886 | 10,570 | passage 보고 질문 생성 ⚠️ |

> ⚠️ **SQuAD는 Open-Domain QA에 적합하지 않음**: annotator가 passage를 먼저 보고 질문을 작성했기 때문에 질문-passage 간 lexical overlap이 매우 높다. passage 없이 질문만 던지는 open-domain 환경에서는 많은 질문이 context 없이 성립하지 않음. 그럼에도 이전 연구와의 공정한 비교를 위해 포함.

### 5.3 Positive Passage 선정

- **SQuAD, NQ**: gold context를 우리 passage pool과 매칭
- **TREC, WQ, TriviaQA**: 정답이 포함된 BM25 상위 passage를 positive로 사용 (distant supervision)

---

## 6. 실험: Passage Retrieval

### 6.1 Main Results

#### 실험 설정 정의

| 설정 | 설명 |
|------|------|
| **Single** | 각 데이터셋별로 DPR을 따로 학습 |
| **Multi** | NQ + TriviaQA + WQ + TREC 합쳐서 DPR 하나를 학습 (SQuAD 제외) |
| **BM25 + DPR** | 학습은 그대로, 검색 시 두 점수를 선형 결합하여 재정렬 |

BM25 + DPR 결합 방식:

$$\text{BM25}(q, p) + \lambda \cdot \text{sim}(q, p), \quad \lambda = 1.1$$

```
BM25 top-2000 ─┐
               ├→ 합집합 → 재정렬 → 최종 top-k
DPR  top-2000 ─┘
```

$\lambda = 1.1$은 dev set 기준으로 튜닝. BM25와 DPR의 상보성 덕분에 일부 데이터셋에서 DPR 단독보다 성능이 더 높게 나옴.

**Top-20 / Top-100 Retrieval Accuracy (%)**

| Retriever | NQ | TriviaQA | WQ | TREC | SQuAD |
|-----------|-----|---------|-----|------|-------|
| BM25 | 59.1 / 73.7 | 66.9 / 76.7 | 55.0 / 71.1 | 70.9 / 84.1 | **68.8 / 80.0** |
| DPR Single | **78.4 / 85.4** | 79.4 / 85.0 | 73.2 / 81.4 | 79.8 / 89.1 | 63.2 / 77.2 |
| DPR Multi | 79.4 / 86.0 | 78.8 / 84.7 | **75.0 / 82.9** | **89.1 / 93.9** | 51.6 / 67.6 |

- SQuAD에서만 BM25 우세 → 질문 작성 시 passage를 보고 썼기 때문에 lexical overlap이 매우 높음
- Multi 설정 시 데이터셋별 반응이 다름: TREC은 학습 데이터가 가장 적어 크게 향상, NQ/WQ는 소폭 향상, TriviaQA는 데이터 특성 차이로 오히려 소폭 하락

### 6.2 Ablation: 학습 방식 비교

**[상단 블록] 표준 1-of-N 학습 (IB=✗)**
각 질문마다 positive 1개 + 자신만의 negative n개를 따로 준비. negative 종류(random/BM25/gold)를 바꿔도 k≥20에서 성능 차이가 거의 없음.

**[중간 블록] In-Batch Negative 학습 (IB=✓)**
같은 배치 안의 다른 질문의 positive를 negative로 재사용. 동일한 gold negative 7개를 써도 표준 방식보다 성능이 크게 향상. negative가 전체 학습셋이 아닌 같은 배치에서 오기 때문에 메모리 효율적이고 학습 쌍 수가 늘어남. 배치 크기가 클수록 성능 향상.

**[하단 블록] In-Batch + BM25 Hard Negative 추가**
BM25 점수는 높지만 정답이 없는 passage를 모든 질문의 공통 hard negative로 추가. 1개 추가 시 크게 향상되지만 2개는 1개보다 더 낫지 않음.

| Type | #N | IB | Top-5 | Top-20 | Top-100 |
|------|-----|-----|-------|--------|---------|
| Random | 7 | ✗ | 47.0 | 64.3 | 77.8 |
| BM25 | 7 | ✗ | 50.0 | 63.3 | 74.8 |
| Gold | 7 | ✗ | 42.6 | 63.1 | 78.3 |
| Gold | 7 | ✓ | 51.1 | 69.1 | 80.8 |
| Gold | 31 | ✓ | 52.1 | 70.8 | 82.1 |
| Gold | 127 | ✓ | 55.8 | 73.0 | 83.1 |
| **Gold+BM25(1)** | **31+32** | **✓** | **65.0** | **77.3** | **84.4** |
| Gold+BM25(2) | 31+64 | ✓ | 64.5 | 76.4 | 84.0 |

→ 최종 모델: **In-batch gold + BM25 hard negative 1개**

### 6.3 Ablation: 학습 데이터 크기 (Sample Efficiency)

- **단 1,000개**로도 BM25를 능가
- 1k → 59k로 늘릴수록 꾸준히 성능 향상

### 6.4 Ablation: Distant Supervision 영향

gold context 대신 "정답 문자열을 포함한 BM25 상위 passage"를 positive로 사용(distant supervision)해도 top-k accuracy가 약 1점만 하락. 정교한 annotation 없이 정답 문자열만 있어도 DPR 학습이 충분히 가능함을 의미.

### 6.5 Ablation: 유사도 함수 & Loss

| Sim | Loss | Top-5 | Top-20 | Top-100 |
|-----|------|-------|--------|---------|
| **DP** | **NLL** | **66.8** | **78.1** | **85.0** |
| DP | Triplet | 65.0 | 77.2 | 84.5 |
| L2 | NLL | 64.7 | 76.1 | 83.1 |
| L2 | Triplet | 66.0 | 78.1 | 84.9 |

→ dot product + NLL이 최적. 더 복잡한 방식이 유의미하게 낫지 않음.

### 6.6 Cross-Dataset Generalization

NQ로만 학습 후 WQ, TREC에 직접 적용 (fine-tuning 없이):
- WQ: 69.9% top-20 (fine-tuned 75.0%, BM25 55.0%)
- TREC: 86.3% top-20 (fine-tuned 89.1%, BM25 70.9%)

### 6.7 Run-time 효율성

| | DPR | BM25 |
|--|-----|------|
| 검색 속도 | **995 q/s** (top-100) | 23.7 q/s/thread |
| 인덱스 구축 | 8.8h (임베딩, 8 GPU) + 8.5h (FAISS) | ~30분 (Lucene) |

---

## 7. 실험: Question Answering

### 7.1 Reader 모델 구조

```
top-k passages → BERT(base, uncased) → answer span 추출 + passage 선택
```

DPR retriever가 가져온 k개의 passage 각각을 BERT에 넣어 두 가지를 동시에 계산한다.

**① 각 토큰이 정답 span의 시작/끝일 확률**

$P_i \in \mathbb{R}^{L \times h}$ = i번째 passage의 BERT 출력 (토큰마다 768차원 벡터)

$$P_{\text{start},i}(s) = \text{softmax}(P_i\, w_{\text{start}})_s$$
$$P_{\text{end},i}(t) = \text{softmax}(P_i\, w_{\text{end}})_t$$

- $w_{\text{start}},\ w_{\text{end}} \in \mathbb{R}^h$: 학습되는 벡터
- 각 토큰 위치마다 "여기서 정답이 시작/끝날 확률" 계산
- **Span score** = $P_{\text{start},i}(s) \times P_{\text{end},i}(t)$

**② 이 passage가 정답을 포함할 확률 (passage selection)**

$$P_{\text{selected}}(i) = \text{softmax}(\hat{P}^\top w_{\text{selected}})_i$$

- $\hat{P} = [P_1^{[CLS]}, \ldots, P_k^{[CLS]}] \in \mathbb{R}^{h \times k}$: k개 passage의 [CLS] 벡터를 모은 행렬
- passage 전체를 [CLS] 토큰 하나로 대표해서 "이 passage가 정답을 담고 있을 확률" 계산

**③ 최종 답 선택**

```
passage selection score가 가장 높은 passage에서
span score가 가장 높은 (start, end) 구간을 최종 답으로 선택
```

**학습**

배치당 top-100 passages 중에서 1 positive + 23 negative를 샘플링 (총 $\tilde{m}=24$).

Loss는 두 가지를 합산:

$$L = L_{\text{span}} + L_{\text{passage}}$$

**$L_{\text{span}}$: correct span의 marginal log-likelihood**

정답 문자열이 passage 안에 여러 번 등장할 수 있으므로, 모든 정답 위치의 확률을 합산(marginalize):

$$L_{\text{span}} = -\log \sum_{\text{모든 정답 위치 }(s,t)} P_{\text{start}}(s) \times P_{\text{end}}(t)$$

예: "Sauron"이 3번, 8번 토큰에 모두 있으면 → 두 위치의 확률을 더해서 loss 계산. 하나를 정답으로 고르지 않고 전부 인정.

**$L_{\text{passage}}$: positive passage 선택 log-likelihood**

k개 passage 중 정답이 있는 passage가 선택될 확률을 높임:

$$L_{\text{passage}} = -\log P_{\text{selected}}(i^+)$$

→ "정답 passage를 고르는 것"과 "그 안에서 정답 위치를 찾는 것"을 동시에 학습.

**Inference**
- k=100 passages를 single 32GB GPU 한 배치로 처리 → latency ~20ms/question

### 7.2 End-to-End QA Results (Exact Match %)

| 모델 | NQ | TriviaQA | WQ | TREC | SQuAD |
|------|-----|---------|-----|------|-------|
| ORQA (Lee et al., 2019) | 33.3 | 45.0 | 36.4 | 30.1 | 20.2 |
| REALM (Guu et al., 2020) | 40.4 | - | 40.7 | - | - |
| **DPR Single** | **41.5** | **56.8** | 34.6 | 25.9 | 29.8 |
| **DPR Multi** | **41.5** | **56.8** | **42.4** | **49.4** | 24.1 |

**주요 결과 분석**

1. **Retriever 정확도 → QA 정확도**: SQuAD 제외 모든 데이터셋에서 DPR retriever가 가져온 passage일수록 정답 추출 정확도가 높음. 검색이 잘 될수록 QA도 잘 됨.

2. **Single vs Multi**:
   - NQ, TriviaQA (대형): Single ≈ Multi (이미 데이터가 충분)
   - WQ, TREC (소형): Multi가 명확히 유리 (다른 데이터셋이 보완)

3. **ORQA, REALM 대비**: 두 모델 모두 복잡한 사전학습 + end-to-end 학습을 사용하지만 DPR이 NQ, TriviaQA에서 앞섬. 추가 사전학습은 학습 데이터가 적을 때만 유효할 가능성이 높음.

4. **분리 학습 vs 공동 학습**: Retriever와 Reader를 따로 학습하는 pipeline 방식(41.5 EM)이 함께 학습하는 joint 방식(39.8 EM)보다 오히려 더 나음. 단순한 설계로도 충분.

5. **Inference 효율**:
   - k=100 passages를 32GB GPU 한 배치로 처리 → latency ~20ms (passage 1개일 때와 거의 동일)
   - ORQA는 passage 길이가 2~3배 길어(288 토큰 vs 100 토큰) 실질적으로 더 느림
   - k=50이 NQ에서 최적, k=10도 성능 손실 미미 (40.8 vs 41.5 EM)

---

## 8. Related Work

| 모델 | 접근 방식 | 비고 |
|------|---------|------|
| BM25 | sparse, inverted index | 키워드 매칭 강함, 의미 유사성 포착 불가 |
| ORQA | ICT 사전학습 + 공동 fine-tuning | DPR보다 비용 크고 성능 낮음 |
| REALM | 사전학습 중 passage encoder 비동기 업데이트 | 매우 복잡, 비용 큼 |
| ColBERT | BERT 위 late-interaction operator | decomposable하지만 저장 비용 큼 |
| ANCE | DPR 기반 iterative hard negative mining | DPR 이후 검색 성능 추가 향상 |
| FiD | DPR retriever + T5 reader (fusion-in-decoder) | 생성 모델과 결합 |
| RAG | DPR + BART | knowledge-intensive NLP |

---

## 9. 부록

### A. Distant Supervision

gold context 대신 BM25로 찾은 positive passage를 사용해도 성능 차이가 작음을 수치로 확인.

| 방식 | Top-1 | Top-5 | Top-20 | Top-100 |
|------|-------|-------|--------|---------|
| Gold | 44.9 | 66.8 | 78.1 | 85.0 |
| Distant Supervision | 43.9 | 65.3 | 77.1 | 84.4 |

→ 약 1점 차이. 정답 문자열만 있어도 DPR 학습이 충분히 가능.

### B. Alternative Similarity & Triplet Loss

dot product + NLL 외에 L2, cosine, triplet loss를 비교.

| Sim | Loss | Top-1 | Top-5 | Top-20 | Top-100 |
|-----|------|-------|-------|--------|---------|
| **DP** | **NLL** | **44.9** | **66.8** | **78.1** | **85.0** |
| DP | Triplet | 41.6 | 65.0 | 77.2 | 84.5 |
| L2 | NLL | 43.5 | 64.7 | 76.1 | 83.1 |
| L2 | Triplet | 42.2 | 66.0 | 78.1 | 84.9 |

- L2 ≈ dot product > cosine
- triplet loss는 NLL과 성능 차이 거의 없음
- 복잡한 함수보다 단순한 dot product + NLL이 최적

### C. Qualitative Analysis

BM25와 DPR이 검색하는 passage가 질적으로 다름.

| 질문 | BM25 | DPR |
|------|------|-----|
| "What is the body of water between England and Ireland?" | 키워드(England, Ireland)가 많이 등장하는 무관한 passage 반환 | "Irish Sea" passage 정확히 반환 (body of water ↔ sea 의미 매칭) |
| "Who plays Thoros of Myr in Game of Thrones?" | "Thoros of Myr" 희귀 고유명사 포함 passage 정확히 반환 | 엉뚱한 노르웨이 배우 passage 반환 (고유명사 포착 실패) |

- BM25 강점: 희귀 고유명사, 정확한 키워드
- DPR 강점: 의미적 유사성, 동의어/패러프레이즈
- 두 방식이 실패하는 케이스가 다름 → 상보적

### D. Joint Training of Retriever and Reader

Retriever와 Reader를 함께 학습하는 방식 실험 (ORQA 방식 참고).

- passage encoder는 고정, question encoder만 retriever+reader 합산 loss로 업데이트
- FAISS HNSW 인덱스를 그대로 사용 (reindexing 없이)
- 배치 크기 16, 질문당 top-100 passages로 retriever loss 계산
- Reader는 질문당 24개 passages 사용

결과: joint training 39.8 EM = pipeline 방식 41.5 EM보다 낮음.
→ retriever와 reader를 분리해서 각각 충분히 학습하는 것이 더 효과적.

---

## 10. 핵심 인사이트 & 재현 체크리스트

### 재현 시 중요 포인트

| 항목 | 값 |
|------|-----|
| 인코더 | 독립적인 BERT-base × 2, CLS 토큰 사용 |
| 유사도 | dot product |
| Negative | In-batch gold + BM25 hard negative **1개** |
| 배치 크기 | 128 (클수록 좋음) |
| Passage 길이 | 100 words (고정), `title [SEP]` 추가 |
| 인덱스 | FAISS HNSW |
| 최소 학습 데이터 | 1,000개로도 BM25 능가 |
| SQuAD | lexical overlap 높아 BM25가 유리한 특수 케이스 |

### 체크리스트

- [ ] Wikipedia dump 전처리 (DrQA 방식, 100 words 분할, 21M passages)
- [ ] 5개 QA 데이터셋 준비 및 positive passage 선정
- [ ] BERT-base dual-encoder 구현 (CLS 토큰 출력)
- [ ] In-batch negative + BM25 hard negative 1개 학습 루프
- [ ] FAISS HNSW 인덱스 구축
- [ ] Top-k retrieval accuracy 측정
- [ ] Reader (BERT-base) 학습 및 Exact Match 측정
