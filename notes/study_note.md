# DPR 공부 노트

---

## Dual-Encoder

질문 인코더와 passage 인코더를 **따로** 두는 구조.
BERT-base 두 개를 독립적으로 초기화하며, 파라미터를 공유하지 않는다.

```
질문   → Question Encoder (BERT) → q_emb (768d)
Passage → Passage Encoder (BERT)  → p_emb (768d)

유사도 = q_emb · p_emb  (dot product)
```

### 왜 두 개로 나누는가?

Cross-encoder는 질문과 passage를 하나의 BERT에 같이 넣어 표현력이 높지만,
Wikipedia 21M passages를 미리 계산해둘 수 없어서 실시간 검색이 불가능하다.

Dual-encoder는 passage 임베딩을 **오프라인에 미리 계산**해두고,
검색 시에는 질문 임베딩과 내적 한 번으로 끝낼 수 있다.

```
[오프라인] Wikipedia 21M passages → Passage Encoder → 임베딩 → FAISS 인덱스
[온라인]   새 질문 → Question Encoder → q_emb → FAISS 내적 → Top-k 반환
```

---

## In-Batch Negative

Dual-encoder를 학습시키는 방법. 배치 안의 다른 질문의 positive passage를 negative로 재사용한다.

```
Q = (B, 768)   # B개 질문 임베딩
P = (B, 768)   # B개 positive passage 임베딩

S = Q @ P.T    # (B, B) 유사도 행렬
```

```
     p0    p1    p2    p3
q0 [0.9,  0.2,  0.1,  0.3]   ← q0의 정답은 p0 (대각선)
q1 [0.1,  0.8,  0.3,  0.2]   ← q1의 정답은 p1
q2 [0.2,  0.1,  0.7,  0.4]   ← q2의 정답은 p2
q3 [0.3,  0.2,  0.1,  0.9]   ← q3의 정답은 p3
```

- **정답**: 대각선 — QA 데이터셋에서 주어진 질문-passage 쌍
- **negative**: 같은 행의 나머지 — 다른 질문의 positive passage를 그대로 활용
- **Loss**: CrossEntropy(S, target=[0,1,...,B-1]) — 각 행에서 대각선 점수가 가장 높아지도록

유사도가 높다고 정답이 되는 게 아니라, **학습이 잘 될수록** 정답 쌍의 유사도가 높아지는 것이다.

### BM25 Hard Negative 추가

BM25 점수는 높지만 정답이 없는 passage를 각 질문에 1개씩 추가한다.
passage 행렬 뒤에 붙이기만 하면 CrossEntropy가 나머지를 전부 negative로 처리한다.

```
p_all = [p_pos_0, ..., p_pos_{B-1}, p_hard_0, ..., p_hard_{B-1}]
S = Q @ p_all.T    # (B, 2B)
target = [0, 1, ..., B-1]   # 여전히 대각선이 정답
```

배치 크기가 클수록 negative 수가 늘어나 성능이 향상된다 (논문: 배치 128 사용).

---

## BERT 입력 파라미터

토크나이저가 텍스트를 받아 세 가지 숫자 배열을 만든다.

| 파라미터 | 내용 | DPR에서 |
|---------|------|---------|
| `input_ids` | 각 토큰의 어휘 사전 인덱스 | 질문/passage 토큰 ID |
| `attention_mask` | 실제 토큰(1) vs 패딩(0) | 패딩 위치 마스킹 |
| `token_type_ids` | 문장 A(0) vs 문장 B(1) | 전부 0 (문장 하나씩 인코딩) |

`[CLS]` 토큰의 hidden state (768d)를 임베딩으로 사용한다.

---

## 전체 파이프라인

```
[학습]
NQ 데이터 (질문 + positive passage + BM25 hard negative)
    → Dual-Encoder forward
    → In-batch negative loss
    → 역전파 → 두 인코더 파라미터 업데이트

[추론]
① Passage Encoder로 Wikipedia 21M passages 임베딩 → FAISS 인덱스 구축
② Question Encoder로 질문 임베딩 → FAISS에서 Top-k 검색
③ Top-k passages → Reader → 정답 span 추출
```
