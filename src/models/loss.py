import torch
import torch.nn as nn
import torch.nn.functional as F


def in_batch_negative_loss(
    q_emb: torch.Tensor,            # (B, 768)
    p_emb: torch.Tensor,            # (B, 768)
    hard_neg_emb: torch.Tensor | None = None,  # (B, 768) or None
) -> torch.Tensor:
    # hard negative가 있으면 passage 행렬에 붙임: (B, 768) → (2B, 768)
    if hard_neg_emb is not None:
        p_all = torch.cat([p_emb, hard_neg_emb], dim=0)  # (2B, 768)
    else:
        p_all = p_emb  # (B, 768)

    # 유사도 행렬: (B, B) 또는 (B, 2B)
    # S[i][j] = q_i · p_j
    S = q_emb @ p_all.T

    # 정답은 대각선: i번째 질문의 정답은 i번째 passage
    target = torch.arange(q_emb.size(0), device=q_emb.device)

    return F.cross_entropy(S, target)
