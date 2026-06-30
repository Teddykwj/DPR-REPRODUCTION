import torch
import torch.nn as nn
from transformers import BertModel


class BertEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        token_type_ids: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, 768)
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] 토큰의 hidden state를 임베딩으로 사용
        return output.last_hidden_state[:, 0, :]


class BiEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        # 두 인코더는 파라미터를 공유하지 않음
        self.question_encoder = BertEncoder(model_name)
        self.passage_encoder = BertEncoder(model_name)

    def forward(
        self,
        q_input_ids: torch.Tensor,       # (B, L_q)
        q_attention_mask: torch.Tensor,
        q_token_type_ids: torch.Tensor,
        p_input_ids: torch.Tensor,       # (B, L_p)
        p_attention_mask: torch.Tensor,
        p_token_type_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_emb = self.question_encoder(q_input_ids, q_attention_mask, q_token_type_ids)
        p_emb = self.passage_encoder(p_input_ids, p_attention_mask, p_token_type_ids)
        return q_emb, p_emb  # (B, 768), (B, 768)
