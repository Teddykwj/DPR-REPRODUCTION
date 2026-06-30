import torch
from transformers import BertTokenizerFast


class DPRCollator:
    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        q_max_length: int = 64,
        p_max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length

    def __call__(self, samples: list[dict]) -> dict:
        questions    = [s["question"]      for s in samples]
        positives    = [s["positive"]      for s in samples]
        hard_negs    = [s["hard_negative"] for s in samples]

        q_enc = self._tokenize(questions, self.q_max_length)
        p_enc = self._tokenize(positives, self.p_max_length)
        h_enc = self._tokenize(hard_negs, self.p_max_length)

        return {
            "q_input_ids":      q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "q_token_type_ids": q_enc["token_type_ids"],
            "p_input_ids":      p_enc["input_ids"],
            "p_attention_mask": p_enc["attention_mask"],
            "p_token_type_ids": p_enc["token_type_ids"],
            "h_input_ids":      h_enc["input_ids"],
            "h_attention_mask": h_enc["attention_mask"],
            "h_token_type_ids": h_enc["token_type_ids"],
        }

    def _tokenize(self, texts: list[str], max_length: int) -> dict:
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
