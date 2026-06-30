import json
import random
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class NQDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizerFast,
        q_max_length: int = 64,
        p_max_length: int = 256,
    ):
        with open(data_path) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        question = sample["question"]

        # positive passage 1개 랜덤 선택
        pos = random.choice(sample["positive_ctxs"])

        # hard negative 1개 선택 (없으면 positive로 대체)
        hard_negs = sample.get("hard_negative_ctxs", [])
        hard_neg = hard_negs[0] if hard_negs else pos

        return {
            "question": question,
            "positive": self._format_passage(pos),
            "hard_negative": self._format_passage(hard_neg),
        }

    def _format_passage(self, ctx: dict) -> str:
        # 논문: title [SEP] text 형태로 passage 구성
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        return title + " [SEP] " + text

    def tokenize(self, texts: list[str], max_length: int) -> dict:
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
