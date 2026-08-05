"""Retrieval 평가 (Phase 4-3): Top-k accuracy 측정.

학습된 question_encoder로 NQ test 질문을 인코딩하고, FAISS 인덱스에서 top-k passage를
검색해서 정답 문자열이 포함된 비율(Top-k accuracy)을 계산한다.

사용법:
    python src/retriever/evaluate.py --checkpoint outputs/checkpoints/best.pt
"""
import argparse
import ast
import csv
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

from src.models.biencoder import BiEncoder

csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-path", default="outputs/embeddings/faiss.index")
    parser.add_argument("--wiki-path", default="data/wikipedia/psgs_w100.tsv")
    parser.add_argument("--qa-path", default="data/nq/nq-test.qa.csv")
    parser.add_argument("--top-k", type=int, nargs="+", default=[20, 100])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-q-length", type=int, default=64)
    parser.add_argument("--out-path", default="outputs/results/retrieval_eval.json")
    return parser.parse_args()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text).lower()


def has_answer(passage_text: str, answers: list[str]) -> bool:
    # 논문의 정확한 tokenized 매칭 대신 단순화된 substring 매칭 (근사치)
    normalized_passage = normalize(passage_text)
    return any(normalize(ans) in normalized_passage for ans in answers)


def load_qas(path: str) -> list[tuple[str, list[str]]]:
    qas = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            question, answers_raw = row[0], row[1]
            qas.append((question, ast.literal_eval(answers_raw)))
    return qas


def load_passage_lookup(wiki_path: str) -> dict[int, str]:
    """id -> text. 21M개 전부 메모리에 올리므로 RAM이 넉넉한 환경(수십GB)에서 실행할 것."""
    lookup: dict[int, str] = {}
    with open(wiki_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in tqdm(reader, desc="loading wikipedia passages"):
            pid, text = row[0], row[1]
            lookup[int(pid)] = text
    return lookup


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = BiEncoder()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    question_encoder = model.question_encoder.to(device).eval()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print("FAISS 인덱스 로드...")
    index = faiss.read_index(args.index_path)

    print("QA 데이터 로드...")
    qas = load_qas(args.qa_path)
    print(f"  {len(qas):,} 질문")

    print("Wikipedia passage lookup 구성...")
    passage_lookup = load_passage_lookup(args.wiki_path)

    max_k = max(args.top_k)

    all_retrieved = []
    for i in tqdm(range(0, len(qas), args.batch_size), desc="encoding + searching"):
        batch = qas[i : i + args.batch_size]
        questions = [q for q, _ in batch]
        enc = tokenizer(
            questions, max_length=args.max_q_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = question_encoder(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                enc["token_type_ids"].to(device),
            )
        _, retrieved_ids = index.search(q_emb.float().cpu().numpy(), max_k)
        all_retrieved.append(retrieved_ids)

    all_retrieved_arr = np.concatenate(all_retrieved, axis=0)

    hits = {k: 0 for k in args.top_k}
    for (question, answers), retrieved_ids in zip(qas, all_retrieved_arr):
        for k in args.top_k:
            topk_ids = retrieved_ids[:k]
            if any(
                pid != -1 and has_answer(passage_lookup.get(int(pid), ""), answers)
                for pid in topk_ids
            ):
                hits[k] += 1

    results = {f"top_{k}_accuracy": hits[k] / len(qas) for k in args.top_k}
    print(results)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
