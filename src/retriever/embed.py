"""Wikipedia passage 임베딩 추출 (Phase 4-1).

학습된 BiEncoder 체크포인트의 passage_encoder로 21M Wikipedia passage를 인코딩해서
outputs/embeddings/ 에 샤드 단위(.npz: ids, embeddings)로 저장한다.

사용법:
    python src/retriever/embed.py --checkpoint outputs/checkpoints/best.pt
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

from src.models.biencoder import BiEncoder

csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wiki-path", default="data/wikipedia/psgs_w100.tsv")
    parser.add_argument("--out-dir", default="outputs/embeddings")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=500_000, help="샤드 파일당 passage 수")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def iter_passages(wiki_path: str):
    """psgs_w100.tsv (헤더: id, text, title) 를 스트리밍으로 읽는다."""
    with open(wiki_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            pid, text, title = row[0], row[1], row[2]
            yield pid, title + " [SEP] " + text


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = BiEncoder()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    passage_encoder = model.passage_encoder.to(device).eval()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    shard_ids: list[int] = []
    shard_embs: list[np.ndarray] = []
    shard_idx = 0
    total = 0

    def flush_shard() -> None:
        nonlocal shard_ids, shard_embs, shard_idx
        if not shard_ids:
            return
        emb = np.concatenate(shard_embs, axis=0)
        ids = np.array(shard_ids, dtype=np.int64)
        out_path = out_dir / f"shard_{shard_idx:04d}.npz"
        np.savez(out_path, ids=ids, embeddings=emb)
        print(f"  → {out_path} 저장 ({len(ids):,} passages)")
        shard_ids, shard_embs = [], []
        shard_idx += 1

    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def encode_batch() -> None:
        nonlocal batch_ids, batch_texts, total
        if not batch_texts:
            return
        enc = tokenizer(
            batch_texts, max_length=args.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            emb = passage_encoder(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                enc["token_type_ids"].to(device),
            )
        shard_embs.append(emb.float().cpu().numpy())
        shard_ids.extend(int(pid) for pid in batch_ids)
        total += len(batch_ids)
        batch_ids, batch_texts = [], []

    pbar = tqdm(iter_passages(args.wiki_path), desc="encoding passages")
    for pid, text in pbar:
        batch_ids.append(pid)
        batch_texts.append(text)
        if len(batch_texts) >= args.batch_size:
            encode_batch()
        if len(shard_ids) >= args.shard_size:
            flush_shard()
        pbar.set_postfix(total=total)

    encode_batch()
    flush_shard()

    print(f"완료: 총 {total:,}개 passage 인코딩, {shard_idx}개 샤드")


if __name__ == "__main__":
    main()
