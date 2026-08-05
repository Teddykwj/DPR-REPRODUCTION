"""FAISS 인덱스 구축 (Phase 4-2).

outputs/embeddings/ 의 샤드(.npz: ids, embeddings)를 모아 FAISS IndexFlatIP에 넣는다.
dot product 기반 exhaustive 검색 — HNSW 근사 대신 flat을 사용해 정확도를 최대로 유지한다.

사용법:
    python src/retriever/index.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faiss
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", default="outputs/embeddings")
    parser.add_argument("--out-path", default="outputs/embeddings/faiss.index")
    parser.add_argument("--dim", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(Path(args.embeddings_dir).glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(
            f"{args.embeddings_dir}에 shard_*.npz가 없습니다. embed.py를 먼저 실행하세요."
        )

    index = faiss.IndexIDMap(faiss.IndexFlatIP(args.dim))

    total = 0
    for path in tqdm(shard_paths, desc="loading shards"):
        data = np.load(path)
        ids = data["ids"]
        embeddings = data["embeddings"].astype(np.float32)
        index.add_with_ids(embeddings, ids)
        total += len(ids)

    print(f"인덱스에 총 {total:,}개 passage 추가")
    faiss.write_index(index, args.out_path)
    print(f"저장: {args.out_path}")


if __name__ == "__main__":
    main()
