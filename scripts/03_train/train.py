"""DPR retriever 학습 (GPU 렌탈 서버용).

사용법:
    python scripts/03_train/train.py --mini              # sanity check (NQ 5K, batch 32, 5 epochs)
    python scripts/03_train/train.py                     # 풀 학습 (NQ 전체, batch 128, 40 epochs)
    python scripts/03_train/train.py --batch-size 64      # 배치 크기만 조정
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

from src.data.collator import DPRCollator
from src.data.dataset import NQDataset
from src.models.biencoder import BiEncoder
from src.models.loss import in_batch_negative_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="NQ 5K 서브셋으로 짧게 sanity check")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-dir", default="data/nq", help="biencoder-nq-{train,dev}.json이 있는 디렉토리")
    parser.add_argument("--ckpt-dir", default="outputs/checkpoints")
    return parser.parse_args()


def run_epoch(model, loader, optimizer, scheduler, scaler, device, training: bool) -> float:
    model.train() if training else model.eval()
    total_loss, steps = 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                q_emb, p_emb = model(
                    batch["q_input_ids"].to(device),
                    batch["q_attention_mask"].to(device),
                    batch["q_token_type_ids"].to(device),
                    batch["p_input_ids"].to(device),
                    batch["p_attention_mask"].to(device),
                    batch["p_token_type_ids"].to(device),
                )
                h_emb = model.passage_encoder(
                    batch["h_input_ids"].to(device),
                    batch["h_attention_mask"].to(device),
                    batch["h_token_type_ids"].to(device),
                )
                loss = in_batch_negative_loss(q_emb, p_emb, h_emb)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()
            steps += 1

    return total_loss / steps


def main() -> None:
    args = parse_args()

    batch_size  = args.batch_size or (32 if args.mini else 128)
    num_epochs  = args.epochs or (5 if args.mini else 40)
    max_samples = args.max_samples or (5_000 if args.mini else None)

    nq_train = os.path.join(args.data_dir, "biencoder-nq-train.json")
    nq_dev   = os.path.join(args.data_dir, "biencoder-nq-dev.json")
    if not os.path.exists(nq_train) or not os.path.exists(nq_dev):
        raise FileNotFoundError(
            f"NQ 데이터를 찾을 수 없습니다: {args.data_dir}\n"
            f"먼저 데이터를 받으세요: bash scripts/00_setup_server.sh "
            f"(또는 SKIP_WIKI=1 bash scripts/01_download_data/download_server.sh ./data)"
        )
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, mini: {args.mini}, batch: {batch_size}, epochs: {num_epochs}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    collator  = DPRCollator(tokenizer)

    train_dataset = NQDataset(nq_train, tokenizer)
    dev_dataset   = NQDataset(nq_dev, tokenizer)
    if max_samples:
        train_dataset.data = train_dataset.data[:max_samples]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )
    print(f"train: {len(train_dataset):,}  dev: {len(dev_dataset):,}")
    print(f"train steps/epoch: {len(train_loader)}")

    model     = BiEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"total steps: {total_steps:,}  warmup: {warmup_steps:,}")

    best_dev_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, scaler, device, training=True)
        dev_loss   = run_epoch(model, dev_loader,   optimizer, scheduler, scaler, device, training=False)

        print(f"epoch {epoch:02d}  train_loss: {train_loss:.4f}  dev_loss: {dev_loss:.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → checkpoint saved (dev_loss: {dev_loss:.4f})")

    print("학습 완료")


if __name__ == "__main__":
    main()
