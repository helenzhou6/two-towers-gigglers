import argparse
import torch
import wandb
import json
import os
import multiprocessing as mp

from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from utils import load_model_path, init_wandb, get_device, save_model, load_artifact_path
from two_towers import QryTower, DocTower
from dataloader import KeyQueryDataset, collate_fn_emb_bag
from sweep_config import sweep_config


def train():
    wandb.init()
    config = wandb.config
    LEARNING_RATE = config.learning_rate
    BATCH_SIZE = config.batch_size
    MARGIN = torch.tensor(config.margin)
    EPOCHS = config.epochs
    QUERY_END = BATCH_SIZE * 1000
    device = get_device()

    wandb.config.learning_rate
    wandb.config.learning_rate
    wandb.config.learning_rate

    ft_embedded_path = load_model_path('fasttext_tensor:latest')
    ft_state_dict = torch.load(ft_embedded_path, map_location=device)

    embedding_bag_doc = torch.nn.EmbeddingBag.from_pretrained(
        embeddings=ft_state_dict["weight"],
        freeze=True,
        padding_idx=ft_state_dict.get('padding_idx', None),
        mode='mean',
        sparse=True,
    )

    embedding_bag_query = torch.nn.EmbeddingBag.from_pretrained(
        embeddings=ft_state_dict["weight"],
        freeze=True,
        padding_idx=ft_state_dict.get('padding_idx', None),
        mode='mean',
        sparse=True,
    )

    # Init Two Towers
    query_model = QryTower(embedding_bag_query).to(device)
    doc_model = DocTower(embedding_bag_doc).to(device)

    # Compilation doesn't work well with embeddingbag
    # Compile models for better performance (PyTorch 2.0+)
    # if device.type == 'cuda':
    #     query_model = torch.compile(query_model)
    #     doc_model = torch.compile(doc_model)

    wandb.watch(query_model, log="all", log_freq=100)
    wandb.watch(doc_model, log="all", log_freq=100)

    optimizer = torch.optim.Adam(
        list(query_model.parameters()) + list(doc_model.parameters()),
        lr=LEARNING_RATE
    )

    # ── 4) Use the built-in triplet loss with cosine distance ───────────────────────
    criterion = torch.nn.TripletMarginWithDistanceLoss(
        margin=MARGIN,
        distance_function=lambda x, y: 1 -
        torch.nn.functional.cosine_similarity(x, y, dim=1),
        reduction='mean'
    )

    # May need to change the file path
    vocab_path = load_model_path('vocab:latest')
    with open(vocab_path) as file:
        w2ix = json.load(file)

    # Calculate optimal number of workers (usually num_cores - 1, but cap at 8 for memory)
    num_workers = 0 #min(mp.cpu_count() - 1, 4) if mp.cpu_count() > 1 else 0
    print(f'Using {num_workers} workers for data loading')

    # Pre-tokenize and convert to index lists once
    queries_path = load_artifact_path(
        'query_processed', file_extension='parquet')
    queries_processed = pd.read_parquet(queries_path)

    documents_path = load_artifact_path(
        'docs_processed', file_extension='parquet')
    documents_processed = pd.read_parquet(documents_path)

    queries = queries_processed['query'].tolist()
    positives = queries_processed["doc"].tolist()
    negatives = documents_processed["doc"].tolist()

    dataset = KeyQueryDataset(start=0, end=QUERY_END, word2idx=w2ix,
                              queries=queries, positives=positives, negatives=negatives)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        collate_fn=collate_fn_emb_bag,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,  # Better prefetching
        drop_last=True  # Consistent batch sizes for better performance
    )

    for epoch in range(0, EPOCHS):
        query_model.train()
        doc_model.train()
        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for (q_flat, q_off), (pos_flat, pos_off), (neg_flat, neg_off) in progress_bar:
            # move to device
            q_flat = q_flat.to(device, non_blocking=True)
            q_off = q_off.to(device, non_blocking=True)
            pos_flat = pos_flat.to(device, non_blocking=True)
            pos_off = pos_off.to(device, non_blocking=True)
            neg_flat = neg_flat.to(device, non_blocking=True)
            neg_off = neg_off.to(device, non_blocking=True)

            # forward
            optimizer.zero_grad()

            q_vec = query_model((q_flat, q_off))
            pos_vec = doc_model((pos_flat, pos_off))
            neg_vec = doc_model((neg_flat, neg_off))

            # compute scalar loss
            loss = criterion(q_vec, pos_vec, neg_vec)

            loss = criterion(q_vec, pos_vec, neg_vec).to(device)

            # backward + step
            loss.backward()
            optimizer.step()

            if os.getenv('DEBUG'):
                for name, param in doc_model.named_parameters():
                    if param.grad is None:
                        print(f"No grad for {name}")
                    else:
                        print(f"Grad for {name}: {param.grad.norm()}")

            total_loss += loss.detach()
            num_batches += 1

            # Update progress bar with current loss
            if num_batches % 100 == 0:
                current_avg_loss = total_loss.item() / num_batches
                progress_bar.set_postfix(
                    {'avg_loss': f'{current_avg_loss:.4f}'})

        avg_loss = total_loss / num_batches
        print(f">>> Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_loss,
            "epoch": epoch + 1
        })

    torch.save(query_model.state_dict(), 'data/query_model.pt')
    save_model('query_model', 'The trained model for our queries')

    torch.save(doc_model.state_dict(), 'data/doc_model.pt')
    save_model('doc_model', 'The trained model for our documents')


def main():
    # Default hyperparameters
    default_config = {
        'learning_rate': 2e-3,
        'batch_size':     128,
        'margin':         0.2,
        'epochs':         5
    }
    init_wandb(
        config=default_config
    )

    parser = argparse.ArgumentParser(description='Two-Tower Training')
    parser.add_argument('--sweep', action='store_true',
                        help='Run a W&B sweep instead of a single training run')
    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project='two-towers-gigglers',
            entity='mlx-gg'
        )
        wandb.agent(sweep_id, function=train, count=20)
    else:
        train()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
