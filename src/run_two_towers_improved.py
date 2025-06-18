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

def main():
    LEARNING_RATE = 2e-3  # Slightly higher LR for larger batches
    EPOCHS = 5
    BATCH_SIZE = 128  # Increased from 32 for better GPU utilization
    QUERY_END = 5_000_000
    MARGIN = torch.tensor(0.2)
    device = get_device()

    print('Starting... Two Towers Document Search Training')
    print(f'Using device {device}.\n\n\n\n')

    # Loads embeddings and test out works
    init_wandb(LEARNING_RATE, EPOCHS)
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

    # Compile models for better performance (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        query_model = torch.compile(query_model)
        doc_model = torch.compile(doc_model)

    wandb.watch(query_model, log="all", log_freq=100)
    wandb.watch(doc_model, log="all", log_freq=100)

    optimizer = torch.optim.Adam(
        list(query_model.parameters()) + list(doc_model.parameters()),
        lr=LEARNING_RATE
    )

    # ── 4) Use the built-in triplet loss with cosine distance ───────────────────────
    criterion = torch.nn.TripletMarginWithDistanceLoss(
        margin=MARGIN,
        distance_function=lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, dim=1),
        reduction='mean'
    )

    # May need to change the file path
    vocab_path = load_model_path('vocab:latest')
    with open(vocab_path) as file:
        w2ix = json.load(file)

    # Calculate optimal number of workers (usually num_cores - 1, but cap at 8 for memory)
    num_workers = 0 #min(mp.cpu_count() - 1, 8) if mp.cpu_count() > 1 else 0
    print(f'Using {num_workers} workers for data loading')

    # Pre-tokenize and convert to index lists once
    queries_path = load_artifact_path('query_processed', file_extension='parquet')
    queries_processed = pd.read_parquet(queries_path)
    
    documents_path = load_artifact_path('docs_processed', file_extension='parquet')
    documents_processed = pd.read_parquet(documents_path)

    queries = queries_processed['query'].tolist()
    positives = queries_processed["doc"].tolist()
    negatives = documents_processed["doc"].tolist()

    dataset = KeyQueryDataset(start=0, end=QUERY_END, word2idx=w2ix, queries=queries, positives=positives, negatives=negatives)
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
        total_loss = 0.0
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

            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar with current loss
            if num_batches % 100 == 0:
                current_avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'avg_loss': f'{current_avg_loss:.4f}'})

        scheduler.step()  # Update learning rate
        
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f">>> Epoch {epoch+1} average loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_loss, 
            "epoch": epoch + 1,
            "learning_rate": current_lr
        })

    # Save models
    os.makedirs('data', exist_ok=True)
    torch.save(query_model.state_dict(), 'data/query_model.pt')
    save_model('query_model', 'The trained model for our queries')

    torch.save(doc_model.state_dict(), 'data/doc_model.pt')
    save_model('doc_model', 'The trained model for our documents')

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()