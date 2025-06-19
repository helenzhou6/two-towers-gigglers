import argparse
import torch
import wandb
import json
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np

from torch.utils.data import DataLoader
from utils import load_model_path, init_wandb, get_device, load_artifact_path
from two_towers import QryTower, DocTower
from dataloader import KeyQueryDataset, collate_fn_emb_bag  # Re-using for consistency


def collate_for_inference(batch_of_lists):
    """
    Collates a batch of token lists into the flat_tensor, offsets_tensor format.
    """
    # Convert lists to tensors
    batch_of_tensors = [torch.tensor(item, dtype=torch.long)
                        for item in batch_of_lists]

    # Concatenate all tensors in the batch
    flat_tensor = torch.cat(batch_of_tensors, dim=0)

    # Create the offsets tensor
    lengths = torch.tensor([len(t)
                           for t in batch_of_tensors], dtype=torch.long)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])

    return (flat_tensor, offsets)


def evaluate_cmd(config):
    """
    Evaluates the model on the validation set using Mean Average Precision (MAP).
    """
    # Initialize W&B run
    if config.get("wandb_log", False):
        init_wandb(
            name=f"eval-{config['query_model_artifact']}",
            config=config,
            job_type='evaluation'
        )
    else:  # Still need to init wandb to download artifacts
        wandb.init(project="two-towers-gigglers",
                   entity="mlx-gg", anonymous="allow")

    print("Starting evaluation...")
    device = get_device()
    BATCH_SIZE = config.get("batch_size", 32)

    # --- 1. Load Models ---
    print("Loading models...")
    # Load FastText embedding layer
    ft_embedded_path = load_model_path('fasttext_tensor:latest')
    ft_state_dict = torch.load(ft_embedded_path, map_location=device)

    embedding_bag = torch.nn.EmbeddingBag.from_pretrained(
        embeddings=ft_state_dict["weight"],
        freeze=True,  # We don't train during evaluation
        padding_idx=ft_state_dict.get('padding_idx', None),
        mode='mean',
        sparse=True,
    )

    # Load Query Tower
    query_model_path = load_model_path(config['query_model_artifact'])
    query_model = QryTower(embedding_bag).to(device)
    query_model.load_state_dict(torch.load(
        query_model_path, map_location=device))

    # Load Document Tower
    doc_model_path = load_model_path(config['doc_model_artifact'])
    doc_model = DocTower(embedding_bag).to(device)
    doc_model.load_state_dict(torch.load(doc_model_path, map_location=device))

    mean_average_precision = evaluate(query_model, doc_model)

    print(f"---" * 10)
    print(f"📈 Mean Average Precision (MAP): {mean_average_precision:.4f}")
    print(f"---" * 10)

    if config.get("wandb_log", False):
        wandb.log({"validation_map": mean_average_precision})


def evaluate(query_model, doc_model):
    """
    Evaluates the model on the validation set using Mean Average Precision (MAP).
    """

    print("Starting evaluation...")
    device = get_device()
    EVAL_BATCH_SIZE = 32

    query_model.eval()
    doc_model.eval()
    print("Model evaluation on...")

    # --- 2. Load Data ---
    print("Loading validation data...")
    # Load queries and the set of relevant documents for each query
    queries_path = load_artifact_path(
        'query_processed_validation', file_extension='parquet')
    queries_df = pd.read_parquet(queries_path)

    # Load all unique documents from the validation set
    docs_path = load_artifact_path(
        'docs_processed_validation', file_extension='parquet')
    docs_df = pd.read_parquet(docs_path)

    # Convert lists to tuples so they can be hashed for groupby
    queries_df['query'] = queries_df['query'].apply(tuple)

    # Create a mapping from query to its set of relevant documents
    query_to_positives = queries_df.groupby(
        'query').agg(lambda x: list(x))['doc'].to_dict()

    print("Data loaded.")

    # --- 3. Pre-compute Document Embeddings (Create Index) ---
    print("Creating document index...")

    doc_embeddings = []
    # Ensure data is in the expected list-of-lists format, not numpy arrays
    doc_list = [doc.tolist() for doc in docs_df['doc'].tolist()]

    # Use a dataloader for efficient batching
    doc_dataloader = DataLoader(
        doc_list,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=collate_for_inference  # Use the dedicated collate function
    )

    with torch.no_grad():
        for (doc_flat, doc_off) in tqdm(doc_dataloader, desc="Embedding Documents"):
            doc_flat, doc_off = doc_flat.to(device), doc_off.to(device)
            embeddings = doc_model((doc_flat, doc_off))
            # Move to CPU to save GPU memory
            doc_embeddings.append(embeddings.cpu())

    doc_index = torch.cat(doc_embeddings, dim=0)
    print(f"Document index created with shape: {doc_index.shape}")

    # --- 4. Evaluate Queries ---
    print("Evaluating queries...")
    average_precisions = []

    # Ensure data is in the expected list-of-lists-of-ints format
    query_list = [list(q) for q in query_to_positives.keys()]

    query_dataloader = DataLoader(
        query_list,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=collate_for_inference  # Use the dedicated collate function
    )

    with torch.no_grad():
        for i, (q_flat, q_off) in enumerate(tqdm(query_dataloader, desc="Evaluating Queries")):
            q_flat, q_off = q_flat.to(device), q_off.to(device)
            # Shape: [batch_size, emb_dim]
            q_vecs = query_model((q_flat, q_off)).cpu()

            # Calculate cosine similarity between each query and all docs
            similarity_scores = F.cosine_similarity(
                q_vecs.unsqueeze(1), doc_index.unsqueeze(0), dim=2)

            # For each query in the batch, calculate its AP
            for j in range(similarity_scores.shape[0]):
                query_idx_in_full_list = i * EVAL_BATCH_SIZE + j
                current_query_tokens = tuple(
                    query_list[query_idx_in_full_list])

                # Create the binary relevance label vector for this query
                # Convert lists to tuples for correct 'in' comparison
                positive_docs_for_query = {
                    tuple(item) for item in query_to_positives[current_query_tokens]}

                # is_relevant is 1 if doc is in positives, 0 otherwise
                is_relevant = np.array(
                    [1 if tuple(doc) in positive_docs_for_query else 0 for doc in doc_list])

                # Check if there are any relevant documents for this query in our index
                if np.sum(is_relevant) == 0:
                    continue  # Skip queries with no relevant docs in the index

                # label_ranking_average_precision_score expects a 2D array of [n_samples, n_labels]
                # Here, we have one sample (the query) and n_labels (all docs)
                ap_score = label_ranking_average_precision_score(
                    is_relevant.reshape(1, -1),
                    similarity_scores[j].numpy().reshape(1, -1)
                )
                average_precisions.append(ap_score)

    # --- 5. Final Score ---
    return np.mean(average_precisions)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Two-Towers model.")
    parser.add_argument(
        "--query_model_artifact",
        type=str,
        required=True,
        help="W&B artifact for the query model (e.g., 'query_model:v5').",
    )
    parser.add_argument(
        "--doc_model_artifact",
        type=str,
        required=True,
        help="W&B artifact for the document model (e.g., 'doc_model:v5').",
    )
    parser.add_argument(
        "--wandb_log",
        action='store_true',
        help="Log the results to W&B.",
    )
    args = parser.parse_args()

    config = vars(args)

    evaluate_cmd(config)


if __name__ == "__main__":
    main()
