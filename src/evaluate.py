import argparse
import torch
import wandb
import json
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import faiss

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

    k = config.get('k', 5000) # Get k from config, default to 5000
    mean_average_precision = evaluate(query_model, doc_model, k=k)

    print(f"---" * 10)
    print(f"📈 Mean Average Precision (MAP@{k}): {mean_average_precision:.4f}")
    print(f"---" * 10)

    if config.get("wandb_log", False):
        wandb.log({"validation_map": mean_average_precision})


def evaluate(query_model, doc_model, k=5000):
    """
    Evaluates the model on the validation set using Mean Average Precision (MAP).
    Uses faiss for efficient top-k retrieval.
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

    doc_index = torch.cat(doc_embeddings, dim=0).numpy() # Faiss requires numpy array
    print(f"Document index created with shape: {doc_index.shape}")

    # --- 3.5 Build Faiss Index ---
    print(f"Building Faiss index for {doc_index.shape[0]} documents...")
    embedding_dim = doc_index.shape[1]
    
    if torch.cuda.is_available():
        print("Using GPU for Faiss index.")
        res = faiss.StandardGpuResources() # use a single GPU
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True # Use float16 for faster processing
        faiss_index = faiss.GpuIndexFlatL2(res, embedding_dim, flat_config)
    else:
        print("No GPU found, using CPU for Faiss index.")
        faiss_index = faiss.IndexFlatL2(embedding_dim) # Using L2 distance
        
    faiss.normalize_L2(doc_index) # Normalize for cosine similarity search
    faiss_index.add(doc_index)
    print("Faiss index built.")

    # --- 4. Evaluate Queries ---
    print(f"Evaluating queries by retrieving top {k} documents...")
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
            q_vecs = query_model((q_flat, q_off)).cpu().numpy()
            faiss.normalize_L2(q_vecs) # Normalize for cosine similarity search

            # Search the Faiss index for the top k documents for the entire batch
            similarity_scores, top_k_indices = faiss_index.search(q_vecs, k)

            # For each query in the batch, calculate its AP
            for j in range(q_vecs.shape[0]):
                query_idx_in_full_list = i * EVAL_BATCH_SIZE + j
                current_query_tokens = tuple(
                    query_list[query_idx_in_full_list])

                # Get the ground truth positive documents for the current query
                positive_docs_for_query = {
                    tuple(item) for item in query_to_positives[current_query_tokens]}
                
                # From the top_k_indices, create the relevance label vector
                # The doc_list contains the original token lists
                retrieved_docs = [tuple(doc_list[idx]) for idx in top_k_indices[j]]
                is_relevant = np.array([1 if doc in positive_docs_for_query else 0 for doc in retrieved_docs])

                # Check if there were any relevant documents retrieved at all
                if np.sum(is_relevant) == 0:
                    average_precisions.append(0.0) # Score is 0 if no relevant docs are in top k
                    continue

                # The similarity scores from Faiss are squared L2 distances.
                # Convert to cosine similarity: cos(sim) = 1 - (d^2 / 2)
                ranking_scores = 1 - (similarity_scores[j] / 2)

                # label_ranking_average_precision_score expects a 2D array of [n_samples, n_labels]
                # Here, we have one sample (the query) and k_labels (the retrieved docs)
                ap_score = label_ranking_average_precision_score(
                    is_relevant.reshape(1, -1),
                    ranking_scores.reshape(1, -1)
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
        "--k",
        type=int,
        default=5000,
        help="Number of top documents to retrieve for evaluation.",
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
