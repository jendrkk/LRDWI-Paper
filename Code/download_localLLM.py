"""
Download and prepare a local copy of a simple multilingual BERT from Hugging Face.

This script ensures a `target_dir` contains the tokenizer and model. It will:
 - Use the existing `target_dir` if non-empty.
 - Try to locate the repo in the local HF cache (no network).
 - Otherwise download the repo from HF and save a copy into `target_dir`.

Recommended model: `bert-base-multilingual-cased` (simple multilingual BERT).
"""

import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Use the multilingual BERT model (not distilbert) as requested
REPO_ID = "bert-base-multilingual-cased"
LOCAL_DIR = "/Users/jedrek/Documents/Utilities/Bert"   # change to wherever you want the stable copy


def ensure_local_model(repo_id: str = REPO_ID, target_dir: str = LOCAL_DIR) -> str:
    """Ensure `target_dir` contains a ready-to-load tokenizer+model.

    Returns the path to the local copy (target_dir).
    The function first returns the dir if it already contains files. Otherwise it:
      - tries to find a cached snapshot with `snapshot_download(..., local_files_only=True)`
      - if not found, downloads with `snapshot_download(repo_id)`
      - then loads tokenizer+model from the snapshot and saves them into target_dir
    """
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print("Using existing local model at", target_dir)
        return target_dir

    # Try to locate in the local HF cache without network
    try:
        cached_dir = snapshot_download(repo_id, local_files_only=True)
        print("Found model in HF cache:", cached_dir)
    except Exception:
        # Not in cache -> download from the Hub
        print("Model not cached. Downloading from Hugging Face...")
        cached_dir = snapshot_download(repo_id)
        print("Downloaded model snapshot to:", cached_dir)

    # Load from snapshot and save a clean copy into the project folder
    tokenizer = AutoTokenizer.from_pretrained(cached_dir)
    model = AutoModel.from_pretrained(cached_dir)

    os.makedirs(target_dir, exist_ok=True)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    print("Saved model and tokenizer to", target_dir)
    return target_dir


def use_local_model(local_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModel.from_pretrained(local_dir, local_files_only=True)
    return tokenizer, model


def main():
    # 1) Ensure model/tokenizer are available locally (downloads if needed).
    local_dir = ensure_local_model()

    # 2) Load tokenizer and model from the local copy. We keep `local_files_only=True`
    #    so later runs don't require network access.
    tokenizer, model = use_local_model(local_dir)

    # 3) Device handling: use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -----------------------
    # Example A: Batch encode multiple texts and compute mean-pooled embeddings
    # -----------------------
    # Prepare a small multilingual batch. The tokenizer will pad to the longest
    # sequence and return PyTorch tensors when `return_tensors='pt'`.
    texts = ["Hello world", "Bonjour le monde", "Hallo Welt", "Cześć świecie"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Move tokenized inputs to the chosen device (important when using GPU).
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass (no gradient computation needed for embeddings).
    with torch.no_grad():
        outputs = model(**inputs)
        # `last_hidden_state` has shape (batch_size, seq_len, hidden_dim)
        last_hidden = outputs.last_hidden_state

    # Create an attention mask with shape (batch_size, seq_len, 1) to zero-out pad tokens
    mask = inputs["attention_mask"].unsqueeze(-1)

    # Mean pooling over the non-padded tokens: sum(hidden * mask) / sum(mask)
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1)
    embeddings = summed / counts

    # Move embeddings back to CPU and convert to numpy for downstream use.
    embeddings_np = embeddings.cpu().numpy()
    print("Batch embeddings shape:", embeddings_np.shape)

    # Quick similarity check between first two items (cosine similarity)
    # Note: for serious use prefer `sklearn.metrics.pairwise.cosine_similarity`.
    a = embeddings_np[0]
    b = embeddings_np[1]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Cosine similarity between example 0 and 1: {cos:.4f}")

    # -----------------------
    # Example B: Single-string workflow and token inspection
    # -----------------------
    single = "This is a short test in English."
    single_inputs = tokenizer(single, return_tensors="pt")
    single_inputs = {k: v.to(device) for k, v in single_inputs.items()}

    with torch.no_grad():
        single_out = model(**single_inputs).last_hidden_state

    # Mean-pooled embedding for the single example
    single_mask = single_inputs["attention_mask"].unsqueeze(-1)
    single_emb = (single_out * single_mask).sum(1) / single_mask.sum(1)
    print("Single text embedding shape:", single_emb.shape)

    # If you want to see token-level info (ids -> tokens):
    ids = single_inputs["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print("Tokens for the single input:", tokens)

    # -----------------------
    # Notes & options
    # -----------------------
    # - Pooling: here we use mean pooling over token vectors with attention mask.
    #   Alternative: use the `[CLS]` token vector `last_hidden[:, 0, :]` for BERT
    #   or (when available) `outputs.pooler_output` (some models provide this).
    # - Speed: for large batches or longer sequences, move inputs and model to GPU.
    # - Reproducibility: set `torch.manual_seed(...)` if you rely on randomness.
    # - Downstream: you can now feed `embeddings_np` into similarity search,
    #   clustering, classifiers, or store them for retrieval.

    # Return the batch embeddings for interactive use when importing this module.
    return embeddings_np


if __name__ == "__main__":
    main()
