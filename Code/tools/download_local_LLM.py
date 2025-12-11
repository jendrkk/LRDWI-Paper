"""
General-purpose module for downloading and managing local LLM copies from Hugging Face.

This module provides utility functions to:
 - Check if a model is already available locally
 - Download a model from HF cache or Hub
 - Save a local copy for offline use

Usage:
    from download_localLLM import ensure_local_model
    local_path = ensure_local_model("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "/path/to/local/dir")
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download


def ensure_local_model(repo_id: str, target_dir: str) -> str:
    """Ensure `target_dir` contains a ready-to-load model from HuggingFace.

    Parameters:
    -----------
    repo_id : str
        The HuggingFace Hub repository ID (e.g., "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    target_dir : str
        The local directory where the model should be stored/accessed

    Returns:
    --------
    str
        The path to the local model directory (target_dir)

    Behavior:
    ---------
    1. If `target_dir` already exists and contains files, returns it immediately (no download)
    2. Tries to locate the model in the local HF cache without network access
    3. If not found locally, downloads from HuggingFace Hub
    4. Returns the path to the local model for use with transformers/sentence_transformers
    """
    target_dir = str(target_dir)  # Ensure it's a string
    
    # Check if model already exists locally
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[LLM Downloader] Using existing local model at {target_dir}")
        return target_dir

    os.makedirs(target_dir, exist_ok=True)

    # Try to locate in the local HF cache without network
    try:
        print(f"[LLM Downloader] Searching for '{repo_id}' in local HF cache...")
        cached_dir = snapshot_download(repo_id, local_files_only=True, cache_dir=target_dir)
        print(f"[LLM Downloader] Found model in HF cache: {cached_dir}")
        return cached_dir
    except Exception as e:
        print(f"[LLM Downloader] Model not in local cache. Downloading from HuggingFace Hub...")
        try:
            cached_dir = snapshot_download(repo_id, cache_dir=target_dir)
            print(f"[LLM Downloader] Successfully downloaded model to {cached_dir}")
            return cached_dir
        except Exception as download_error:
            raise RuntimeError(
                f"Failed to download model '{repo_id}' to '{target_dir}'. Error: {download_error}"
            ) from download_error
