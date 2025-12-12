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


def ensure_local_model(
    repo_id: str,
    target_dir: str,
    *,
    force_download: bool = False,
    local_files_only: bool | None = None,
) -> str:
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
    1. If `target_dir` already exists and contains model files and `force_download` is False, returns it
    2. Otherwise downloads into `target_dir` (tries offline cache first unless `local_files_only` is set)
    3. Returns the path to the local model for use with transformers/sentence_transformers
    """
    target_dir = str(target_dir)  # Ensure it's a string

    # Look for concrete model files (not just any files) before accepting target_dir
    required_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
    ]

    def _has_model_files(d: str) -> bool:
        try:
            # check top-level first
            names = set(os.listdir(d))
        except Exception:
            return False
        for name in required_files:
            if name in names:
                return True
        # fallback: check recursively for any of the required filenames
        for root, _, files in os.walk(d):
            for f in files:
                if f in required_files:
                    return True
        return False

    if force_download and os.path.isdir(target_dir):
        import shutil

        print(f"[LLM Downloader] force_download=True, clearing {target_dir}")
        shutil.rmtree(target_dir, ignore_errors=True)

    if not force_download and _has_model_files(target_dir):
        print(f"[LLM Downloader] Using existing local model at {target_dir}")
        return target_dir

    os.makedirs(target_dir, exist_ok=True)

    def _download(local_only: bool) -> str:
        return snapshot_download(
            repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            local_files_only=local_only,
            force_download=force_download,
            resume_download=True,
        )

    # Resolve whether we should try offline first or only online
    attempts: list[bool] = []
    if local_files_only is None:
        attempts = [True, False]
    else:
        attempts = [bool(local_files_only)]

    last_err: Exception | None = None
    for local_only in attempts:
        try:
            source = "local cache" if local_only else "HF Hub"
            print(f"[LLM Downloader] Fetching '{repo_id}' from {source}...")
            return _download(local_only)
        except Exception as err:  # noqa: PERF203
            last_err = err
            print(f"[LLM Downloader] Attempt with local_files_only={local_only} failed: {err}")
            continue

    raise RuntimeError(
        f"Failed to download model '{repo_id}' to '{target_dir}'. Last error: {last_err}"
    )
