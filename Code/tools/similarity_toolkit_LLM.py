"""
similarity_toolkit_LLM.py

Local, importable toolkit for precise multilingual sentence/question similarity.

Usage:
    from similarity_toolkit_LLM import SimilarityToolkit
    st = SimilarityToolkit()                # uses the recommended default
    result = st.similarity("Czy pada dziś deszcz?", "Is it raining today?")
    print(result)  # {'score': 0.85, 'raw_cosine': 0.70, 'method': 'cosine'}
"""

from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    # CrossEncoder is optional; imported only if requested by user
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None
    
import sys
import os
wd = os.getcwd()
print(wd)




def _cosine_to_unit(x: float) -> float:
    """Map cosine in [-1,1] to [0,1] for user-friendly score."""
    return float((x + 1.0) / 2.0)


class SimilarityToolkit:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        cache_root: Optional[str] = None,
        use_cross_encoder: bool = False,
        cross_encoder_model: Optional[str] = None,
    ):
        """
        Initialize the SimilarityToolkit with a sentence transformer model.

        Parameters:
        -----------
        model_name : str
            HF repo id of sentence-transformers model (bi-encoder).
            Default: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        cache_root : Optional[str]
            Directory where downloaded models are stored.
            Defaults to ~/.cache/similarity_toolkit
        use_cross_encoder : bool
            Whether to load a cross-encoder for higher-precision scoring.
        cross_encoder_model : Optional[str]
            HF repo id for cross-encoder (required if use_cross_encoder=True).
        """
        self.model_name = model_name
        self.cache_root = Path(cache_root or Path.home() / ".cache" / "similarity_toolkit")
        
        # Download and load the main sentence transformer model
        self.local_model_path = dl.ensure_local_model(
            self.model_name,
            str(self.cache_root / model_name.replace("/", "--"))
        )
        self.encoder = SentenceTransformer(str(self.local_model_path))

        # Optionally load cross-encoder for refined similarity scoring
        self.cross_encoder = None
        if use_cross_encoder:
            if cross_encoder_model is None:
                raise ValueError("use_cross_encoder=True requires cross_encoder_model to be set.")
            if CrossEncoder is None:
                raise ImportError("sentence-transformers CrossEncoder not available. Install sentence-transformers >= 2.x")
            cross_encoder_path = dl.ensure_local_model(
                cross_encoder_model,
                str(self.cache_root / cross_encoder_model.replace("/", "--"))
            )
            self.cross_encoder = CrossEncoder(str(cross_encoder_path))

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Parameters:
        -----------
        texts : Sequence[str]
            List of texts to encode.
        normalize : bool
            If True, return L2-normalized vectors (useful for cosine similarity).

        Returns:
        --------
        np.ndarray
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        embs = self.encoder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
        return embs

    def similarity(self, a: str, b: str, method: str = "cosine") -> Dict[str, Any]:
        """
        Compute similarity between two strings.

        Parameters:
        -----------
        a : str
            First string to compare.
        b : str
            Second string to compare.
        method : str
            Similarity method: "cosine" (fast bi-encoder) or "cross" (requires cross-encoder).

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - score: float in [0,1]
            - raw_cosine or cross_raw: raw score before mapping (if available)
            - method: the method used
        """
        if method == "cross":
            if self.cross_encoder is None:
                raise ValueError("Cross-encoder not loaded. Set use_cross_encoder=True with a model.")
            # CrossEncoder returns a scalar (often in [0,1] or other scale depending on model)
            cross_out = self.cross_encoder.predict([(a, b)])
            val = float(cross_out[0])
            # Normalize to [0,1] by min-max heuristic if outside range
            if -1.0 <= val <= 1.0:
                score = _cosine_to_unit(val) if val < 0 else val
            elif 0.0 <= val <= 1.0:
                score = val
            else:
                # unknown range, map via sigmoid as fallback
                score = float(1.0 / (1.0 + np.exp(-val)))
            return {"score": score, "cross_raw": val, "method": "cross"}
        else:
            # bi-encoder cosine similarity
            embs = self.encode([a, b], normalize=True)
            cos = float(np.dot(embs[0], embs[1]))
            score = _cosine_to_unit(cos)
            return {"score": score, "raw_cosine": cos, "method": "cosine"}

    def pairwise_similarity_matrix(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two sets of texts.

        Parameters:
        -----------
        texts_a : Sequence[str]
            First set of texts.
        texts_b : Sequence[str]
            Second set of texts.

        Returns:
        --------
        np.ndarray
            Similarity matrix of shape (len(texts_a), len(texts_b)) with values in [0,1].
        """
        A = self.encode(texts_a, normalize=True)
        B = self.encode(texts_b, normalize=True)
        mat = cosine_similarity(A, B)  # values in [-1,1]
        # map to [0,1]
        return (mat + 1.0) / 2.0

    def report(self, a: str, b: str, method: str = "cosine") -> str:
        """
        Generate a human-friendly report of similarity between two strings.

        Parameters:
        -----------
        a : str
            First string.
        b : str
            Second string.
        method : str
            Similarity method to use.

        Returns:
        --------
        str
            Formatted report string.
        """
        res = self.similarity(a, b, method=method)
        s = res["score"]
        pct = round(s * 100, 1)
        return f"Similarity ({res['method']}): {s:.4f}  —  {pct}%"

def main():
    # Example usage
    st = SimilarityToolkit()
    a = "Czy pada dziś deszcz?"
    b = "Is it raining today?"
    result = st.similarity(a, b)
    print(result)
    print(st.report(a, b))
    
if __name__ == "__main__":
    main()