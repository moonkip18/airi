import hashlib
import os
import re
from typing import List, Optional

import numpy as np
import structlog
from app.config import settings

logger = structlog.get_logger()

# The HF hub prints a tqdm progress bar on EVERY encode batch plus an
# unauthenticated-request warning; both are pure console noise once the
# model is cached (it is downloaded once, then loaded from disk).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:  # pragma: no cover - depends on environment
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # ImportError or deeper incompatibility (e.g. Python 3.14 without torch)
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
_FALLBACK_DIM = 512


class FallbackEmbeddingModel:
    """Hashed lexical embedding used when sentence-transformers is unavailable.

    Deterministic, dependency-light (numpy only), fast on modest laptops.
    Combines word tokens + character trigrams so it works reasonably for both
    English and Persian text. Not as semantically deep as a transformer, but
    preserves the same interface and ranking behavior.
    """

    def __init__(self, dim: int = _FALLBACK_DIM):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def _sign(self, token: str) -> float:
        digest = hashlib.md5(("sign:" + token).encode("utf-8")).digest()
        return 1.0 if digest[0] % 2 == 0 else -1.0

    def encode_single(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = (text or "").lower()
        words = WORD_RE.findall(text)
        for word in words:
            vec[self._bucket("w:" + word)] += 1.0 * self._sign("w:" + word)
            if len(word) > 3:
                for i in range(len(word) - 2):
                    tri = word[i:i + 3]
                    vec[self._bucket("c:" + tri)] += 0.35 * self._sign("c:" + tri)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [self.encode_single(t) for t in texts]

    @staticmethod
    def similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            # Mixed-backend database (e.g. memories embedded by the hashed
            # fallback before sentence-transformers became available) —
            # vectors from different spaces can't be compared.
            return 0.0
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)


class EmbeddingModel:
    """Embedding facade.

    Uses sentence-transformers when available; otherwise transparently falls
    back to FallbackEmbeddingModel so memory features keep working on modest
    machines (e.g. Python versions without torch wheels).
    """

    _instance: Optional["EmbeddingModel"] = None
    _model = None
    _backend: str = "unloaded"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def backend(self) -> str:
        return self._backend

    def load(self, model_name: Optional[str] = None):
        if self._model is None:
            name = model_name or settings.embedding_model
            if _SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    logger.info("loading_embedding_model", model=name)
                    self._model = SentenceTransformer(name)
                    self._backend = "sentence-transformers"
                except Exception as e:
                    # Most common: no/blocked route to huggingface.co for the
                    # first-time model download (this machine often needs the
                    # proxy). The fallback keeps memory features working.
                    logger.warning(
                        "embedding_model_load_failed_using_fallback",
                        error=str(e)[:200],
                    )
            if self._model is None:
                logger.warning(
                    "using_fallback_embedding_backend",
                    reason=(
                        "sentence-transformers not importable"
                        if not _SENTENCE_TRANSFORMERS_AVAILABLE
                        else f"failed to load model '{name}'"
                    ),
                )
                self._model = FallbackEmbeddingModel()
                self._backend = "fallback"
        return self._model

    def get_model(self):
        if self._model is None:
            return self.load()
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        model = self.get_model()
        if isinstance(model, FallbackEmbeddingModel):
            return model.encode(texts)
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]

    def similarity(self, a: List[float], b: List[float]) -> float:
        model = self.get_model()
        if isinstance(model, FallbackEmbeddingModel):
            return FallbackEmbeddingModel.similarity(a, b)
        if not a or not b or len(a) != len(b):
            # Mixed-backend database (old hash-fallback vectors vs current
            # transformer vectors) — not comparable, never crash retrieval.
            return 0.0
        return float(np.dot(a, b))


embedding_model = EmbeddingModel()