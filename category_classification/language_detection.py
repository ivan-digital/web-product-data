from __future__ import annotations

import os
import pathlib
import threading
from typing import Iterable, List
import warnings

import requests

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_CACHE_DIR = pathlib.Path(os.getenv("FASTTEXT_CACHE_DIR", pathlib.Path.home() / ".cache" / "fasttext"))
FASTTEXT_MODEL_PATH = FASTTEXT_CACHE_DIR / "lid.176.bin"

_fasttext_model = None
_fasttext_lock = threading.Lock()

try:
    import fasttext  # type: ignore
    _FASTTEXT_AVAILABLE = True
except ImportError:
    try:
        import fasttext_wheel as fasttext  # type: ignore
        _FASTTEXT_AVAILABLE = True
    except ImportError:
        fasttext = None  # type: ignore
        _FASTTEXT_AVAILABLE = False

try:
    from langdetect import detect as _langdetect_single  # type: ignore
except ImportError:
    _langdetect_single = None


def _download_fasttext_model() -> None:
    FASTTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if FASTTEXT_MODEL_PATH.exists():
        return
    response = requests.get(FASTTEXT_MODEL_URL, stream=True, timeout=60)
    response.raise_for_status()
    with FASTTEXT_MODEL_PATH.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)


def _load_fasttext_model():
    global _fasttext_model
    if _fasttext_model is not None:
        return _fasttext_model
    with _fasttext_lock:
        if _fasttext_model is None:
            warnings.filterwarnings(
                "ignore",
                message="`load_model` does not return WordVectorModel or SupervisedModel any more",
                category=UserWarning,
            )
            _download_fasttext_model()
            _fasttext_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
    return _fasttext_model


def ensure_language_detector() -> str:
    if _FASTTEXT_AVAILABLE:
        try:
            _load_fasttext_model()
            return "fasttext"
        except Exception:
            pass
    if _langdetect_single is not None:
        return "langdetect"
    raise RuntimeError(
        "No language detector available. Install 'fasttext' or 'langdetect'."
    )


def detect_language_batch(texts: Iterable[str]) -> List[str]:
    detector = ensure_language_detector()
    texts_list = []
    for text in texts:
        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            cleaned = "?"
        texts_list.append(cleaned)
    if detector == "fasttext":
        model = _load_fasttext_model()
        labels, _ = model.predict(texts_list)
        if isinstance(labels[0], list):
            labels = [lab[0] for lab in labels]
        cleaned = [lab.replace("__label__", "") for lab in labels]
        return cleaned
    else:
        result = []
        for text in texts_list:
            try:
                result.append(_langdetect_single(text))
            except Exception:
                result.append("unk")
        return result


def annotate_dataset(dataset, num_proc: int | None = None, batch_size: int = 256):
    """Add a `language` column to a HuggingFace Dataset."""
    from datasets import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a datasets.Dataset")

    ensure_language_detector()
    workers = num_proc or max(1, (os.cpu_count() or 2) // 2)

    def _map_fn(batch):
        return {"language": detect_language_batch(batch["text"])}

    return dataset.map(
        _map_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=workers,
        new_fingerprint="language-annotated",
    )
