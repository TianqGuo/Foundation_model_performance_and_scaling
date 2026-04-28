from __future__ import annotations

from pathlib import Path

import fasttext

fasttext.FastText.eprint = lambda x: None  # suppress stderr on load

_ASSETS = Path(__file__).parents[2] / "assets"
_NSFW_PATH = _ASSETS / "dolma_fasttext_nsfw_jigsaw_model.bin"
_HATE_PATH = _ASSETS / "dolma_fasttext_hatespeech_jigsaw_model.bin"

_nsfw_model: fasttext.FastText._FastText | None = None
_hate_model: fasttext.FastText._FastText | None = None


def _get_nsfw() -> fasttext.FastText._FastText:
    global _nsfw_model
    if _nsfw_model is None:
        _nsfw_model = fasttext.load_model(str(_NSFW_PATH))
    return _nsfw_model


def _get_hate() -> fasttext.FastText._FastText:
    global _hate_model
    if _hate_model is None:
        _hate_model = fasttext.load_model(str(_HATE_PATH))
    return _hate_model


def _predict(model: fasttext.FastText._FastText, text: str) -> tuple[str, float]:
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    score = float(min(probs[0], 1.0))
    return label, score


def classify_nsfw(text: str) -> tuple[str, float]:
    """Return ('nsfw' | 'non-nsfw', confidence)."""
    return _predict(_get_nsfw(), text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Return ('toxic' | 'non-toxic', confidence)."""
    return _predict(_get_hate(), text)