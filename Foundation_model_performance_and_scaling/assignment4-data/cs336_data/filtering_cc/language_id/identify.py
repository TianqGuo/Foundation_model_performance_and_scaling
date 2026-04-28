from __future__ import annotations

from pathlib import Path

import fasttext

_MODEL_PATH = Path(__file__).parents[2] / "assets" / "lid.176.bin"
_model: fasttext.FastText._FastText | None = None


def _get_model() -> fasttext.FastText._FastText:
    global _model
    if _model is None:
        fasttext.FastText.eprint = lambda x: None  # suppress stderr warnings
        _model = fasttext.load_model(str(_MODEL_PATH))
    return _model


def identify_language(text: str) -> tuple[str, float]:
    """Return (language_id, confidence) for the dominant language in text.

    Uses the fastText lid.176.bin model. The language id is a BCP-47-style
    two-letter code (e.g. "en", "zh", "de").
    """
    model = _get_model()
    # predict() returns ([label], [prob]); newlines confuse fastText
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    lang = labels[0].replace("__label__", "")
    score = float(min(probs[0], 1.0))
    return lang, score