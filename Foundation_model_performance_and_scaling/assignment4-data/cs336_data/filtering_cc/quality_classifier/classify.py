from __future__ import annotations

from pathlib import Path

import fasttext

fasttext.FastText.eprint = lambda x: None

_MODEL_PATH = Path(__file__).parents[2] / "assets" / "quality_classifier.bin"
_model: fasttext.FastText._FastText | None = None


def _get_model() -> fasttext.FastText._FastText:
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Quality classifier model not found at {_MODEL_PATH}.\n"
                "Run training first:\n"
                "  cd cs336_data/filtering_cc/quality_classifier && ./part_2_7.sh"
            )
        _model = fasttext.load_model(str(_MODEL_PATH))
    return _model


def classify_quality(text: str) -> tuple[str, float]:
    """Return ('wiki' | 'cc', confidence) for the input text."""
    model = _get_model()
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    score = float(min(probs[0], 1.0))
    return label, score