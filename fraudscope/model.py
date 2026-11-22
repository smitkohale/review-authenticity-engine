"""Text classifier: loading, batched inference and SHAP explanations.

torch, transformers and shap are imported lazily so the rest of the package
(and the test suite) works without them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .config import MAX_LENGTH, MODEL_ID, UNCERTAIN_HIGH, UNCERTAIN_LOW


@dataclass(frozen=True)
class Verdict:
    label: str  # "fake", "genuine" or "uncertain"
    headline: str
    fake_probability: float


def load_model(model_id: str = MODEL_ID):
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
    model = DistilBertForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def make_predict_fn(tokenizer, model, batch_size: int = 32) -> Callable[[Sequence[str]], np.ndarray]:
    """Return f(texts) -> array of shape (n, 2) with [p_genuine, p_fake]."""
    import torch

    def predict(texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        out = []
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))

    return predict


def make_explainer(predict_fn, tokenizer):
    import shap

    return shap.Explainer(predict_fn, shap.maskers.Text(tokenizer), output_names=["genuine", "fake"])


def explain_html(explainer, text: str, max_evals: int = 300) -> str:
    """Token attributions toward the 'fake' class, rendered as HTML by SHAP."""
    import shap

    values = explainer([text], max_evals=max_evals)
    return shap.plots.text(values[0, :, 1], display=False)


def explain_tokens(explainer, text: str, max_evals: int = 300) -> tuple[list[tuple[str, float]], float]:
    """Return ([(token, attribution toward 'fake')], base value).

    Tokens keep their trailing whitespace, so joining them reproduces the text.
    """
    values = explainer([text], max_evals=max_evals)
    tokens = [str(t) for t in values.data[0]]
    attributions = [float(v) for v in values.values[0][:, 1]]
    base = float(np.asarray(values.base_values[0])[1])
    return list(zip(tokens, attributions, strict=False)), base


def verdict_for(fake_probability: float) -> Verdict:
    p = float(fake_probability)
    if p >= UNCERTAIN_HIGH:
        return Verdict("fake", "Likely synthetic", p)
    if p <= UNCERTAIN_LOW:
        return Verdict("genuine", "Likely genuine", p)
    return Verdict("uncertain", "Uncertain: needs human review", p)
