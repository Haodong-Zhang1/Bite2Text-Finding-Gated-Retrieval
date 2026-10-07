"""V2 hash-bound G0 probability inference with frozen-order tie handling."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from baselines.vc_fulltrain_model import (
    CLASS_VOCABULARIES,
    deserialize_model,
    predict_fields,
)
from parsing.schema_parser import FIELDS


def _binary_probabilities(score: float) -> np.ndarray:
    # sklearn's binary multinomial path applies softmax([-score, score]),
    # which is sigmoid(2 * score), rather than the usual binary sigmoid(score).
    if score >= 0.0:
        exp_negative = math.exp(-2.0 * score)
        positive = 1.0 / (1.0 + exp_negative)
    else:
        exp_score = math.exp(2.0 * score)
        positive = exp_score / (1.0 + exp_score)
    return np.asarray([1.0 - positive, positive], dtype=np.float64)


def _multiclass_probabilities(scores: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            shifted = scores - np.max(scores)
            weights = np.exp(shifted)
            probabilities = weights / weights.sum()
    except FloatingPointError as error:
        raise ValueError("runtime softmax produced a non-finite value") from error
    if not np.isfinite(probabilities).all():
        raise ValueError("runtime softmax produced a non-finite value")
    return probabilities


def _validated_probability_map(
    probability_map: dict[str, float], class_order: list[str]
) -> dict[str, float]:
    values = np.asarray(
        [probability_map[label] for label in class_order], dtype=np.float64
    )
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("runtime probability vector is invalid")
    return {
        label: float(value)
        for label, value in zip(class_order, values, strict=True)
    }


def _first_maximum_in_frozen_order(
    probability_map: dict[str, float], class_order: list[str]
) -> str:
    """Return the first maximum in the supplied frozen order, never lexical order."""
    if not class_order:
        raise ValueError("frozen class order must be non-empty")
    try:
        maximum = max(probability_map[label] for label in class_order)
    except KeyError as error:
        raise ValueError("probability map differs from frozen class order") from error
    return next(
        label for label in class_order if probability_map[label] == maximum
    )


def predict_field_probability_vectors(
    model_payload: bytes,
    *,
    expected_sha256: str,
    feature: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    """Return one full frozen-vocabulary probability record per field."""
    model = deserialize_model(model_payload, expected_sha256=expected_sha256)
    if feature is None:
        return {
            field: {
                "status": "unavailable",
                "source": model["fields"][field]["source"],
                "class_order": list(CLASS_VOCABULARIES["normalized"][field]),
                "category": None,
                "probabilities": None,
            }
            for field in FIELDS
        }
    categories = predict_fields(model, feature)
    width = model["feature_dimension"]
    array = np.asarray(feature, dtype=np.float64)
    mean = np.asarray(model["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(model["scaler"]["scale"], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = (array - mean) / scale
    except FloatingPointError as error:
        raise ValueError("runtime standardization produced a non-finite value") from error
    if standardized.shape != (width,) or not np.isfinite(standardized).all():
        raise ValueError("runtime standardization produced a non-finite value")

    output: dict[str, dict[str, Any]] = {}
    vocabularies = CLASS_VOCABULARIES["normalized"]
    for field in FIELDS:
        row = model["fields"][field]
        source = row["source"]
        class_order = list(vocabularies[field])
        head = row["head"]
        if source == "fold_mode":
            counts = row["class_counts"]
            total = sum(counts.values())
            probability_map = {
                label: float(counts.get(label, 0) / total) for label in class_order
            }
        else:
            classes = list(head["classes"])
            coef = np.asarray(head["coef"], dtype=np.float64)
            intercept = np.asarray(head["intercept"], dtype=np.float64)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    scores = coef @ standardized + intercept
            except FloatingPointError as error:
                raise ValueError("runtime logistic scores are non-finite") from error
            if not np.isfinite(scores).all():
                raise ValueError("runtime logistic scores are non-finite")
            if len(classes) == 2:
                probabilities = _binary_probabilities(float(scores[0]))
            else:
                probabilities = _multiclass_probabilities(scores)
            probability_by_class = dict(zip(classes, probabilities, strict=True))
            probability_map = {
                label: float(probability_by_class.get(label, 0.0))
                for label in class_order
            }
        probability_map = _validated_probability_map(probability_map, class_order)
        probability_category = _first_maximum_in_frozen_order(
            probability_map, class_order
        )
        if probability_category != categories[field]:
            raise ValueError("probability argmax differs from frozen category runtime")
        output[field] = {
            "status": "available",
            "source": source,
            "class_order": class_order,
            "category": probability_category,
            "probabilities": probability_map,
        }
    return output
