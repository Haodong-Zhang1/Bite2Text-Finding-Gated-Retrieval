#!/usr/bin/env python3
"""Build and run the frozen full-train V-C structured field policy.

This module is deliberately independent of validation and hidden-test data.  A
single global source is selected per field from five grouped train-OOF folds,
then the selected heads are refit on all feature-available training cases.
"""
from __future__ import annotations

import json
import hashlib
import math
import warnings
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from parsing.schema_parser import FIELDS


MODEL_VERSION = "bite2text_vc_fulltrain_model_v1"
MODEL_PURPOSE = "full_train_global_lr_mode_policy_no_outer_data"
EVIDENCE_BOUNDARY = (
    "Train-only deployable field policy; not outer100, hidden-test, "
    "RadFact, Captioning, Clinical Score, or Final Score evidence."
)
MODEL_SPEC = {
    "C": 1.0,
    "class_weight": None,
    "max_iter": 2000,
    "multi_class": "multinomial",
    "penalty": "l2",
    "solver": "lbfgs",
    "tol": 1e-6,
}
_COMMON_CLASS_VOCABULARIES = {
    "dentition": ["early_mixed", "mixed", "permanent"],
    "constriction": ["absent", "present"],
    "crossbite": ["absent", "borderline", "present"],
    "overjet": ["increased", "negative", "normal"],
    "midline": ["centered", "deviated", "unassessable"],
    "spee": ["increased", "normal"],
    "wilson": ["increased", "normal"],
    "crowding_upper": [
        "mild",
        "mild_moderate",
        "moderate",
        "moderate_severe",
        "none",
        "severe",
    ],
    "crowding_lower": [
        "mild",
        "mild_moderate",
        "moderate",
        "moderate_severe",
        "none",
        "severe",
    ],
    "molar_R": ["I", "II", "III", "NA"],
    "molar_L": ["I", "II", "III", "NA"],
    "canine_R": ["I", "II", "III", "NA"],
    "canine_L": ["I", "II", "III", "NA"],
}
CLASS_VOCABULARIES = {
    "raw": {
        **_COMMON_CLASS_VOCABULARIES,
        "vertical": ["deep_bite", "normal", "open_bite", "overbite_increased"],
    },
    "normalized": {
        **_COMMON_CLASS_VOCABULARIES,
        "vertical": ["increased_overlap", "normal", "open_bite"],
    },
}
SELECTOR_SPEC = {
    "default_source": "field_logistic",
    "override_source": "fold_mode",
    "folds_required": 5,
    "mode_delta_vs_lr_min": 0.02,
    "positive_folds_mode_vs_lr_min": 4,
    "lr_minus_mode_ci95_upper_max_exclusive": 0.0,
    "candidate_status_required": "fitted",
    "candidate_convergence_warnings_required": 0,
    "permuted_convergence_warnings_required": 0,
    "outer_reference_reads_required": 0,
}
SELECTOR_TOP_KEYS = frozenset(
    {"folds", "lr_minus_mode_ci95", "outer_reference_reads"}
)
SELECTOR_FOLD_KEYS = frozenset(
    {
        "fold_id",
        "evaluable",
        "observed_cases",
        "field_logistic_correct",
        "fold_mode_correct",
        "candidate_status",
        "candidate_convergence_warnings",
        "permuted_convergence_warnings",
    }
)


def deterministic_mode(values: Iterable[str]) -> str:
    counts = Counter(str(value) for value in values)
    if not counts:
        raise ValueError("deterministic mode requires non-empty labels")
    maximum = max(counts.values())
    return min(label for label, count in counts.items() if count == maximum)


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _reject_nonfinite(value: Any, name: str) -> None:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{name}[{index}]")


def select_global_source(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen five-fold global LR-versus-mode selector."""
    if not isinstance(diagnostics, dict) or set(diagnostics) != SELECTOR_TOP_KEYS:
        raise ValueError("selector diagnostics top-level contract differs")
    _reject_nonfinite(diagnostics, "selector diagnostics")
    folds = diagnostics["folds"]
    if not isinstance(folds, list) or len(folds) != SELECTOR_SPEC["folds_required"]:
        raise ValueError("selector requires exactly five fold rows")

    fold_ids: list[int] = []
    for row in folds:
        if not isinstance(row, dict) or set(row) != SELECTOR_FOLD_KEYS:
            raise ValueError("selector fold row contract differs")
        fold_id = _strict_nonnegative_int(row["fold_id"], "selector fold ID")
        if fold_id >= SELECTOR_SPEC["folds_required"]:
            raise ValueError("selector fold IDs must cover exactly 0..4")
        if not isinstance(row["candidate_status"], str):
            raise ValueError("selector candidate status must be a string")
        fold_ids.append(fold_id)
    if len(set(fold_ids)) != len(fold_ids) or set(fold_ids) != set(range(5)):
        raise ValueError("selector fold IDs must be unique and cover exactly 0..4")

    total = lr_correct = mode_correct = positive = 0
    statuses: list[str] = []
    candidate_warnings = permuted_warnings = 0
    all_evaluable = True
    for row in sorted(folds, key=lambda item: item["fold_id"]):
        fold_index = row["fold_id"]
        all_evaluable = all_evaluable and row["evaluable"] is True
        observed = _strict_nonnegative_int(
            row["observed_cases"], f"fold {fold_index} observed cases"
        )
        if observed == 0:
            raise ValueError("selector fold must contain observed cases")
        lr_fold = _strict_nonnegative_int(
            row["field_logistic_correct"],
            f"fold {fold_index} LR correct",
        )
        mode_fold = _strict_nonnegative_int(
            row["fold_mode_correct"], f"fold {fold_index} mode correct"
        )
        if lr_fold > observed or mode_fold > observed:
            raise ValueError("selector correct count exceeds denominator")
        statuses.append(row["candidate_status"])
        candidate_warnings += _strict_nonnegative_int(
            row["candidate_convergence_warnings"],
            f"fold {fold_index} candidate warnings",
        )
        permuted_warnings += _strict_nonnegative_int(
            row["permuted_convergence_warnings"],
            f"fold {fold_index} permuted warnings",
        )
        total += observed
        lr_correct += lr_fold
        mode_correct += mode_fold
        positive += mode_fold > lr_fold

    interval = diagnostics["lr_minus_mode_ci95"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in interval
        )
        or float(interval[0]) > float(interval[1])
    ):
        raise ValueError("selector LR-minus-mode CI must be finite and ordered")
    outer_reads = _strict_nonnegative_int(
        diagnostics["outer_reference_reads"], "outer reference reads"
    )
    mode_delta = (mode_correct - lr_correct) / total
    checks = {
        "all_five_folds_evaluable": all_evaluable,
        "candidate_fitted_all_five_folds": statuses
        == [SELECTOR_SPEC["candidate_status_required"]]
        * SELECTOR_SPEC["folds_required"],
        "candidate_no_convergence_warnings": candidate_warnings
        == SELECTOR_SPEC["candidate_convergence_warnings_required"],
        "permuted_no_convergence_warnings": permuted_warnings
        == SELECTOR_SPEC["permuted_convergence_warnings_required"],
        "mode_delta_vs_lr_min": mode_delta
        >= SELECTOR_SPEC["mode_delta_vs_lr_min"],
        "positive_folds_mode_vs_lr_min": positive
        >= SELECTOR_SPEC["positive_folds_mode_vs_lr_min"],
        "lr_minus_mode_ci95_upper_negative": float(interval[1])
        < SELECTOR_SPEC["lr_minus_mode_ci95_upper_max_exclusive"],
        "outer_reference_reads_zero": outer_reads
        == SELECTOR_SPEC["outer_reference_reads_required"],
    }
    return {
        "selected_source": (
            SELECTOR_SPEC["override_source"]
            if all(checks.values())
            else SELECTOR_SPEC["default_source"]
        ),
        "checks": checks,
        "observed_cases": total,
        "field_logistic_correct": lr_correct,
        "fold_mode_correct": mode_correct,
        "mode_delta_vs_lr": mode_delta,
        "positive_folds_mode_vs_lr": positive,
        "lr_minus_mode_ci95": [float(interval[0]), float(interval[1])],
        "candidate_statuses": statuses,
        "candidate_convergence_warnings": candidate_warnings,
        "permuted_convergence_warnings": permuted_warnings,
        "outer_reference_reads": outer_reads,
    }


def _validated_training_inputs(
    features: dict[str, np.ndarray | None],
    labels: dict[str, dict[str, Any]],
    selector_diagnostics: dict[str, dict[str, Any]],
) -> tuple[list[str], np.ndarray, int]:
    if not isinstance(features, dict) or not features or set(labels) != set(features):
        raise ValueError("features and labels must cover identical non-empty cases")
    if set(selector_diagnostics) != set(FIELDS):
        raise ValueError("selector diagnostics must cover exactly the frozen fields")
    available_ids = sorted(
        case_id for case_id, value in features.items() if value is not None
    )
    if not available_ids:
        raise ValueError("full-train model requires feature-available cases")
    arrays: list[np.ndarray] = []
    widths: set[int] = set()
    vocabulary = CLASS_VOCABULARIES["normalized"]
    for case_id in sorted(features):
        row = labels[case_id]
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise ValueError("every label row must cover exactly the frozen fields")
        for field, value in row.items():
            if value is not None and value not in vocabulary[field]:
                raise ValueError(f"label is outside frozen vocabulary: {field}")
        feature = features[case_id]
        if feature is None:
            continue
        array = np.asarray(feature, dtype=np.float64)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError("available features must be finite one-dimensional vectors")
        widths.add(len(array))
        arrays.append(array)
    if len(widths) != 1:
        raise ValueError("available features must share one width")
    matrix = np.stack(arrays)
    return available_ids, matrix, next(iter(widths))


def fit_fulltrain_model(
    features: dict[str, np.ndarray | None],
    labels: dict[str, dict[str, Any]],
    selector_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fit the frozen global field policy using only supplied training rows."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    available_ids, matrix, feature_dimension = _validated_training_inputs(
        features, labels, selector_diagnostics
    )
    scaler = StandardScaler().fit(matrix)
    matrix_z = scaler.transform(matrix)
    selected_fields: dict[str, Any] = {}
    for field in FIELDS:
        decision = select_global_source(selector_diagnostics[field])
        labelled_positions = [
            index
            for index, case_id in enumerate(available_ids)
            if labels[case_id][field] is not None
        ]
        if not labelled_positions:
            raise ValueError(f"full-train field has no observed labels: {field}")
        field_labels = [
            str(labels[available_ids[index]][field]) for index in labelled_positions
        ]
        counts = dict(sorted(Counter(field_labels).items()))
        common = {
            "source": decision["selected_source"],
            "selector_input": selector_diagnostics[field],
            "selector_decision": decision,
            "train_labelled_cases": len(labelled_positions),
            "class_counts": counts,
        }
        if decision["selected_source"] == "fold_mode":
            selected_fields[field] = {
                **common,
                "head": {
                    "kind": "constant_mode",
                    "value": deterministic_mode(field_labels),
                },
            }
            continue

        if len(counts) < 2:
            raise ValueError(f"logistic field needs at least two classes: {field}")
        estimator = LogisticRegression(**MODEL_SPEC)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(matrix_z[labelled_positions], field_labels)
        convergence_warnings = sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        if convergence_warnings:
            raise ValueError(f"full-train logistic field did not converge: {field}")
        selected_fields[field] = {
            **common,
            "head": {
                "kind": "linear_logistic",
                "classes": [str(value) for value in estimator.classes_],
                "coef": np.asarray(estimator.coef_, dtype=np.float64).tolist(),
                "intercept": np.asarray(
                    estimator.intercept_, dtype=np.float64
                ).tolist(),
                "convergence_warnings": convergence_warnings,
                "n_iter": [int(value) for value in estimator.n_iter_],
            },
        }

    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "purpose": MODEL_PURPOSE,
        "label_space": "normalized",
        "selector_spec": SELECTOR_SPEC,
        "model_spec": MODEL_SPEC,
        "feature_dimension": feature_dimension,
        "num_training_cases": len(features),
        "num_feature_available_cases": len(available_ids),
        "num_feature_missing_cases": len(features) - len(available_ids),
        "scaler": {
            "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        },
        "fields": selected_fields,
        "missing_feature_policy": "abstain_all_fields",
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }


def predict_fields(
    model: dict[str, Any], feature: np.ndarray | None
) -> dict[str, str | None]:
    """Run the exported heads with NumPy only; missing feature abstains."""
    validate_model(model)
    if feature is None:
        return {field: None for field in FIELDS}
    width = model.get("feature_dimension")
    array = np.asarray(feature, dtype=np.float64)
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or array.shape != (width,)
        or not np.isfinite(array).all()
    ):
        raise ValueError("runtime feature differs from model contract")
    scaler = model.get("scaler") or {}
    mean = np.asarray(scaler.get("mean"), dtype=np.float64)
    scale = np.asarray(scaler.get("scale"), dtype=np.float64)
    if mean.shape != (width,) or scale.shape != (width,) or np.any(scale <= 0):
        raise ValueError("runtime scaler differs from model contract")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = (array - mean) / scale
    except FloatingPointError as error:
        raise ValueError("runtime standardization produced a non-finite value") from error
    if not np.isfinite(standardized).all():
        raise ValueError("runtime standardization produced a non-finite value")
    fields = model.get("fields")
    if not isinstance(fields, dict) or set(fields) != set(FIELDS):
        raise ValueError("runtime model field coverage differs")
    result: dict[str, str | None] = {}
    for field in FIELDS:
        head = (fields[field] or {}).get("head") or {}
        if head.get("kind") == "constant_mode":
            result[field] = str(head.get("value"))
            continue
        if head.get("kind") != "linear_logistic":
            raise ValueError("runtime head kind differs from model contract")
        classes = [str(value) for value in head.get("classes", [])]
        coef = np.asarray(head.get("coef"), dtype=np.float64)
        intercept = np.asarray(head.get("intercept"), dtype=np.float64)
        if len(classes) == 2 and coef.shape == (1, width) and intercept.shape == (1,):
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    score = float(coef[0] @ standardized + intercept[0])
            except FloatingPointError as error:
                raise ValueError("runtime logistic score is non-finite") from error
            if not math.isfinite(score):
                raise ValueError("runtime logistic score is non-finite")
            result[field] = classes[1] if score > 0 else classes[0]
        elif (
            len(classes) > 2
            and coef.shape == (len(classes), width)
            and intercept.shape == (len(classes),)
        ):
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    scores = coef @ standardized + intercept
            except FloatingPointError as error:
                raise ValueError("runtime logistic scores are non-finite") from error
            if not np.isfinite(scores).all():
                raise ValueError("runtime logistic scores are non-finite")
            result[field] = classes[int(np.argmax(scores))]
        else:
            raise ValueError("runtime linear head shape differs from model contract")
    return result


def _require_sha256(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected SHA-256 must be lowercase hexadecimal")
    return value


def validate_model(model: dict[str, Any]) -> None:
    """Fail closed on malformed, inconsistent, or selector-tampered exports."""
    _reject_nonfinite(model, "model")
    expected_keys = {
        "schema_version",
        "model_version",
        "purpose",
        "label_space",
        "selector_spec",
        "model_spec",
        "feature_dimension",
        "num_training_cases",
        "num_feature_available_cases",
        "num_feature_missing_cases",
        "scaler",
        "fields",
        "missing_feature_policy",
        "evidence_boundary",
    }
    if not isinstance(model, dict) or set(model) != expected_keys:
        raise ValueError("model top-level contract differs")
    if (
        model["schema_version"] != 1
        or model["model_version"] != MODEL_VERSION
        or model["purpose"] != MODEL_PURPOSE
        or model["label_space"] != "normalized"
        or model["selector_spec"] != SELECTOR_SPEC
        or model["model_spec"] != MODEL_SPEC
        or model["missing_feature_policy"] != "abstain_all_fields"
        or model["evidence_boundary"] != EVIDENCE_BOUNDARY
    ):
        raise ValueError("model frozen protocol differs")
    width = _strict_nonnegative_int(model["feature_dimension"], "feature dimension")
    if width == 0:
        raise ValueError("feature dimension must be positive")
    total = _strict_nonnegative_int(model["num_training_cases"], "training cases")
    available = _strict_nonnegative_int(
        model["num_feature_available_cases"], "feature-available cases"
    )
    missing = _strict_nonnegative_int(
        model["num_feature_missing_cases"], "feature-missing cases"
    )
    if total == 0 or available == 0 or available + missing != total:
        raise ValueError("model training-case counts are inconsistent")
    scaler = model["scaler"]
    if not isinstance(scaler, dict) or set(scaler) != {"mean", "scale"}:
        raise ValueError("model scaler contract differs")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if (
        mean.shape != (width,)
        or scale.shape != (width,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("model scaler is invalid")

    fields = model["fields"]
    if not isinstance(fields, dict) or set(fields) != set(FIELDS):
        raise ValueError("model fields differ from frozen schema")
    vocabulary = CLASS_VOCABULARIES["normalized"]
    common_keys = {
        "source",
        "selector_input",
        "selector_decision",
        "train_labelled_cases",
        "class_counts",
        "head",
    }
    for field in FIELDS:
        row = fields[field]
        if not isinstance(row, dict) or set(row) != common_keys:
            raise ValueError(f"model field contract differs: {field}")
        recomputed = select_global_source(row["selector_input"])
        if row["selector_decision"] != recomputed:
            raise ValueError(f"selector decision was tampered: {field}")
        source = row["source"]
        if source != recomputed["selected_source"]:
            raise ValueError(f"field source differs from selector: {field}")
        labelled = _strict_nonnegative_int(
            row["train_labelled_cases"], f"{field} labelled cases"
        )
        if labelled > available:
            raise ValueError(
                f"{field} labelled cases exceed feature-available training cases"
            )
        counts = row["class_counts"]
        if (
            labelled == 0
            or not isinstance(counts, dict)
            or not counts
            or any(label not in vocabulary[field] for label in counts)
        ):
            raise ValueError(f"field class counts are invalid: {field}")
        parsed_counts = {
            str(label): _strict_nonnegative_int(count, f"{field} class count")
            for label, count in counts.items()
        }
        if any(count == 0 for count in parsed_counts.values()) or sum(
            parsed_counts.values()
        ) != labelled:
            raise ValueError(f"field class counts are inconsistent: {field}")
        head = row["head"]
        if source == "fold_mode":
            if not isinstance(head, dict) or set(head) != {"kind", "value"}:
                raise ValueError(f"mode head contract differs: {field}")
            expected_mode = min(
                label
                for label, count in parsed_counts.items()
                if count == max(parsed_counts.values())
            )
            if head["kind"] != "constant_mode" or head["value"] != expected_mode:
                raise ValueError(f"mode head value differs: {field}")
            continue

        expected_head_keys = {
            "kind",
            "classes",
            "coef",
            "intercept",
            "convergence_warnings",
            "n_iter",
        }
        if not isinstance(head, dict) or set(head) != expected_head_keys:
            raise ValueError(f"logistic head contract differs: {field}")
        classes = head["classes"]
        if (
            head["kind"] != "linear_logistic"
            or not isinstance(classes, list)
            or len(classes) < 2
            or classes != sorted(parsed_counts)
        ):
            raise ValueError(f"logistic classes differ: {field}")
        coef = np.asarray(head["coef"], dtype=np.float64)
        intercept = np.asarray(head["intercept"], dtype=np.float64)
        expected_rows = 1 if len(classes) == 2 else len(classes)
        if (
            coef.shape != (expected_rows, width)
            or intercept.shape != (expected_rows,)
            or not np.isfinite(coef).all()
            or not np.isfinite(intercept).all()
            or head["convergence_warnings"] != 0
            or not isinstance(head["n_iter"], list)
            or len(head["n_iter"]) != 1
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in head["n_iter"]
            )
        ):
            raise ValueError(f"logistic parameters are invalid: {field}")


def serialize_model(model: dict[str, Any]) -> bytes:
    validate_model(model)
    return (
        json.dumps(
            model,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def deserialize_model(payload: bytes, *, expected_sha256: str) -> dict[str, Any]:
    """Load only an exactly hash-bound canonical model resource."""
    expected = _require_sha256(expected_sha256)
    if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("model resource SHA-256 mismatch")
    try:
        model = json.loads(
            payload.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("model resource is not valid UTF-8 JSON") from error
    validate_model(model)
    if serialize_model(model) != payload:
        raise ValueError("model resource is not canonical JSON")
    return model
