"""Leakage-safe primitives for the BiteVLM-Lite photo residual gate."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


MAX_P1_TRAINABLE_PARAMETERS = 3_000_000
MAX_P1_RUNTIME_PARAMETERS = 30_000_000


def canonical_case_ids_sha256(case_ids: Sequence[str]) -> str:
    ordered = sorted(case_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("case IDs must be unique")
    payload = "\n".join(ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def align_case_keyed_inputs(
    *,
    case_ids: Sequence[str],
    features_by_case: Mapping[str, np.ndarray],
    g0_by_case: Mapping[str, np.ndarray],
    labels_by_case: Mapping[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, str]:
    ordered = sorted(case_ids)
    expected = set(ordered)
    if len(ordered) != len(expected) or any(
        set(mapping) != expected
        for mapping in (features_by_case, g0_by_case, labels_by_case)
    ):
        raise ValueError("case-keyed photo/G0/label inputs must cover the same unique cases")
    features = np.stack([np.asarray(features_by_case[case_id]) for case_id in ordered])
    g0 = np.stack([np.asarray(g0_by_case[case_id]) for case_id in ordered])
    labels = np.asarray([labels_by_case[case_id] for case_id in ordered], dtype=np.int64)
    return ordered, features, g0, labels, canonical_case_ids_sha256(ordered)


def validate_parameter_budget(
    *, runtime_components: Mapping[str, int], trainable_parameters: int
) -> dict[str, int]:
    if not runtime_components or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in runtime_components.values()
    ):
        raise ValueError("runtime parameter counts must be non-negative integers")
    if (
        isinstance(trainable_parameters, bool)
        or not isinstance(trainable_parameters, int)
        or trainable_parameters < 0
    ):
        raise ValueError("trainable parameter count must be a non-negative integer")
    runtime_total = sum(runtime_components.values())
    if runtime_total > MAX_P1_RUNTIME_PARAMETERS:
        raise ValueError("P1 runtime parameter ceiling exceeded")
    if trainable_parameters > MAX_P1_TRAINABLE_PARAMETERS:
        raise ValueError("P1 trainable parameter ceiling exceeded")
    return {
        "runtime_parameters": runtime_total,
        "trainable_parameters": trainable_parameters,
    }


def selected_g0_probability_map(
    *,
    e0_artifact: Mapping[str, Any],
    global_artifact: Mapping[str, Any],
    case_id: str,
    field: str,
    label_space: str = "normalized",
) -> Mapping[str, float]:
    selected_sources = global_artifact.get("selected_sources")
    if not isinstance(selected_sources, Mapping) or field not in selected_sources:
        raise ValueError("field has no frozen G0 selected source")
    source = selected_sources[field]
    try:
        value = e0_artifact["probabilities"][label_space][source][case_id][field]
    except (KeyError, TypeError) as error:
        raise ValueError("G0 probability lineage is incomplete") from error
    if not isinstance(value, Mapping):
        raise ValueError("G0 probability map must be a mapping")
    return value


def probability_vector(
    probability_map: Mapping[str, float], classes: Sequence[str]
) -> np.ndarray:
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be unique and non-empty")
    unknown = set(probability_map).difference(classes)
    if unknown:
        raise ValueError("G0 probability map contains an unknown class")
    values = np.asarray([float(probability_map.get(label, 0.0)) for label in classes])
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("G0 probabilities must be finite, non-negative, and non-empty")
    return values / values.sum()


def validate_inner_oof_lineage(
    *,
    assignments: Mapping[str, int],
    outer_fold: int,
    meta_train_case_ids: Sequence[str],
    base_fit_case_ids: Sequence[str],
    duplicate_components: Sequence[Sequence[str]],
    required_upstream_sha256: Mapping[str, str],
    source: str,
    class_order: Sequence[str],
    lineage_artifact: Mapping[str, Any],
    training_sets: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    folds = sorted({int(value) for value in assignments.values()})
    if folds != list(range(5)) or outer_fold not in folds:
        raise ValueError("the frozen five-fold assignment is required")
    universe = set(assignments)
    meta_cases = list(meta_train_case_ids)
    base_cases = list(base_fit_case_ids)
    if len(meta_cases) != len(set(meta_cases)) or len(base_cases) != len(set(base_cases)):
        raise ValueError("lineage case lists must be unique")
    if not set(meta_cases).issubset(universe) or not set(base_cases).issubset(universe):
        raise ValueError("lineage contains an unknown case")
    meta_exclusions = lineage_artifact.get("meta_exclusions")
    base_exclusions = lineage_artifact.get("base_fit_exclusions")
    outer_exclusions = lineage_artifact.get("outer_eval_exclusions")
    if not all(isinstance(value, Mapping) for value in (meta_exclusions, base_exclusions, outer_exclusions)):
        raise ValueError("lineage requires explicit exclusion mappings")
    if any(not isinstance(reason, str) or not reason for mapping in (meta_exclusions, base_exclusions, outer_exclusions) for reason in mapping.values()):
        raise ValueError("lineage exclusion reasons must be non-empty strings")
    expected_meta_universe = {case_id for case_id in universe if int(assignments[case_id]) != outer_fold}
    if set(meta_cases).intersection(meta_exclusions) or set(meta_cases).union(meta_exclusions) != expected_meta_universe:
        raise ValueError("meta lineage does not cover every eligible/excluded non-outer case")
    if set(base_cases).intersection(base_exclusions) or set(base_cases).union(base_exclusions) != universe:
        raise ValueError("base-fit lineage does not cover the full case universe")
    outer_cases = {case_id for case_id in universe if int(assignments[case_id]) == outer_fold}
    outer_predictions = lineage_artifact.get("outer_predictions")
    inner_predictions = lineage_artifact.get("inner_predictions")
    if not isinstance(inner_predictions, Mapping) or set(inner_predictions) != set(meta_cases):
        raise ValueError("inner prediction lineage must exactly cover meta-train cases")
    if not isinstance(outer_predictions, Mapping) or set(outer_predictions).intersection(outer_exclusions) or set(outer_predictions).union(outer_exclusions) != outer_cases:
        raise ValueError("outer G0 lineage must cover every predicted/excluded outer case")
    observed_hashes = lineage_artifact.get("upstream_sha256")
    if observed_hashes != dict(required_upstream_sha256) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in required_upstream_sha256.values()
    ):
        raise ValueError("lineage upstream SHA-256 binding is invalid")
    if not source or not class_order or len(set(class_order)) != len(class_order):
        raise ValueError("lineage source and class order must be frozen")
    for component in duplicate_components:
        members = list(component)
        if not set(members).issubset(universe) or len({assignments[case_id] for case_id in members}) > 1:
            raise ValueError("duplicate component is unknown or crosses a fold")

    def validate_record(case_id: str, record: Mapping[str, Any], excluded_folds: set[int]) -> None:
        prediction_fold = int(assignments[case_id])
        if int(record.get("prediction_fold", -1)) != prediction_fold:
            raise ValueError("prediction fold does not match assignment")
        expected_training_ids = sorted(
            candidate
            for candidate in base_cases
            if int(assignments[candidate]) not in excluded_folds
        )
        training_ids = record.get("training_case_ids")
        if training_ids is None:
            training_set_id = record.get("training_set_id")
            if (
                not isinstance(training_set_id, str)
                or not training_set_id
                or not isinstance(training_sets, Mapping)
                or training_set_id not in training_sets
            ):
                raise ValueError("compact lineage training-set reference is invalid")
            training_set = training_sets[training_set_id]
            if not isinstance(training_set, Mapping):
                raise ValueError("compact lineage training-set row is invalid")
            training_ids = training_set.get("training_case_ids")
            if (
                training_set.get("training_case_ids_sha256")
                != record.get("training_case_ids_sha256")
                or training_set.get("training_folds")
                != record.get("training_folds")
            ):
                raise ValueError("compact lineage training-set binding drifted")
        if training_ids != expected_training_ids or len(training_ids) != len(set(training_ids)):
            raise ValueError("lineage training case IDs are not the strict expected set")
        if record.get("training_case_ids_sha256") != canonical_case_ids_sha256(training_ids):
            raise ValueError("lineage training-case hash mismatch")
        expected_training_folds = sorted(set(folds).difference(excluded_folds))
        if record.get("training_folds") != expected_training_folds:
            raise ValueError("lineage training folds are not strict or contain duplicates")
        if record.get("source") != source or record.get("class_order") != list(class_order):
            raise ValueError("lineage source/class order drifted")

    for case_id in meta_cases:
        prediction_fold = int(assignments[case_id])
        validate_record(case_id, inner_predictions[case_id], {outer_fold, prediction_fold})
    for case_id, record in outer_predictions.items():
        validate_record(case_id, record, {outer_fold})


def _hamming(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) != len(right):
        raise ValueError("matched-swap profiles must have equal lengths")
    return sum(a != b for a, b in zip(left, right))


def _strict_view_mask(values: Sequence[Any]) -> tuple[bool, ...]:
    if not values or any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("view masks must contain only explicit booleans")
    return tuple(bool(value) for value in values)


def build_matched_swap_map(
    *,
    case_ids: Sequence[str],
    assignments: Mapping[str, int],
    g0_profiles: Mapping[str, Sequence[Any]],
    view_masks: Mapping[str, Sequence[bool]],
    page_counts: Mapping[str, int],
    duplicate_components: Sequence[Sequence[str]],
    salt: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Create a deterministic within-fold, one-to-one photo derangement.

    Matching uses only scalar availability, selected-page count, and frozen G0
    predicted profiles. Gold labels, report text, paths, filenames, and image
    pixels are absent. Page-count distance is a nuisance-control cost and is
    never exposed to the residual model.
    """
    ordered = sorted(case_ids)
    if len(set(ordered)) != len(ordered) or not salt:
        raise ValueError("matched-swap case IDs and salt must be unique/non-empty")
    expected = set(ordered)
    if set(page_counts) != expected or any(
        isinstance(page_counts[case_id], bool)
        or not isinstance(page_counts[case_id], int)
        or page_counts[case_id] < 0
        for case_id in ordered
    ):
        raise ValueError("matched-swap page counts must cover cases with integers")
    component_by_case: dict[str, int] = {}
    for component_index, component in enumerate(duplicate_components):
        for case_id in component:
            if case_id in component_by_case:
                raise ValueError("duplicate components must be disjoint")
            component_by_case[case_id] = component_index
    result: dict[str, str] = {}
    total_page_count_distance = 0
    total_profile_distance = 0
    for fold in sorted({int(assignments[case_id]) for case_id in ordered}):
        fold_members = [case_id for case_id in ordered if int(assignments[case_id]) == fold]
        mask_groups: dict[tuple[bool, ...], list[str]] = {}
        for case_id in fold_members:
            mask = _strict_view_mask(view_masks[case_id])
            mask_groups.setdefault(mask, []).append(case_id)
        for mask, members in sorted(mask_groups.items()):
            members = sorted(members)
            if len(members) < 2:
                raise ValueError("exact-mask swap stratum has no valid donor")
            profile_width = len(g0_profiles[members[0]])
            if profile_width < 1:
                raise ValueError("swap profiles must be non-empty")
            page_cost_stride = (profile_width + 1) * 1000
            costs = np.empty((len(members), len(members)), dtype=np.float64)
            for row_index, query_id in enumerate(members):
                if len(g0_profiles[query_id]) != profile_width or _strict_view_mask(view_masks[query_id]) != mask:
                    raise ValueError("swap profile width or exact mask drifted")
                for column_index, donor_id in enumerate(members):
                    same_component = (
                        query_id in component_by_case
                        and component_by_case.get(query_id) == component_by_case.get(donor_id)
                    )
                    if query_id == donor_id or same_component:
                        costs[row_index, column_index] = 1e15
                        continue
                    profile_distance = _hamming(g0_profiles[query_id], g0_profiles[donor_id])
                    page_distance = abs(
                        page_counts[query_id] - page_counts[donor_id]
                    )
                    tie = int.from_bytes(
                        hashlib.sha256(f"{salt}|{query_id}|{donor_id}".encode()).digest()[:2],
                        "big",
                    ) % 1000
                    costs[row_index, column_index] = (
                        page_distance * page_cost_stride
                        + profile_distance * 1000
                        + tie
                    )
            row_indices, column_indices = linear_sum_assignment(costs)
            for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
                query_id, donor_id = members[row_index], members[column_index]
                if costs[row_index, column_index] >= 1e15 or query_id == donor_id:
                    raise ValueError("exact-mask duplicate-safe swap derangement is impossible")
                result[query_id] = donor_id
                total_page_count_distance += abs(
                    page_counts[query_id] - page_counts[donor_id]
                )
                total_profile_distance += _hamming(g0_profiles[query_id], g0_profiles[donor_id])
    if set(result) != set(ordered) or len(set(result.values())) != len(ordered):
        raise AssertionError("matched-swap mapping must be a bijection")
    return result, {
        "version": "bitevlm_lite_matched_photo_swap_v1",
        "case_count": len(ordered),
        "within_fold": all(assignments[left] == assignments[right] for left, right in result.items()),
        "bijection": True,
        "derangement": all(left != right for left, right in result.items()),
        "total_selected_page_count_absolute_difference": total_page_count_distance,
        "total_view_mask_hamming": 0,
        "total_g0_profile_hamming": total_profile_distance,
        "uses_gold_labels": False,
        "uses_report_text": False,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def pool_unordered_photo_set(
    features: np.ndarray,
    page_available: Sequence[bool],
) -> tuple[np.ndarray, bool]:
    """Pool per-photo features without assigning or depending on view order."""
    values = np.asarray(features, dtype=np.float64)
    mask = np.asarray(page_available)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("photo-set features must have shape [pages, features]")
    if mask.shape != (values.shape[0],) or mask.dtype != np.bool_:
        raise ValueError("photo-set availability must be one explicit boolean per page")
    pooled = np.zeros(values.shape[1] * 2, dtype=np.float64)
    if not mask.any():
        return pooled, False
    selected = values[mask]
    if not np.isfinite(selected).all():
        raise ValueError("available photo-set features must be finite")
    selected = np.ascontiguousarray(selected, dtype="<f8")
    canonical_order = sorted(
        range(selected.shape[0]),
        key=lambda index: selected[index].tobytes(order="C"),
    )
    canonical = selected[np.asarray(canonical_order, dtype=np.int64)]
    pooled[: values.shape[1]] = canonical.mean(axis=0, dtype=np.float64)
    pooled[values.shape[1] :] = canonical.max(axis=0)
    return pooled, True


def fuse_with_photo_residual(
    g0_probabilities: np.ndarray,
    residual_logits: np.ndarray,
    *,
    photo_available: bool,
    alpha: float = 1.0,
) -> np.ndarray:
    g0 = np.asarray(g0_probabilities, dtype=np.float64)
    if g0.ndim != 1:
        raise ValueError("G0 probabilities must be a 1-D vector")
    if not np.isfinite(g0).all() or (g0 < 0).any() or not np.isclose(g0.sum(), 1.0):
        raise ValueError("G0 probabilities must be finite and sum to one")
    if not photo_available:
        return g0.copy()
    residual = np.asarray(residual_logits, dtype=np.float64)
    if residual.shape != g0.shape:
        raise ValueError("G0 probabilities and residual logits must be equal 1-D vectors")
    if not np.isfinite(residual).all() or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("photo residual and alpha must be finite/non-negative")
    scaled_residual = alpha * residual
    if not np.isfinite(scaled_residual).all():
        raise ValueError("scaled photo residual overflowed")
    return _softmax(np.log(np.clip(g0, 1e-12, 1.0)) + scaled_residual)


@dataclass(frozen=True)
class LinearResidualHead:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @property
    def trainable_parameters(self) -> int:
        return int(self.weights.size)

    def residual_logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("photo feature width does not match residual head")
        if not np.isfinite(values).all():
            raise ValueError("photo features must be finite")
        return ((values - self.mean) / self.scale) @ self.weights

    def predict_proba(
        self,
        features: np.ndarray,
        g0_probabilities: np.ndarray,
        photo_available: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        g0 = np.asarray(g0_probabilities, dtype=np.float64)
        available = np.asarray(photo_available, dtype=bool)
        if g0.ndim != 2 or values.shape[0] != g0.shape[0] or available.shape != (g0.shape[0],):
            raise ValueError("residual prediction row counts do not match")
        result = g0.copy()
        available_indices = np.flatnonzero(available)
        if available_indices.size == 0:
            return result
        logits = self.residual_logits(values[available_indices])
        if logits.shape != (available_indices.size, g0.shape[1]):
            raise ValueError("residual class width does not match G0")
        for row, index in enumerate(available_indices.tolist()):
            result[index] = fuse_with_photo_residual(
                g0[index], logits[row], photo_available=True
            )
        return result


def fit_linear_residual_head(
    *,
    features: np.ndarray,
    g0_probabilities: np.ndarray,
    labels: np.ndarray,
    steps: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
) -> LinearResidualHead:
    """Fit a no-intercept photo residual over frozen G0 probabilities."""
    values = np.asarray(features, dtype=np.float64)
    g0 = np.asarray(g0_probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1 or g0.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("residual training arrays have incompatible shapes")
    if g0.shape[0] != values.shape[0] or g0.shape[1] < 2:
        raise ValueError("residual training requires at least two classes")
    if not np.isfinite(values).all() or not np.isfinite(g0).all():
        raise ValueError("residual training arrays must be finite")
    if (g0 < 0).any() or not np.allclose(g0.sum(axis=1), 1.0):
        raise ValueError("G0 training probabilities must sum to one")
    if (targets < 0).any() or (targets >= g0.shape[1]).any():
        raise ValueError("residual labels are outside the class range")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
        or not np.isfinite(l2)
        or l2 < 0
    ):
        raise ValueError("residual optimizer settings are invalid")
    parameter_count = values.shape[1] * g0.shape[1]
    if parameter_count > MAX_P1_TRAINABLE_PARAMETERS:
        raise ValueError("residual head exceeds the P1 trainable-parameter ceiling")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (values - mean) / scale
    weights = np.zeros((values.shape[1], g0.shape[1]), dtype=np.float64)
    one_hot = np.eye(g0.shape[1], dtype=np.float64)[targets]
    base_logits = np.log(np.clip(g0, 1e-12, 1.0))
    for _ in range(steps):
        probabilities = _softmax(base_logits + standardized @ weights)
        gradient = standardized.T @ (probabilities - one_hot) / values.shape[0]
        gradient += l2 * weights
        weights -= learning_rate * gradient
        if not np.isfinite(weights).all():
            raise ValueError("residual optimization produced non-finite weights")
    return LinearResidualHead(mean=mean, scale=scale, weights=weights)
