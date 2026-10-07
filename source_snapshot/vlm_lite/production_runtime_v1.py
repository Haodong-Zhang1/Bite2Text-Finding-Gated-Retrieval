#!/usr/bin/env python3
"""Lean CPU runtime for the hash-bound BiteVLM-Lite V2 resource."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from baselines import vc_sentence_surgery_runtime as sentence_runtime
from parsing.schema_parser import FIELDS
from submission.mesh_retrieval_v1.mesh_retrieval_runtime import (
    FEATURE_DIMENSIONS,
    MeshRetrievalModel,
    mesh_descriptor,
    paired_mesh_descriptor,
)
from vlm_lite.g0_probability_runtime_v2 import (
    predict_field_probability_vectors,
)
from vlm_lite.photo_encoder_runtime_v1 import (
    PhotoSetUnavailable,
    validate_feature_contract,
)


RUNTIME_VERSION = "bitevlm_lite_production_runtime_v1"
V2_RESOURCE_VERSION = "bitevlm_lite_fulltrain_resource_v2"
V2_RESIDUAL_VERSION = "bitevlm_lite_fulltrain_photo_residual_v2"
V2_RESOURCE_FILES = frozenset(
    {"g0_model.json", "photo_residual.json", "manifest.json"}
)
VC_RESOURCE_FILES = frozenset(
    {"field_model.json", "sentence_pool.json", "manifest.json"}
)
C0_RESOURCE_FILES = frozenset(
    {"train_z.npy", "mean.npy", "scale.npy", "reports.json", "manifest.json"}
)
PRIMARY_FIELDS = (
    "constriction",
    "crossbite",
    "vertical",
    "overjet",
    "midline",
    "crowding_upper",
    "crowding_lower",
    "molar_R",
    "molar_L",
    "canine_R",
    "canine_L",
)
DIAGNOSTIC_FIELDS = ("dentition", "spee", "wilson")
PHOTO_PAGE_DIMENSION = 768
PHOTO_SET_DIMENSION = 1536


PhotoFeatureProvider = Callable[[Path], np.ndarray | None]


@dataclass(frozen=True)
class ResidualHead:
    class_order: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class BiteVLMLiteProductionRuntime:
    c0_model: MeshRetrievalModel
    sentence_pool: sentence_runtime.CompactSentencePool
    g0_payload: bytes
    g0_sha256: str
    residual_heads: Mapping[str, ResidualHead]
    photo_feature_provider: PhotoFeatureProvider | None

    @classmethod
    def load(
        cls,
        *,
        v2_resource_dir: Path,
        expected_v2_manifest_sha256: str,
        expected_v2_execution_protocol_sha256: str,
        vc_resource_dir: Path,
        expected_vc_manifest_sha256: str,
        c0_resource_dir: Path,
        photo_feature_provider: PhotoFeatureProvider | None = None,
    ) -> "BiteVLMLiteProductionRuntime":
        v2_payloads = _strict_payloads(v2_resource_dir, V2_RESOURCE_FILES, "V2")
        manifest_payload = v2_payloads["manifest.json"]
        if _sha256_bytes(manifest_payload) != _require_sha256(
            expected_v2_manifest_sha256, "V2 manifest"
        ):
            raise ValueError("V2 manifest external SHA-256 mismatch")
        v2_manifest = _load_canonical_json(manifest_payload, "V2 manifest")
        if (
            v2_manifest.get("resource_version") != V2_RESOURCE_VERSION
            or v2_manifest.get("status") != "COMPLETE"
            or v2_manifest.get("purpose")
            != "metadata_corrected_all_train_G0_plus_photo_residual"
            or v2_manifest.get("execution_protocol_sha256")
            != _require_sha256(
                expected_v2_execution_protocol_sha256,
                "V2 execution protocol",
            )
        ):
            raise ValueError("V2 manifest identity or protocol binding drifted")
        member_hashes = v2_manifest.get("files")
        if not isinstance(member_hashes, Mapping) or set(member_hashes) != {
            "g0_model.json",
            "photo_residual.json",
        }:
            raise ValueError("V2 member binding schema drifted")
        for name, expected in member_hashes.items():
            if _sha256_bytes(v2_payloads[name]) != _require_sha256(
                expected, f"V2 {name}"
            ):
                raise ValueError(f"V2 resource member SHA-256 mismatch: {name}")

        g0_payload = v2_payloads["g0_model.json"]
        g0_sha256 = member_hashes["g0_model.json"]
        g0_contract = predict_field_probability_vectors(
            g0_payload,
            expected_sha256=g0_sha256,
            feature=None,
        )
        g0_json = _load_canonical_json(g0_payload, "V2 G0 model")
        if g0_json.get("feature_dimension") != FEATURE_DIMENSIONS:
            raise ValueError("V2 G0 descriptor width drifted")

        residual = _load_canonical_json(
            v2_payloads["photo_residual.json"], "V2 photo residual"
        )
        heads = _load_residual_heads(
            residual,
            g0_contract=g0_contract,
            expected_g0_sha256=g0_sha256,
        )

        sentence_pool, vc_manifest = _load_vc_resource(
            vc_resource_dir,
            expected_manifest_sha256=expected_vc_manifest_sha256,
        )
        c0_model = _load_bound_c0(c0_resource_dir, vc_manifest=vc_manifest)
        return cls(
            c0_model=c0_model,
            sentence_pool=sentence_pool,
            g0_payload=g0_payload,
            g0_sha256=g0_sha256,
            residual_heads=heads,
            photo_feature_provider=photo_feature_provider,
        )

    def predict_structured_targets(
        self,
        raw_descriptor: np.ndarray,
        *,
        photo_feature: np.ndarray | None,
    ) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        descriptor = np.asarray(raw_descriptor, dtype=np.float64)
        if descriptor.shape != (FEATURE_DIMENSIONS,) or not np.isfinite(
            descriptor
        ).all():
            raise ValueError("raw IOS descriptor is invalid")
        g0 = predict_field_probability_vectors(
            self.g0_payload,
            expected_sha256=self.g0_sha256,
            feature=descriptor,
        )
        feature = _validated_photo_feature(photo_feature)
        targets: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        for field in FIELDS:
            row = g0[field]
            if row["status"] != "available" or row["category"] is None:
                raise ValueError("available IOS produced unavailable G0 fields")
            class_order = list(row["class_order"])
            g0_map = row["probabilities"]
            if field in DIAGNOSTIC_FIELDS or feature is None:
                category = str(row["category"])
                probability_map = dict(g0_map)
            else:
                head = self.residual_heads[field]
                if class_order != list(head.class_order):
                    raise ValueError("G0 and residual class orders differ")
                probability_map = _fuse_residual(
                    g0_map,
                    head=head,
                    feature=feature,
                )
                category = _first_maximum(class_order, probability_map)
            targets[field] = category
            probabilities[field] = probability_map
        if set(targets) != set(FIELDS) or set(probabilities) != set(FIELDS):
            raise ValueError("structured target coverage drifted")
        return targets, probabilities

    def predict(
        self,
        upper_path: Path | None,
        lower_path: Path | None,
        photo_path: Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        _validate_provided_mesh_path(upper_path)
        _validate_provided_mesh_path(lower_path)
        if upper_path is None or lower_path is None:
            provided = upper_path if upper_path is not None else lower_path
            if provided is not None:
                mesh_descriptor(
                    provided,
                    max_triangles=self.c0_model.max_triangles,
                )
            report = self.c0_model.fallback_report
            return report, {
                "runtime_version": RUNTIME_VERSION,
                "source": "fallback_missing_ios",
                "neighbor_index": None,
                "distance": None,
                "photo_status": "not_evaluated_missing_ios",
                "residual_fields_applied": 0,
                "surgery_applied": False,
                "report_sha256": _sha256_text(report),
            }

        # The paired IOS descriptor is computed exactly once and shared by C0/G0.
        descriptor = paired_mesh_descriptor(
            upper_path,
            lower_path,
            max_triangles=self.c0_model.max_triangles,
        )
        c0_report, neighbor_index, distance = _retrieve_c0(
            self.c0_model, descriptor
        )
        photo_feature, photo_status = self._photo_feature(photo_path)
        targets, fused_probabilities = self.predict_structured_targets(
            descriptor,
            photo_feature=photo_feature,
        )
        report, surgery_audit = sentence_runtime.surgery_report(
            c0_report,
            targets,
            self.sentence_pool,
        )
        residual_count = len(PRIMARY_FIELDS) if photo_feature is not None else 0
        return report, {
            "runtime_version": RUNTIME_VERSION,
            "source": "mesh_1nn_g0_photo_residual"
            if residual_count
            else "mesh_1nn_g0",
            "neighbor_index": neighbor_index,
            "distance": distance,
            "photo_status": photo_status,
            "residual_fields_applied": residual_count,
            "diagnostic_fields_exact_g0": list(DIAGNOSTIC_FIELDS),
            "surgery_applied": report != c0_report,
            "c0_report_sha256": _sha256_text(c0_report),
            "report_sha256": _sha256_text(report),
            "structured_targets_sha256": _canonical_sha256(targets),
            "structured_probabilities_sha256": _canonical_sha256(
                fused_probabilities
            ),
            "surgery": surgery_audit,
        }

    def _photo_feature(
        self, photo_path: Path | None
    ) -> tuple[np.ndarray | None, str]:
        if photo_path is None:
            return None, "missing_exact_g0"
        path = Path(photo_path)
        if not path.is_file() or self.photo_feature_provider is None:
            return None, "unavailable_exact_g0"
        try:
            feature = self.photo_feature_provider(path)
        except PhotoSetUnavailable:
            return None, "corrupt_exact_g0"
        if feature is None:
            return None, "unavailable_exact_g0"
        return _validated_photo_feature(feature), "available_residual_applied"


def pool_photo_feature_set(page_features: np.ndarray) -> np.ndarray:
    """Reproduce the frozen unordered mean+max pool for an all-valid page set."""
    values = np.asarray(page_features)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != PHOTO_PAGE_DIMENSION
        or values.dtype != np.dtype("<f4")
        or not np.isfinite(values).all()
    ):
        raise ValueError("photo page features must be finite float32 [pages,768]")
    selected = np.ascontiguousarray(values, dtype="<f8")
    order = sorted(
        range(selected.shape[0]),
        key=lambda index: selected[index].tobytes(order="C"),
    )
    canonical = selected[np.asarray(order, dtype=np.int64)]
    pooled = np.concatenate(
        [
            canonical.mean(axis=0, dtype=np.float64),
            canonical.max(axis=0),
        ]
    )
    return np.ascontiguousarray(pooled, dtype="<f4")


def _validated_photo_feature(feature: np.ndarray | None) -> np.ndarray | None:
    if feature is None:
        return None
    values = np.asarray(feature)
    if (
        values.shape != (PHOTO_SET_DIMENSION,)
        or values.dtype != np.dtype("<f4")
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
    ):
        raise ValueError("photo-set feature must be finite contiguous float32 [1536]")
    return values


def _fuse_residual(
    g0_map: Mapping[str, float],
    *,
    head: ResidualHead,
    feature: np.ndarray,
) -> dict[str, float]:
    try:
        g0 = np.asarray([g0_map[label] for label in head.class_order], dtype=np.float64)
    except (KeyError, TypeError) as error:
        raise ValueError("G0 map differs from the residual class order") from error
    if (
        set(g0_map) != set(head.class_order)
        or not np.isfinite(g0).all()
        or (g0 < 0.0).any()
        or not np.isclose(g0.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("G0 probability vector is invalid")
    values = np.asarray(feature, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = (values - head.mean) / head.scale
            residual_logits = standardized @ head.weights
            logits = np.log(np.clip(g0, 1e-12, 1.0)) + residual_logits
            shifted = logits - np.max(logits)
            weights = np.exp(shifted)
            fused = weights / weights.sum()
    except FloatingPointError as error:
        raise ValueError("photo residual fusion produced a non-finite value") from error
    if fused.shape != g0.shape or not np.isfinite(fused).all():
        raise ValueError("photo residual fusion produced a non-finite value")
    return {
        label: float(value)
        for label, value in zip(head.class_order, fused, strict=True)
    }


def _first_maximum(
    class_order: list[str] | tuple[str, ...], probability_map: Mapping[str, float]
) -> str:
    if not class_order:
        raise ValueError("class order must be non-empty")
    maximum = max(probability_map[label] for label in class_order)
    return next(label for label in class_order if probability_map[label] == maximum)


def _load_residual_heads(
    residual: Mapping[str, Any],
    *,
    g0_contract: Mapping[str, Mapping[str, Any]],
    expected_g0_sha256: str,
) -> dict[str, ResidualHead]:
    required_top = {
        "schema_version",
        "model_version",
        "status",
        "purpose",
        "label_space",
        "primary_fields",
        "diagnostic_policy",
        "feature_contract",
        "g0_contract",
        "fusion_contract",
        "optimizer_contract",
        "training_contract",
        "parameter_counts",
        "fields",
        "upstream_sha256",
        "evidence_boundary",
        "correction",
    }
    feature_contract = residual.get("feature_contract")
    fusion_contract = residual.get("fusion_contract")
    diagnostic_policy = residual.get("diagnostic_policy")
    if not isinstance(feature_contract, Mapping):
        raise ValueError("V2 photo-residual feature contract is unavailable")
    validate_feature_contract(feature_contract)
    if (
        set(residual) != required_top
        or residual.get("schema_version") != 1
        or residual.get("model_version") != V2_RESIDUAL_VERSION
        or residual.get("status") != "COMPLETE"
        or residual.get("primary_fields") != list(PRIMARY_FIELDS)
        or diagnostic_policy
        != {
            "fields": list(DIAGNOSTIC_FIELDS),
            "action": "exact_G0_identity_no_photo_head",
        }
        or feature_contract.get("cache_dtype") != "little_endian_float32"
        or feature_contract.get("fit_dtype") != "float64"
        or fusion_contract
        != {
            "formula": "softmax(log(clip(G0,1e-12,1))+photo_residual_logits)",
            "tie_break": "frozen_class_order_first_index",
            "missing_photo_policy": "exact_G0",
            "missing_or_invalid_G0_policy": "existing_C0_report_byte_exact",
            "photo_only_fallback": False,
        }
        or residual.get("upstream_sha256", {}).get("g0_model")
        != expected_g0_sha256
        or not isinstance(residual.get("fields"), Mapping)
        or set(residual["fields"]) != set(PRIMARY_FIELDS)
    ):
        raise ValueError("V2 photo-residual contract drifted")

    required_row = {
        "fit_status",
        "class_order",
        "eligible_case_count",
        "eligible_case_ids_sha256",
        "observed_class_counts",
        "cross_fitted_G0_lineage_sha256",
        "feature_rows_sha256",
        "cross_fitted_g0_rows_sha256",
        "target_indices_sha256",
        "training_matrix_sha256",
        "assigned_outer_folds_sha256",
        "mean",
        "scale",
        "weights",
        "mean_sha256",
        "scale_sha256",
        "weights_sha256",
    }
    heads: dict[str, ResidualHead] = {}
    total_parameters = 0
    for field in PRIMARY_FIELDS:
        row = residual["fields"][field]
        class_order = tuple(g0_contract[field]["class_order"])
        if (
            not isinstance(row, Mapping)
            or set(row) != required_row
            or row.get("fit_status") != "FITTED"
            or row.get("class_order") != list(class_order)
        ):
            raise ValueError(f"V2 residual field contract drifted: {field}")
        mean = np.asarray(row["mean"], dtype=np.float64)
        scale = np.asarray(row["scale"], dtype=np.float64)
        weights = np.asarray(row["weights"], dtype=np.float64)
        if (
            mean.shape != (PHOTO_SET_DIMENSION,)
            or scale.shape != (PHOTO_SET_DIMENSION,)
            or weights.shape != (PHOTO_SET_DIMENSION, len(class_order))
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(weights).all()
            or (scale <= 0.0).any()
            or _array_sha256(mean) != row.get("mean_sha256")
            or _array_sha256(scale) != row.get("scale_sha256")
            or _array_sha256(weights) != row.get("weights_sha256")
        ):
            raise ValueError(f"V2 residual numeric payload drifted: {field}")
        total_parameters += int(weights.size)
        heads[field] = ResidualHead(class_order, mean, scale, weights)
    counts = residual.get("parameter_counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("photo_residual_biases") != 0
        or counts.get("photo_residual_weights") != total_parameters
        or counts.get("photo_trainable_total") != total_parameters
    ):
        raise ValueError("V2 residual parameter count drifted")
    return heads


def _load_vc_resource(
    resource_dir: Path, *, expected_manifest_sha256: str
) -> tuple[sentence_runtime.CompactSentencePool, dict[str, Any]]:
    payloads = _strict_payloads(resource_dir, VC_RESOURCE_FILES, "V-C")
    if _sha256_bytes(payloads["manifest.json"]) != _require_sha256(
        expected_manifest_sha256, "V-C manifest"
    ):
        raise ValueError("V-C manifest external SHA-256 mismatch")
    manifest = _load_canonical_json(payloads["manifest.json"], "V-C manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "field_model.json",
        "sentence_pool.json",
    }:
        raise ValueError("V-C member binding schema drifted")
    for name, expected in files.items():
        if _sha256_bytes(payloads[name]) != _require_sha256(expected, name):
            raise ValueError(f"V-C resource member SHA-256 mismatch: {name}")
    sentence_payload = _load_vc_canonical_json(
        payloads["sentence_pool.json"], "V-C sentence pool"
    )
    return sentence_runtime.load_compact_sentence_pool(sentence_payload), manifest


def _load_bound_c0(
    resource_dir: Path, *, vc_manifest: Mapping[str, Any]
) -> MeshRetrievalModel:
    _strict_payloads(resource_dir, C0_RESOURCE_FILES, "C0")
    directory = Path(resource_dir)
    binding = vc_manifest.get("c0_binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "manifest_sha256",
        "files",
    }:
        raise ValueError("V-C to C0 binding schema drifted")
    manifest_hash = _sha256_path(directory / "manifest.json")
    bound_files = binding["files"]
    expected_names = C0_RESOURCE_FILES - {"manifest.json"}
    if (
        manifest_hash
        != _require_sha256(binding["manifest_sha256"], "bound C0 manifest")
        or not isinstance(bound_files, Mapping)
        or set(bound_files) != expected_names
    ):
        raise ValueError("bound C0 manifest or member schema drifted")
    for name, expected in bound_files.items():
        if _sha256_path(directory / name) != _require_sha256(expected, name):
            raise ValueError(f"bound C0 member SHA-256 mismatch: {name}")
    model = MeshRetrievalModel.load(directory)
    if model.manifest.get("files") != dict(bound_files):
        raise ValueError("loaded C0 member binding drifted")
    return model


def _retrieve_c0(
    model: MeshRetrievalModel, descriptor: np.ndarray
) -> tuple[str, int, float]:
    query_z = (descriptor - model.mean) / model.scale
    squared = (
        np.sum(query_z**2)
        + np.sum(model.train_z**2, axis=1)
        - 2.0 * model.train_z @ query_z
    )
    squared = np.maximum(squared, 0.0)
    index = int(np.argmin(squared))
    return model.reports[index], index, float(np.sqrt(squared[index]))


def _validate_provided_mesh_path(path: Path | None) -> None:
    if path is None:
        return
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("provided IOS mesh path is unavailable")
    if candidate.stat().st_size < 84:
        raise ValueError("provided IOS mesh is shorter than the minimum header")


def _strict_payloads(
    resource_dir: Path, expected_files: frozenset[str], name: str
) -> dict[str, bytes]:
    directory = Path(resource_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{name} resource must be a real directory")
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != set(expected_files) or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError(f"{name} resource file inventory drifted")
    return {entry.name: entry.read_bytes() for entry in entries}


def _load_canonical_json(payload: bytes, name: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict) or _canonical_json_bytes(value) != payload:
        raise ValueError(f"{name} must be a canonical JSON object")
    return value


def _load_vc_canonical_json(payload: bytes, name: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict) or _vc_canonical_json_bytes(value) != payload:
        raise ValueError(f"{name} must be a canonical JSON object")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _vc_canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
