#!/usr/bin/env python3
"""Confidence-bounded fact donor selection for the v2 runtime."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baselines import vc_sentence_surgery_runtime as sentence_runtime
from parsing.schema_parser import FIELDS, normalize_for_scoring, parse_report
from submission.mesh_retrieval_v1.mesh_retrieval_runtime import (
    MeshRetrievalModel,
    paired_mesh_descriptor,
)
from vlm_lite.production_runtime_v1 import (
    PRIMARY_FIELDS,
    _validate_provided_mesh_path,
)


RUNTIME_VERSION = "bitevlm_lite_fact_rerank_runtime_v1"


@dataclass(frozen=True)
class FactRerankRuntime:
    base: Any
    donor_labels: tuple[Mapping[str, Any], ...]

    @property
    def c0_model(self) -> Any:
        return self.base.c0_model

    @classmethod
    def from_base(cls, base: Any) -> "FactRerankRuntime":
        labels = tuple(
            normalize_for_scoring(parse_report(report))
            for report in base.c0_model.reports
        )
        return cls(base=base, donor_labels=labels)

    def predict(
        self,
        upper_path: Path | None,
        lower_path: Path | None,
        photo_path: Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if upper_path is None or lower_path is None:
            return self.base.predict(upper_path, lower_path, photo_path)
        _validate_provided_mesh_path(upper_path)
        _validate_provided_mesh_path(lower_path)
        descriptor = paired_mesh_descriptor(
            upper_path,
            lower_path,
            max_triangles=self.base.c0_model.max_triangles,
        )
        photo_feature, photo_status = self.base._photo_feature(photo_path)
        targets, _ = self.base.predict_structured_targets(
            descriptor,
            photo_feature=photo_feature,
        )
        current, selected, current_errors, candidate_errors = select_fact_donor(
            self.base.c0_model,
            descriptor,
            self.donor_labels,
            targets,
        )
        source_report = self.base.c0_model.reports[selected]
        report, _ = sentence_runtime.surgery_report(
            source_report,
            targets,
            self.base.sentence_pool,
        )
        residual_count = len(PRIMARY_FIELDS) if photo_feature is not None else 0
        route = "fact_top5" if selected != current else "mesh_1nn"
        return report, {
            "runtime_version": RUNTIME_VERSION,
            "source": route + ("_g0_photo_residual" if residual_count else "_g0"),
            "neighbor_index": selected,
            "photo_status": photo_status,
            "residual_fields_applied": residual_count,
            "current_mismatches": current_errors,
            "candidate_mismatches": candidate_errors,
            "surgery_applied": report != source_report,
        }


def select_fact_donor(
    model: MeshRetrievalModel,
    descriptor: np.ndarray,
    donor_labels: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    """Return current, selected, current mismatches, and best mismatches."""
    query_z = (np.asarray(descriptor, dtype=np.float64) - model.mean) / model.scale
    squared = (
        np.sum(query_z**2)
        + np.sum(model.train_z**2, axis=1)
        - 2.0 * model.train_z @ query_z
    )
    squared = np.maximum(squared, 0.0)
    indices = np.arange(len(model.train_z), dtype=np.int64)
    shortlist = np.lexsort((indices, squared))[:5]

    def mismatches(index: int) -> int:
        return sum(donor_labels[index].get(field) != targets[field] for field in FIELDS)

    counts = [mismatches(int(index)) for index in shortlist]
    best_position = min(range(len(shortlist)), key=counts.__getitem__)
    current = int(shortlist[0])
    candidate = int(shortlist[best_position])
    selected = (
        candidate
        if counts[0] - counts[best_position] >= 1 and counts[best_position] <= 6
        else current
    )
    return current, selected, counts[0], counts[best_position]
