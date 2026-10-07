"""Grand Challenge entrypoint wiring for the frozen attempt05 tooth gate."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


APP_ROOT = Path("/opt/app")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(1, str(APP_ROOT))

import numpy as np

from submission.bitevlm_lite_v2 import inference as v2_inference
from submission.mesh_retrieval_v1.mesh_retrieval_runtime import (
    paired_mesh_descriptor,
)
from tooth_finding_gate_v1 import (
    MIN_NEIGHBOR_SUPPORT,
    gate_report,
    neighbor_support,
)
from vlm_lite.fact_rerank_runtime_v1 import FactRerankRuntime
from vlm_lite.per_case_fail_open_runtime_v1 import PerCaseFailOpenRuntime


CASE_TIMEOUT_SECONDS = 240.0
RUNTIME_VERSION = "bitevlm_lite_v3_toothgate_attempt05"
ShortlistReportProvider = Callable[[Path, Path], Sequence[str]]
_load_v2 = v2_inference._load_runtime


def _with_audit(audit: Any, **updates: Any) -> dict[str, Any]:
    merged = dict(audit) if isinstance(audit, Mapping) else {}
    merged.update(updates)
    return merged


@dataclass(frozen=True)
class ToothFindingGateRuntime:
    """Apply the frozen gate after v3 generation, failing open per report."""

    base_runtime: FactRerankRuntime
    shortlist_report_provider: ShortlistReportProvider | None = None

    @property
    def c0_model(self) -> Any:
        return self.base_runtime.c0_model

    def _fulltrain_shortlist_reports(
        self,
        upper_path: Path,
        lower_path: Path,
    ) -> tuple[str, ...]:
        model = self.base_runtime.base.c0_model
        descriptor = paired_mesh_descriptor(
            upper_path,
            lower_path,
            max_triangles=model.max_triangles,
        )
        query_z = (
            np.asarray(descriptor, dtype=np.float64) - model.mean
        ) / model.scale
        squared = (
            np.sum(query_z**2)
            + np.sum(model.train_z**2, axis=1)
            - 2.0 * model.train_z @ query_z
        )
        squared = np.maximum(squared, 0.0)
        indices = np.arange(len(model.train_z), dtype=np.int64)
        shortlist = np.lexsort((indices, squared))[:5]
        return tuple(model.reports[int(index)] for index in shortlist)

    def predict(
        self,
        upper_path: Path | None,
        lower_path: Path | None,
        photo_path: Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        report, audit = self.base_runtime.predict(
            upper_path,
            lower_path,
            photo_path,
        )
        if upper_path is None or lower_path is None:
            return report, _with_audit(
                audit,
                tooth_gate_runtime=RUNTIME_VERSION,
                tooth_gate_status="skipped_missing_mesh",
            )

        try:
            provider = (
                self.shortlist_report_provider
                or self._fulltrain_shortlist_reports
            )
            reports = tuple(provider(upper_path, lower_path))
            support = neighbor_support(reports)
            gated, gate_audit = gate_report(
                report,
                support=support,
                min_support=MIN_NEIGHBOR_SUPPORT,
            )
            if not isinstance(gated, str) or not gated.strip():
                raise ValueError("tooth gate returned an empty report")
        except Exception as error:
            return report, _with_audit(
                audit,
                tooth_gate_runtime=RUNTIME_VERSION,
                tooth_gate_status=f"fail_open_{type(error).__name__}",
            )

        return gated, _with_audit(
            audit,
            tooth_gate_runtime=RUNTIME_VERSION,
            tooth_gate_status="changed" if gate_audit.changed else "unchanged",
            tooth_gate_crossbite_scopes_removed=(
                gate_audit.crossbite_scopes_removed
            ),
            tooth_gate_pathology_sentences_removed=(
                gate_audit.pathology_sentences_removed
            ),
            tooth_gate_pathology_sentences_minimised=(
                gate_audit.pathology_sentences_minimised
            ),
        )


def _load_candidate(paths, *, photo_feature_provider):
    base = _load_v2(paths, photo_feature_provider=photo_feature_provider)
    fact_runtime = FactRerankRuntime.from_base(base)
    gate_runtime = ToothFindingGateRuntime(base_runtime=fact_runtime)
    return PerCaseFailOpenRuntime(
        base_runtime=gate_runtime,
        timeout_seconds=CASE_TIMEOUT_SECONDS,
    )


v2_inference._load_runtime = _load_candidate
select_handler = v2_inference.select_handler


if __name__ == "__main__":
    try:
        raise SystemExit(v2_inference.run())
    except SystemExit:
        raise
    except Exception as error:
        print(
            f"inference_failed={type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None
