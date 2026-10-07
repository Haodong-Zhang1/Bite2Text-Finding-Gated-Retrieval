"""Per-case fallback wrapper for the isolated BiteVLM-Lite candidate."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import signal
import threading
from typing import Any


class _CasePredictionTimeout(TimeoutError):
    pass


@contextmanager
def _prediction_deadline(seconds: float):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("per-case deadline requires the main thread")

    def expire(_signum, _frame):
        raise _CasePredictionTimeout

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@dataclass(frozen=True)
class PerCaseFailOpenRuntime:
    base_runtime: Any
    timeout_seconds: float

    def __post_init__(self) -> None:
        report = self.base_runtime.c0_model.fallback_report
        if not isinstance(report, str) or not report.strip():
            raise ValueError("base-rate fallback report is empty")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0.0
        ):
            raise ValueError("timeout_seconds must be positive and finite")

    def predict(self, upper_path, lower_path, photo_path=None):
        try:
            with _prediction_deadline(float(self.timeout_seconds)):
                return self.base_runtime.predict(upper_path, lower_path, photo_path)
        except _CasePredictionTimeout:
            return self._fallback("timeout")
        except Exception:
            return self._fallback("error")

    def _fallback(self, reason: str):
        return self.base_runtime.c0_model.fallback_report, {
            "source": f"fallback_case_{reason}",
            "photo_status": f"not_evaluated_case_{reason}",
            "residual_fields_applied": 0,
            "surgery_applied": False,
        }
