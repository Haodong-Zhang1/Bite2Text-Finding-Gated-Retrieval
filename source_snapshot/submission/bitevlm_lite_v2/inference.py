"""Grand Challenge entrypoint for the hash-bound BiteVLM-Lite V2 runtime."""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, Mapping

from vlm_lite.photo_encoder_runtime_v1 import FrozenDinoV2Small
from vlm_lite.production_runtime_v1 import BiteVLMLiteProductionRuntime


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_ROOT = Path("/opt/ml/model")

V2_RESOURCE_NAME = "bitevlm_lite_v2"
VC_RESOURCE_NAME = "vc_global_policy_v1"
C0_RESOURCE_NAME = "mesh_retrieval_v1"
CHECKPOINT_RESOURCE_NAME = "dinov2-small-ed25f3a"
EXPECTED_MODEL_MEMBERS = frozenset(
    {
        V2_RESOURCE_NAME,
        VC_RESOURCE_NAME,
        C0_RESOURCE_NAME,
        CHECKPOINT_RESOURCE_NAME,
    }
)
V2_EXECUTION_PROTOCOL_SHA256 = (
    "3433c5e386caf056cdb50a873b157b1e3bd90af36a8fb60e5ccbc382c9d8291b"
)
RESOURCE_FILES: dict[str, dict[str, str]] = {
    V2_RESOURCE_NAME: {
        "g0_model.json": "8f8d5ddf700ce0d12c4869bd416c76186bed9c625151ec75117df453bb7471fc",
        "manifest.json": "7d8165524c01296bd6f065fc51d4f413e5b174c1e493d4b56ad759d86170ceef",
        "photo_residual.json": "786fc9daca82d553d754b39034695e3c4fbbfd3ab2f151b9bf9e87320cae8389",
    },
    VC_RESOURCE_NAME: {
        "field_model.json": "96071af2f0f8aeeaab63d3b99cd1e5596904753f263cef8f943118a4a87cd339",
        "manifest.json": "04aed502f41a96029bd7c8d41e8a95a07bcf33b075528a22f66f19bac18cd164",
        "sentence_pool.json": "6bce0debfc16eef18107710077e42abe0e8470c0c0c962742adca419b6309010",
    },
    C0_RESOURCE_NAME: {
        "manifest.json": "791f2ac70445b86d25c9252d790832ee1d45a1196e9b4ce100b552a8770dd56b",
        "mean.npy": "a099b0ea3ee49b023487e4e704718547a52ee851372eb9c43efdbb660e0263d7",
        "reports.json": "8f1a45fc4cbebf985fce11383766b1661d981ebbeb2c85c7d087f5755eff83e7",
        "scale.npy": "1657acdcaa259ed6ae56389f995eff1b5a54af8a5998cc7014d3ac0416fcce58",
        "train_z.npy": "8bb5a53dcfca654d2eec276fb82bd047f994d7df18c10007732fd6dd8253d460",
    },
    CHECKPOINT_RESOURCE_NAME: {
        "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
        "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    },
}
MODEL_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(RESOURCE_FILES, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
PHOTO_PATTERNS = ("*.mha", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_root(model_root: Path) -> dict[str, Path]:
    root = Path(model_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("model root must be a real directory")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != EXPECTED_MODEL_MEMBERS or any(
        entry.is_symlink() or not entry.is_dir() for entry in entries
    ):
        raise ValueError("model root must contain exactly four real resource directories")
    paths: dict[str, Path] = {}
    for resource_name, expected in RESOURCE_FILES.items():
        directory = root / resource_name
        members = list(directory.iterdir())
        if {member.name for member in members} != set(expected) or any(
            member.is_symlink() or not member.is_file() for member in members
        ):
            raise ValueError(f"resource inventory differs: {resource_name}")
        for name, digest in expected.items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"resource SHA-256 differs: {resource_name}/{name}")
        paths[resource_name] = directory
    return paths


def _load_runtime(
    paths: Mapping[str, Path],
    *,
    photo_feature_provider: Callable[[Path], Any] | None,
) -> BiteVLMLiteProductionRuntime:
    return BiteVLMLiteProductionRuntime.load(
        v2_resource_dir=paths[V2_RESOURCE_NAME],
        expected_v2_manifest_sha256=RESOURCE_FILES[V2_RESOURCE_NAME]["manifest.json"],
        expected_v2_execution_protocol_sha256=V2_EXECUTION_PROTOCOL_SHA256,
        vc_resource_dir=paths[VC_RESOURCE_NAME],
        expected_vc_manifest_sha256=RESOURCE_FILES[VC_RESOURCE_NAME]["manifest.json"],
        c0_resource_dir=paths[C0_RESOURCE_NAME],
        photo_feature_provider=photo_feature_provider,
    )


def load_frozen_runtime(model_root: Path) -> BiteVLMLiteProductionRuntime:
    return _load_runtime(validate_model_root(model_root), photo_feature_provider=None)


def _photo_feature_contract(v2_resource: Path) -> Mapping[str, Any]:
    payload = json.loads(
        (v2_resource / "photo_residual.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    contract = payload.get("feature_contract") if isinstance(payload, dict) else None
    if not isinstance(contract, Mapping):
        raise ValueError("photo feature contract is unavailable")
    return contract


def _photo_provider(
    paths: Mapping[str, Path],
) -> Callable[[Path], Any]:
    encoder = FrozenDinoV2Small.load(
        paths[CHECKPOINT_RESOURCE_NAME],
        feature_contract=_photo_feature_contract(paths[V2_RESOURCE_NAME]),
    )

    def encode_one(path: Path):
        return encoder.encode_set([path])

    return encode_one


def run_model(
    *,
    lower_scan_path: Path | None,
    upper_scan_path: Path | None,
    intraoral_photo_path: Path | None,
    model_root: Path,
) -> str:
    paths = validate_model_root(model_root)
    provider = None
    if upper_scan_path is not None and lower_scan_path is not None and intraoral_photo_path is not None:
        provider = _photo_provider(paths)
    runtime = _load_runtime(paths, photo_feature_provider=provider)
    report, audit = runtime.predict(
        upper_scan_path,
        lower_scan_path,
        intraoral_photo_path if provider is not None else None,
    )
    print(
        "inference_status="
        + json.dumps(
            {
                "photo_status": audit["photo_status"],
                "residual_fields_applied": audit["residual_fields_applied"],
                "source": audit["source"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return report


def select_handler(inputs: list[dict]) -> Callable[[Path, Path, Path], int]:
    if not isinstance(inputs, list):
        raise RuntimeError("inputs.json must contain a list")
    slugs = {
        str((row.get("socket") or {}).get("slug") or "").strip()
        for row in inputs
        if isinstance(row, dict)
    }
    groups = {
        "lower": {"3d-lower-teeth-scan", "ios-lower-scan"},
        "upper": {"3d-upper-teeth-scan", "ios-upper-scan"},
        "photos": {"2d-intraoral-photographs", "intraoral-photo"},
    }
    if all(slugs & aliases for aliases in groups.values()):
        return interf0_handler
    raise RuntimeError("unsupported input socket configuration")


def _discover_single_mesh(
    *,
    preferred: tuple[Path, ...],
    locations: tuple[Path, ...],
) -> Path | None:
    candidates: dict[Path, Path] = {}
    for path in preferred:
        if path.is_file():
            candidates[path.resolve()] = path
    for location in locations:
        for pattern in ("*.obj", "*.stl"):
            for value in glob.glob(str(location / pattern)):
                path = Path(value)
                if path.is_file():
                    candidates[path.resolve()] = path
    values = sorted(candidates.values(), key=lambda path: path.as_posix())
    if len(values) > 1:
        raise RuntimeError("multiple mesh inputs found for one socket")
    return values[0] if values else None


def _discover_single_photo(input_path: Path) -> Path | None:
    candidates: dict[Path, Path] = {}
    for location in (
        input_path / "images" / "2d-intraoral-photographs",
        input_path / "images" / "intraoral-photo",
    ):
        for pattern in PHOTO_PATTERNS:
            for value in glob.glob(str(location / pattern)):
                path = Path(value)
                if path.is_file():
                    candidates[path.resolve()] = path
    values = sorted(candidates.values(), key=lambda path: path.as_posix())
    if len(values) > 1:
        raise RuntimeError("multiple photo files found; exactly zero or one is supported")
    return values[0] if values else None


def interf0_handler(input_path: Path, output_path: Path, model_root: Path) -> int:
    upper = _discover_single_mesh(
        preferred=(input_path / "3d-upper-teeth-scan.obj", input_path / "3d-upper-teeth-scan.stl"),
        locations=(input_path / "files" / "ios-upper",),
    )
    lower = _discover_single_mesh(
        preferred=(input_path / "3d-lower-teeth-scan.obj", input_path / "3d-lower-teeth-scan.stl"),
        locations=(input_path / "files" / "ios-lower",),
    )
    photo = _discover_single_photo(input_path)
    report = run_model(
        lower_scan_path=lower,
        upper_scan_path=upper,
        intraoral_photo_path=photo,
        model_root=model_root,
    )
    if not isinstance(report, str) or not report.strip():
        raise RuntimeError("model returned an empty report")
    output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.iterdir()):
        raise RuntimeError("output directory must be empty")
    target = output_path / "diagnostic-imaging-report.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"report": report.strip()},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    if {path.name for path in output_path.iterdir()} != {target.name}:
        raise RuntimeError("output inventory drifted")
    return 0


def _peak_rss_mib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return peak / divisor


def run() -> int:
    started = time.perf_counter()
    inputs = json.loads((INPUT_PATH / "inputs.json").read_text(encoding="utf-8"))
    result = select_handler(inputs)(INPUT_PATH, OUTPUT_PATH, MODEL_ROOT)
    print(
        "runtime_metrics="
        + json.dumps(
            {
                "peak_rss_mib": _peak_rss_mib(),
                "runtime_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except SystemExit:
        raise
    except Exception as error:
        print(f"inference_failed={type(error).__name__}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
