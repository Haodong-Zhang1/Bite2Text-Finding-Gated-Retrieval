"""Deterministic Bite2Text mesh-retrieval runtime.

The feature contract is identical to ``binary_stl_global_geometry_v1`` for
valid binary STL inputs. ASCII STL and OBJ inputs are canonicalized to the
same triangle representation before feature extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


STL_TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ],
    align=False,
)
QUANTILES = np.asarray([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
AREA_QUANTILES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95])
DESCRIPTOR_VERSION = "binary_stl_global_geometry_v1"
PARSER_VERSION = "stl_obj_canonical_triangles_v1"
FEATURE_DIMENSIONS = 112
MODEL_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_triangles(
    triangles: np.ndarray,
    max_triangles: int,
) -> np.ndarray:
    triangle_count = len(triangles)
    if triangle_count <= 0:
        raise ValueError("mesh has no triangles")
    sample_count = min(triangle_count, max_triangles)
    if sample_count == triangle_count:
        return np.asarray(triangles, dtype=np.float64)
    indices = np.linspace(
        0,
        triangle_count - 1,
        num=sample_count,
        dtype=np.int64,
    )
    return np.asarray(triangles[indices], dtype=np.float64)


def _try_binary_stl(
    path: Path,
    max_triangles: int,
) -> tuple[np.ndarray, int, int] | None:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(84)
    if len(header) != 84:
        return None
    triangle_count = struct.unpack("<I", header[80:84])[0]
    expected_size = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    if triangle_count <= 0 or file_size != expected_size:
        return None

    mapped = np.memmap(
        path,
        dtype=STL_TRIANGLE_DTYPE,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    sample_count = min(triangle_count, max_triangles)
    if sample_count == triangle_count:
        sampled = mapped
    else:
        indices = np.linspace(
            0,
            triangle_count - 1,
            num=sample_count,
            dtype=np.int64,
        )
        sampled = mapped[indices]
    triangles = np.asarray(sampled["vertices"], dtype=np.float64)
    return triangles, triangle_count, file_size


def _parse_ascii_stl(path: Path, max_triangles: int) -> tuple[np.ndarray, int, int]:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts or parts[0].lower() != "vertex":
                continue
            if len(parts) != 4:
                raise ValueError(f"invalid ASCII STL vertex in {path.name}")
            vertices.append(tuple(float(value) for value in parts[1:4]))
    if not vertices or len(vertices) % 3:
        raise ValueError(f"invalid ASCII STL triangles: {path.name}")
    triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(triangles).all():
        raise ValueError(f"non-finite ASCII STL vertices: {path.name}")
    triangle_count = len(triangles)
    canonical_size = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    return _sample_triangles(triangles, max_triangles), triangle_count, canonical_size


def _parse_obj(path: Path, max_triangles: int) -> tuple[np.ndarray, int, int]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"invalid OBJ vertex in {path.name}")
                vertices.append(tuple(float(value) for value in parts[1:4]))
            elif parts[0] == "f":
                if len(parts) < 4:
                    raise ValueError(f"invalid OBJ face in {path.name}")
                indices: list[int] = []
                for token in parts[1:]:
                    raw_index = token.split("/", 1)[0]
                    if not raw_index:
                        raise ValueError(f"invalid OBJ face index in {path.name}")
                    index = int(raw_index)
                    if index == 0:
                        raise ValueError(f"OBJ indices are one-based: {path.name}")
                    resolved = index - 1 if index > 0 else len(vertices) + index
                    if resolved < 0 or resolved >= len(vertices):
                        raise ValueError(f"OBJ face index out of range: {path.name}")
                    indices.append(resolved)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no triangles: {path.name}")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"non-finite OBJ vertices: {path.name}")
    triangles = vertex_array[np.asarray(faces, dtype=np.int64)]
    triangle_count = len(triangles)
    canonical_size = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    return _sample_triangles(triangles, max_triangles), triangle_count, canonical_size


def _load_mesh_triangles(
    path: Path,
    max_triangles: int,
) -> tuple[np.ndarray, int, int]:
    if max_triangles <= 0:
        raise ValueError("max_triangles must be positive")
    if not path.is_file():
        raise ValueError(f"mesh file is unavailable: {path.name}")
    if path.stat().st_size < 84:
        raise ValueError(f"mesh is shorter than the minimum header: {path.name}")

    binary = _try_binary_stl(path, max_triangles)
    if binary is not None:
        return binary
    try:
        if path.suffix.lower() == ".obj":
            return _parse_obj(path, max_triangles)
        if path.suffix.lower() == ".stl":
            return _parse_ascii_stl(path, max_triangles)
    except (UnicodeDecodeError, OverflowError) as error:
        raise ValueError(f"malformed non-binary mesh: {path.name}") from error
    raise ValueError(f"unsupported or malformed mesh format: {path.name}")


def mesh_descriptor(path: Path, max_triangles: int = 20_000) -> np.ndarray:
    """Return the 53-dimensional v1 descriptor for STL or OBJ input."""
    triangles, triangle_count, canonical_file_size = _load_mesh_triangles(
        path,
        max_triangles,
    )
    vertices = triangles.reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError(f"non-finite mesh vertices: {path.name}")

    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    extents = np.ptp(vertices, axis=0)
    coordinate_quantiles = np.quantile(centered, QUANTILES, axis=0).reshape(-1)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    principal_scales = np.sqrt(np.clip(eigenvalues, 0.0, None))
    radii = np.linalg.norm(centered, axis=1)
    radial_quantiles = np.quantile(radii, QUANTILES)

    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edges_a, edges_b)
    double_areas = np.linalg.norm(cross, axis=1)
    area_quantiles = np.log1p(
        np.quantile(0.5 * double_areas, AREA_QUANTILES)
    )
    geometric_normals = cross / np.maximum(double_areas[:, None], 1e-12)
    normal_features = np.concatenate(
        [
            geometric_normals.mean(axis=0),
            geometric_normals.std(axis=0),
            np.abs(geometric_normals).mean(axis=0),
        ]
    )

    descriptor = np.concatenate(
        [
            np.asarray(
                [np.log1p(triangle_count), np.log1p(canonical_file_size)],
                dtype=np.float64,
            ),
            centroid,
            extents,
            coordinate_quantiles,
            principal_scales,
            radial_quantiles,
            area_quantiles,
            normal_features,
        ]
    )
    if descriptor.shape != (53,) or not np.isfinite(descriptor).all():
        raise ValueError(f"invalid mesh descriptor for {path.name}")
    return descriptor


def paired_mesh_descriptor(
    upper_path: Path,
    lower_path: Path,
    max_triangles: int = 20_000,
) -> np.ndarray:
    upper = mesh_descriptor(upper_path, max_triangles=max_triangles)
    lower = mesh_descriptor(lower_path, max_triangles=max_triangles)
    center_delta = upper[2:5] - lower[2:5]
    extent_log_ratio = np.log(
        (upper[5:8] + 1e-9) / (lower[5:8] + 1e-9)
    )
    descriptor = np.concatenate([upper, lower, center_delta, extent_log_ratio])
    if descriptor.shape != (FEATURE_DIMENSIONS,) or not np.isfinite(descriptor).all():
        raise ValueError("invalid paired mesh descriptor")
    return descriptor


@dataclass(frozen=True)
class Prediction:
    report: str
    source: str
    neighbor_index: int | None
    distance: float | None


@dataclass(frozen=True)
class MeshRetrievalModel:
    train_z: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    reports: tuple[str, ...]
    fallback_report: str
    max_triangles: int
    manifest: dict[str, Any]

    @classmethod
    def load(cls, model_dir: Path) -> "MeshRetrievalModel":
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("mesh-retrieval manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported mesh-retrieval model schema")
        if manifest.get("descriptor_version") != DESCRIPTOR_VERSION:
            raise ValueError("mesh descriptor version mismatch")
        if manifest.get("parser_version") != PARSER_VERSION:
            raise ValueError("mesh parser version mismatch")
        if manifest.get("runtime_code_sha256") != sha256_file(Path(__file__)):
            raise ValueError("runtime code hash does not match the model manifest")

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("model file manifest is invalid")
        expected_files = {
            "train_z.npy",
            "mean.npy",
            "scale.npy",
            "reports.json",
        }
        if set(files) != expected_files:
            raise ValueError("model file manifest has an unexpected file set")
        for name in sorted(expected_files):
            path = model_dir / name
            if not path.is_file() or sha256_file(path) != files[name]:
                raise ValueError(f"model resource hash mismatch: {name}")

        train_z = np.load(model_dir / "train_z.npy", allow_pickle=False)
        mean = np.load(model_dir / "mean.npy", allow_pickle=False)
        scale = np.load(model_dir / "scale.npy", allow_pickle=False)
        report_resource = json.loads(
            (model_dir / "reports.json").read_text(encoding="utf-8")
        )
        reports = report_resource.get("reports")
        fallback_report = report_resource.get("fallback_report")
        num_train = manifest.get("num_train_rows")
        if not isinstance(num_train, int) or num_train <= 0:
            raise ValueError("invalid training-row count")
        if train_z.dtype != np.float64 or train_z.shape != (num_train, FEATURE_DIMENSIONS):
            raise ValueError("invalid standardized training feature matrix")
        if mean.dtype != np.float64 or mean.shape != (FEATURE_DIMENSIONS,):
            raise ValueError("invalid feature mean")
        if scale.dtype != np.float64 or scale.shape != (FEATURE_DIMENSIONS,):
            raise ValueError("invalid feature scale")
        if not np.isfinite(train_z).all() or not np.isfinite(mean).all():
            raise ValueError("non-finite model features")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("invalid model feature scale")
        if not isinstance(reports, list) or len(reports) != num_train:
            raise ValueError("training report table is not aligned with the index")
        if any(not isinstance(report, str) or not report.strip() for report in reports):
            raise ValueError("training report table contains an empty report")
        if not isinstance(fallback_report, str) or not fallback_report.strip():
            raise ValueError("fallback report is empty")
        max_triangles = manifest.get("max_triangles_per_scan")
        if not isinstance(max_triangles, int) or max_triangles <= 0:
            raise ValueError("invalid max-triangles setting")
        return cls(
            train_z=train_z,
            mean=mean,
            scale=scale,
            reports=tuple(report.strip() for report in reports),
            fallback_report=fallback_report.strip(),
            max_triangles=max_triangles,
            manifest=manifest,
        )

    def predict(
        self,
        upper_path: Path | None,
        lower_path: Path | None,
    ) -> Prediction:
        if _mesh_is_unavailable(upper_path) or _mesh_is_unavailable(lower_path):
            return Prediction(
                report=self.fallback_report,
                source="fallback_missing_ios",
                neighbor_index=None,
                distance=None,
            )
        assert upper_path is not None and lower_path is not None
        query = paired_mesh_descriptor(
            upper_path,
            lower_path,
            max_triangles=self.max_triangles,
        )
        query_z = (query - self.mean) / self.scale
        squared_distances = (
            np.sum(query_z**2)
            + np.sum(self.train_z**2, axis=1)
            - 2.0 * self.train_z @ query_z
        )
        squared_distances = np.maximum(squared_distances, 0.0)
        neighbor_index = int(np.argmin(squared_distances))
        distance = float(np.sqrt(squared_distances[neighbor_index]))
        return Prediction(
            report=self.reports[neighbor_index],
            source="mesh_1nn",
            neighbor_index=neighbor_index,
            distance=distance,
        )


def _mesh_is_unavailable(path: Path | None) -> bool:
    return path is None or not path.is_file() or path.stat().st_size < 84
