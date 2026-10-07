"""Lean, fail-closed photo encoder for the BiteVLM-Lite runtime.

This module contains deployment-only decoding, DINOv2-S inference, and the
frozen unordered set pooling.  Training cache construction and SciPy are
deliberately absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


MODEL_ID = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_CLASS = "Dinov2Model"
PROCESSOR_CLASS = "BitImageProcessor"
ENCODER_PARAMETERS = 22_056_576
HIDDEN_SIZE = 384
TOKENS_PER_PAGE = 257
PAGE_WIDTH = 768
SET_WIDTH = 1536
PHYSICAL_BATCH_SIZE = 16
CHECKPOINT_FILES = {
    "config.json": {
        "bytes": 547,
        "sha256": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    },
    "model.safetensors": {
        "bytes": 88_249_960,
        "sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    },
    "preprocessor_config.json": {
        "bytes": 436,
        "sha256": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    },
}
PREPROCESSING = {
    "center_crop": [224, 224],
    "convert_rgb": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "processor_class": PROCESSOR_CLASS,
    "resample": "PIL_BICUBIC_code_3",
    "rescale_factor": 1.0 / 255.0,
    "resize_shortest_edge": 256,
}
REQUIRED_PROCESS_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "20260804",
    "VECLIB_MAXIMUM_THREADS": "1",
}
RASTER_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
SUPPORTED_SUFFIXES = RASTER_SUFFIXES | {".mha"}


class PhotoSetUnavailable(ValueError):
    """The complete photo set is unusable and must not be partially fused."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PhotoDecodeLimits:
    max_files: int = 16
    max_pages: int = 16
    max_pixels_per_page: int = 50_000_000
    max_total_pixels: int = 200_000_000

    def validate(self) -> None:
        values = (
            self.max_files,
            self.max_pages,
            self.max_pixels_per_page,
            self.max_total_pixels,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
            raise ValueError("photo decode limits must be positive integers")
        if self.max_total_pixels < self.max_pixels_per_page:
            raise ValueError("total photo pixel limit is smaller than the page limit")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_rgb(
    image: Image.Image,
    *,
    limits: PhotoDecodeLimits,
    total_pixels: int,
) -> tuple[Image.Image, int]:
    width, height = (int(value) for value in image.size)
    pixels = width * height
    if width < 1 or height < 1 or pixels > limits.max_pixels_per_page:
        raise PhotoSetUnavailable("photo_page_dimensions_out_of_bounds")
    if total_pixels + pixels > limits.max_total_pixels:
        raise PhotoSetUnavailable("photo_set_pixel_budget_exceeded")
    try:
        rgb = image.convert("RGB")
        rgb.load()
    except Exception as error:
        raise PhotoSetUnavailable("photo_page_decode_failed") from error
    if rgb.mode != "RGB" or rgb.size != (width, height):
        raise PhotoSetUnavailable("photo_rgb_conversion_drifted")
    return rgb.copy(), total_pixels + pixels


def _decode_raster(
    path: Path,
    *,
    limits: PhotoDecodeLimits,
    total_pixels: int,
) -> tuple[list[Image.Image], int]:
    pages: list[Image.Image] = []
    try:
        with Image.open(path) as image:
            frames = int(getattr(image, "n_frames", 1))
            if frames < 1:
                raise PhotoSetUnavailable("photo_container_is_empty")
            if path.suffix.lower() not in {".tif", ".tiff"} and frames != 1:
                raise PhotoSetUnavailable("non_tiff_multiframe_photo_is_unsupported")
            for index in range(frames):
                image.seek(index)
                rgb, total_pixels = _checked_rgb(
                    image, limits=limits, total_pixels=total_pixels
                )
                pages.append(rgb)
    except PhotoSetUnavailable:
        raise
    except Exception as error:
        raise PhotoSetUnavailable("photo_container_decode_failed") from error
    return pages, total_pixels


def _decode_mha(
    path: Path,
    *,
    limits: PhotoDecodeLimits,
    total_pixels: int,
) -> tuple[list[Image.Image], int]:
    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
    except Exception as error:
        raise PhotoSetUnavailable("mha_decode_failed") from error

    dimension = int(image.GetDimension())
    components = int(image.GetNumberOfComponentsPerPixel())
    size = tuple(int(value) for value in image.GetSize())
    if dimension not in {2, 3} or components not in {1, 3}:
        raise PhotoSetUnavailable("mha_layout_is_unsupported")
    if len(size) != dimension or any(value < 1 for value in size):
        raise PhotoSetUnavailable("mha_dimensions_are_invalid")
    width, height = size[0], size[1]
    frames = 1 if dimension == 2 else size[2]
    pixels = width * height
    if (
        pixels > limits.max_pixels_per_page
        or total_pixels + frames * pixels > limits.max_total_pixels
    ):
        raise PhotoSetUnavailable("mha_pixel_budget_exceeded")
    try:
        array = sitk.GetArrayFromImage(image)
    except Exception as error:
        raise PhotoSetUnavailable("mha_array_decode_failed") from error
    if array.dtype != np.uint8:
        raise PhotoSetUnavailable("mha_pixel_type_is_not_uint8")

    if dimension == 2 and components == 1 and array.shape == (height, width):
        arrays = [array]
    elif dimension == 2 and components == 3 and array.shape == (height, width, 3):
        arrays = [array]
    elif dimension == 3 and components == 1 and array.shape == (frames, height, width):
        arrays = [array[index] for index in range(frames)]
    elif (
        dimension == 3
        and components == 3
        and array.shape == (frames, height, width, 3)
    ):
        arrays = [array[index] for index in range(frames)]
    else:
        raise PhotoSetUnavailable("mha_array_layout_is_ambiguous")

    pages: list[Image.Image] = []
    for frame in arrays:
        try:
            pil = Image.fromarray(np.ascontiguousarray(frame))
        except Exception as error:
            raise PhotoSetUnavailable("mha_frame_conversion_failed") from error
        rgb, total_pixels = _checked_rgb(
            pil, limits=limits, total_pixels=total_pixels
        )
        pages.append(rgb)
    return pages, total_pixels


def decode_photo_set(
    paths: Sequence[Path],
    *,
    limits: PhotoDecodeLimits = PhotoDecodeLimits(),
) -> tuple[Image.Image, ...]:
    """Decode a complete unordered photo set or reject the whole set."""
    limits.validate()
    candidates = [Path(path) for path in paths]
    if not candidates:
        raise PhotoSetUnavailable("photo_set_missing")
    if len(candidates) > limits.max_files:
        raise PhotoSetUnavailable("photo_file_count_exceeded")
    resolved: list[Path] = []
    for path in candidates:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise PhotoSetUnavailable("photo_file_is_missing_or_empty")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise PhotoSetUnavailable("photo_suffix_is_unsupported")
        resolved.append(path.resolve(strict=True))
    if len(set(resolved)) != len(resolved):
        raise PhotoSetUnavailable("photo_file_is_duplicated")

    pages: list[Image.Image] = []
    total_pixels = 0
    for path in sorted(resolved, key=lambda value: value.as_posix()):
        if path.suffix.lower() == ".mha":
            decoded, total_pixels = _decode_mha(
                path, limits=limits, total_pixels=total_pixels
            )
        else:
            decoded, total_pixels = _decode_raster(
                path, limits=limits, total_pixels=total_pixels
            )
        pages.extend(decoded)
        if len(pages) > limits.max_pages:
            raise PhotoSetUnavailable("photo_page_count_exceeded")
    if not pages:
        raise PhotoSetUnavailable("photo_set_is_empty")
    return tuple(pages)


def pool_unordered_page_features(page_features: np.ndarray) -> np.ndarray:
    """Match float64 pooling followed by the frozen float32 cache boundary."""
    values = np.asarray(page_features)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != PAGE_WIDTH:
        raise ValueError("page features must have shape [pages, 768]")
    if not np.isfinite(values).all():
        raise ValueError("page features contain non-finite values")
    selected = np.ascontiguousarray(values, dtype="<f8")
    order = sorted(
        range(selected.shape[0]),
        key=lambda index: selected[index].tobytes(order="C"),
    )
    canonical = selected[np.asarray(order, dtype=np.int64)]
    pooled = np.concatenate(
        [canonical.mean(axis=0, dtype=np.float64), canonical.max(axis=0)]
    )
    if pooled.shape != (SET_WIDTH,) or not np.isfinite(pooled).all():
        raise ValueError("pooled photo feature differs from the frozen contract")
    return np.ascontiguousarray(pooled, dtype="<f4")


def validate_feature_contract(contract: Mapping[str, Any]) -> None:
    """Validate the exact frozen encoder identity and preprocessing contract."""
    expected = {
        "canonical_model": "DINOv2 ViT-S/14 distilled without register tokens",
        "checkpoint_files": CHECKPOINT_FILES,
        "dimension": SET_WIDTH,
        "encoder_parameters": ENCODER_PARAMETERS,
        "encoder_trainable_parameters": 0,
        "hidden_size": HIDDEN_SIZE,
        "last_hidden_state_shape": [1, TOKENS_PER_PAGE, HIDDEN_SIZE],
        "model_class": MODEL_CLASS,
        "model_id": MODEL_ID,
        "per_page_dimension": PAGE_WIDTH,
        "per_page_representation": "concatenate_cls_token_and_mean_of_all_patch_tokens_v1",
        "preprocessing": PREPROCESSING,
        "revision": REVISION,
        "set_pooling": "concatenated_mean_and_max_v1",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"photo encoder feature contract drifted: {key}")


def _require_feature_contract(contract: Mapping[str, Any]) -> None:
    """Backward-compatible private entry point for the frozen test surface."""
    validate_feature_contract(contract)


@dataclass
class FrozenDinoV2Small:
    processor: Any
    model: Any
    torch: Any
    batch_size: int = PHYSICAL_BATCH_SIZE

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
        *,
        feature_contract: Mapping[str, Any],
        batch_size: int = PHYSICAL_BATCH_SIZE,
    ) -> "FrozenDinoV2Small":
        validate_feature_contract(feature_contract)
        if batch_size != PHYSICAL_BATCH_SIZE:
            raise ValueError("photo encoder physical batch size must be exactly 16")
        checkpoint = Path(checkpoint_dir)
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError("DINOv2 checkpoint directory is unavailable")
        members = list(checkpoint.iterdir())
        if {path.name for path in members} != set(CHECKPOINT_FILES) or any(
            path.is_symlink() or not path.is_file() for path in members
        ):
            raise ValueError("DINOv2 checkpoint inventory drifted")
        for name, expected in CHECKPOINT_FILES.items():
            path = checkpoint / name
            if path.stat().st_size != expected["bytes"] or _sha256_path(path) != expected["sha256"]:
                raise ValueError(f"DINOv2 checkpoint member drifted: {name}")
        if any(os.environ.get(name) != value for name, value in REQUIRED_PROCESS_ENVIRONMENT.items()):
            raise RuntimeError("deterministic process environment was not fixed")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        random.seed(20_260_804)
        np.random.seed(20_260_804)

        import torch
        from transformers import AutoImageProcessor, AutoModel

        torch.manual_seed(20_260_804)
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        processor = AutoImageProcessor.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        if type(processor).__name__ != PROCESSOR_CLASS:
            raise ValueError("DINOv2 processor class drifted")
        observed_preprocessing = {
            "center_crop": [int(processor.crop_size["height"]), int(processor.crop_size["width"])],
            "convert_rgb": True,
            "image_mean": [float(value) for value in processor.image_mean],
            "image_std": [float(value) for value in processor.image_std],
            "processor_class": type(processor).__name__,
            "resample": f"PIL_BICUBIC_code_{int(processor.resample)}",
            "rescale_factor": float(processor.rescale_factor),
            "resize_shortest_edge": int(processor.size["shortest_edge"]),
        }
        if observed_preprocessing != PREPROCESSING:
            raise ValueError("DINOv2 preprocessing contract drifted")
        model = AutoModel.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to(device="cpu", dtype=torch.float32)
        model.requires_grad_(False)
        model.eval()
        if (
            type(model).__name__ != MODEL_CLASS
            or sum(parameter.numel() for parameter in model.parameters()) != ENCODER_PARAMETERS
            or any(parameter.requires_grad for parameter in model.parameters())
        ):
            raise ValueError("DINOv2 model contract drifted")
        return cls(processor=processor, model=model, torch=torch, batch_size=batch_size)

    def encode_pages(self, pages: Sequence[Image.Image]) -> np.ndarray:
        if (
            not pages
            or len(pages) > PHYSICAL_BATCH_SIZE
            or any(not isinstance(page, Image.Image) or page.mode != "RGB" for page in pages)
        ):
            raise ValueError("photo encoder requires 1--16 RGB PIL pages")
        outputs: list[np.ndarray] = []
        for start in range(0, len(pages), self.batch_size):
            batch = list(pages[start : start + self.batch_size])
            logical_size = len(batch)
            batch.extend([batch[-1]] * (self.batch_size - logical_size))
            pixel_values = self.processor(images=batch, return_tensors="pt")[
                "pixel_values"
            ].to(device="cpu", dtype=self.torch.float32)
            if tuple(pixel_values.shape) != (self.batch_size, 3, 224, 224):
                raise ValueError("DINOv2 processor output shape drifted")
            with self.torch.inference_mode():
                hidden = self.model(pixel_values=pixel_values).last_hidden_state
            if tuple(hidden.shape) != (
                self.batch_size,
                TOKENS_PER_PAGE,
                HIDDEN_SIZE,
            ):
                raise ValueError("DINOv2 token shape drifted")
            page_features = self.torch.cat(
                [hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1
            )[:logical_size]
            if tuple(page_features.shape) != (logical_size, PAGE_WIDTH) or not self.torch.isfinite(page_features).all():
                raise ValueError("DINOv2 page feature drifted")
            outputs.append(
                np.ascontiguousarray(page_features.cpu().numpy(), dtype="<f4")
            )
        return np.concatenate(outputs, axis=0)

    def encode_set(
        self,
        paths: Sequence[Path],
        *,
        limits: PhotoDecodeLimits = PhotoDecodeLimits(),
    ) -> np.ndarray:
        pages = decode_photo_set(paths, limits=limits)
        return pool_unordered_page_features(self.encode_pages(pages))
