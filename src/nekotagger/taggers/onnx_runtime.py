"""Shared ONNX Runtime session and device handling."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import onnxruntime as ort

from nekotagger.errors import InputError, ModelError

_DEVICE_PATTERN = re.compile(r"^cuda(?::(?P<index>[0-9]+))?$")


def _parse_device(device: str) -> tuple[bool, bool, int]:
    if device == "auto":
        return True, False, 0
    if device == "cpu":
        return False, True, 0
    match = _DEVICE_PATTERN.fullmatch(device)
    if match:
        index = int(match.group("index") or 0)
        return True, True, index
    raise InputError("device must be auto, cpu, cuda, or cuda:N")


def _initialize_session(
    model_path: Path,
    providers: list[str | tuple[str, dict[str, str]]],
) -> ort.InferenceSession:
    last_error: Exception | None = None
    for optimization in (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = optimization
        try:
            return ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=providers,
            )
        except Exception as error:
            last_error = error
    assert last_error is not None
    raise ModelError(f"failed to initialize ONNX model: {last_error}") from last_error


def create_session(
    model_path: Path,
    device: str,
) -> tuple[ort.InferenceSession, str]:
    """Initialize an ONNX session and return its normalized active device."""

    wants_cuda, strict, device_index = _parse_device(device)
    if wants_cuda and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except Exception as error:
            if strict:
                raise ModelError(f"failed to preload CUDA/cuDNN libraries: {error}") from error
            warnings.warn(
                f"could not preload CUDA/cuDNN libraries; CPU fallback may be used: {error}",
                stacklevel=2,
            )

    available = ort.get_available_providers()
    providers: list[str | tuple[str, dict[str, str]]]
    if wants_cuda and "CUDAExecutionProvider" in available:
        providers = [
            ("CUDAExecutionProvider", {"device_id": str(device_index)}),
            "CPUExecutionProvider",
        ]
    elif wants_cuda and strict:
        raise ModelError(
            "CUDAExecutionProvider is unavailable; "
            "use --device cpu or repair CUDA/driver access"
        )
    else:
        if wants_cuda:
            warnings.warn(
                "CUDAExecutionProvider is unavailable; falling back to CPU",
                stacklevel=2,
            )
        providers = ["CPUExecutionProvider"]

    session = _initialize_session(model_path, providers)
    session_providers = session.get_providers()
    if wants_cuda and "CUDAExecutionProvider" not in session_providers:
        if strict:
            raise ModelError(
                "ONNX Runtime initialized without CUDAExecutionProvider; "
                "use --device cpu or repair CUDA/cuDNN access"
            )
        warnings.warn(
            "ONNX Runtime could not initialize CUDA; falling back to CPU",
            stacklevel=2,
        )
        active_device = "cpu"
    elif wants_cuda:
        active_device = f"cuda:{device_index}"
    else:
        active_device = "cpu"
    return session, active_device
