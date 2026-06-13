from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


MAMBA_DEPENDENCY_MODULES = {
    "einops": "einops",
    "causal-conv1d": "causal_conv1d",
    "mamba-ssm": "mamba_ssm",
}


class MambaDependencyError(RuntimeError):
    """Raised when optional Mamba packages are missing or incompatible.

    Mamba support is intentionally optional in the public repo because the
    packages are CUDA/toolchain-sensitive.  Classical/CNN/LSTM/Transformer
    verification should still work on a normal laptop, while full Mamba reruns
    require the documented GPU environment.
    """

    pass


@dataclass(frozen=True)
class MambaImportTarget:
    module_name: str
    class_name: str


MAMBA_IMPORT_TARGETS = {
    "mamba1": [
        MambaImportTarget("mamba_ssm", "Mamba"),
        MambaImportTarget("mamba_ssm.modules.mamba_simple", "Mamba"),
    ],
    "mamba2": [
        MambaImportTarget("mamba_ssm", "Mamba2"),
        MambaImportTarget("mamba_ssm.modules.mamba2", "Mamba2"),
        MambaImportTarget("mamba_ssm.modules.mamba2_simple", "Mamba2"),
    ],
}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _missing_dependency_names() -> list[str]:
    return [package_name for package_name, module_name in MAMBA_DEPENDENCY_MODULES.items() if not _module_available(module_name)]


def _import_mamba_class(version: str) -> type[nn.Module]:
    missing = _missing_dependency_names()
    if missing:
        raise MambaDependencyError(f"missing Mamba dependencies: {', '.join(missing)}")

    errors: list[str] = []
    for target in MAMBA_IMPORT_TARGETS[version]:
        try:
            module = importlib.import_module(target.module_name)
            mamba_class = getattr(module, target.class_name)
            return mamba_class
        except Exception as exc:  # pragma: no cover - depends on optional package layout
            errors.append(f"{target.module_name}.{target.class_name}: {exc}")

    raise MambaDependencyError(f"could not import {version} class from mamba-ssm: {'; '.join(errors)}")


def check_mamba_dependencies(version: str) -> dict[str, Any]:
    """Return a non-throwing dependency report for setup/debug scripts."""

    dependency_status = {
        package_name: {
            "module": module_name,
            "available": _module_available(module_name),
        }
        for package_name, module_name in MAMBA_DEPENDENCY_MODULES.items()
    }
    result: dict[str, Any] = {
        "passed": False,
        "version": version,
        "dependencies": dependency_status,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    missing = [name for name, status in dependency_status.items() if not status["available"]]
    if missing:
        result["reason"] = f"missing packages: {', '.join(missing)}"
        result["install_hint"] = (
            "install einops first, then build causal-conv1d and mamba-ssm from source in WSL2 "
            "against the active pytorch cuda environment for the local gpu architecture"
        )
        return result

    try:
        imported_class = _import_mamba_class(version)
    except MambaDependencyError as exc:
        result["reason"] = str(exc)
        return result

    result["passed"] = True
    result["imported_class"] = f"{imported_class.__module__}.{imported_class.__name__}"
    return result


class MambaResidualBlock(nn.Module):
    def __init__(
        self,
        version: str,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        mamba_class = _import_mamba_class(version)
        self.norm = nn.LayerNorm(d_model)
        self.mixer = mamba_class(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pre-norm residual: normalize, mix, drop, then add back the input.
        return inputs + self.dropout(self.mixer(self.norm(inputs)))


# mamba baseline for genre classification from mel spectrograms. we wrap
# mamba-1 and mamba-2 in the same class so differences in results come from
# the mixer family, not from divergent preprocessing or pooling code.
class MambaClassifier(nn.Module):
    def __init__(
        self,
        version: str,
        num_classes: int = 16,
        input_dim: int = 128,
        d_model: int = 256,
        num_layers: int = 5,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if version not in MAMBA_IMPORT_TARGETS:
            raise ValueError(f"unsupported mamba version: {version}")
        self.version = version
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    version=version,
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _encode(self, inputs: torch.Tensor, collect_intermediate: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # as with the lstm and transformer, time frames become sequence tokens.
        tokens = inputs.transpose(1, 2)
        tokens = self.input_projection(tokens)
        tokens = self.input_dropout(tokens)

        intermediates: dict[str, torch.Tensor] = {}
        middle_block = max(1, len(self.blocks) // 2)
        for index, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if collect_intermediate and index == middle_block:
                intermediates[f"block_{index}_mean"] = tokens.mean(dim=1)

        tokens = self.final_norm(tokens)
        # we use mean pooling deliberately so the selective-state mixer
        # carries the modeling burden, not an attention-style pooling head.
        pooled = tokens.mean(dim=1)
        if collect_intermediate:
            intermediates["penultimate"] = pooled
        return pooled, intermediates

    def get_penultimate(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled, _ = self._encode(inputs)
        return pooled

    def get_intermediate_representations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        _, intermediates = self._encode(inputs, collect_intermediate=True)
        return intermediates

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        penultimate = self.get_penultimate(inputs)
        return self.head(penultimate)


class Mamba1Classifier(MambaClassifier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(version="mamba1", d_state=16, **kwargs)


class Mamba2Classifier(MambaClassifier):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(version="mamba2", d_state=64, **kwargs)
