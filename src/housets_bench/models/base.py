from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from housets_bench.bundles.datatypes import ProcBundle


class BaseForecaster(ABC):
    name: str = "base"

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        """Optional fit step. Default: no-op."""
        return

    @abstractmethod
    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Predict the next horizon for a single batch in processed space."""
        raise NotImplementedError

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist the trained network to disk. No-op for non-DL models."""
        net = getattr(self, "_net", None)
        if net is None:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net, p)

    def load_checkpoint(self, path: Union[str, Path], *, device: Optional[torch.device] = None) -> None:
        """Restore a previously saved network. No-op for non-DL models."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        net = torch.load(p, map_location=device or "cpu", weights_only=False)
        self._net = net
