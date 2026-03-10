from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from oceanbench_core.eval_grid import EvalGrid

ArrayLike = np.ndarray


class CovarianceBackend(ABC):
    """
    Common interface for covariance backends used by Binney-style objectives.

    The API is deliberately NumPy-based and operates on feature matrices rather
    than higher-level measurement objects. Callers are responsible for mapping
    their measurement items or evaluation grids to feature matrices with a
    consistent convention, e.g.:

    - columns [lat, lon]          for static fields, or
    - columns [lat, lon, time]    for spatio-temporal kernels.
    """

    @abstractmethod
    def cov_block(self, Xa: ArrayLike, Xb: ArrayLike) -> ArrayLike:
        """
        Return the covariance block Σ(Xa, Xb).

        Parameters
        ----------
        Xa, Xb:
            Feature matrices with shapes (Na, D) and (Nb, D).
        """

    def diag_cov(self, X: ArrayLike) -> ArrayLike:
        """
        Optional acceleration: diagonal of Σ(X, X).

        Default implementation derives it from `cov_block`; subclasses may
        override with cheaper computations.
        """
        K = self.cov_block(X, X)
        return np.asarray(np.diag(K), dtype=float)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """
        Optional hyperparameter fitting hook.

        Implementations that support hyperparameter learning (e.g. kernel
        backends using GP marginal likelihood) should override this. Backends
        that do not require fitting may ignore it.
        """
        _ = (X, y)  # unused by default


@dataclass
class CovarianceConfig:
    """
    Configuration bundle for covariance backends.

    This is a thin, strongly-typed wrapper around a mapping so that higher
    layers can pass in structured configuration without depending on any
    particular YAML loader.
    """

    backend: str = "kernel"  # "kernel" or "empirical"
    kernel: Mapping[str, Any] = None  # type: ignore[assignment]
    empirical: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kernel is None:
            self.kernel = {}
        if self.empirical is None:
            self.empirical = {}


def parse_covariance_config(cfg: Mapping[str, Any]) -> CovarianceConfig:
    """
    Parse a loosely-structured config mapping into a CovarianceConfig.

    Expected structure (conceptual; actual YAML nesting may vary):

    .. code-block:: yaml

        covariance:
          backend: kernel  # or empirical
          kernel:
            lengthscale_space: 1.0
            lengthscale_time: 1.0
            variance: 1.0
            fit: true
            fit:
              max_iters: 50
              subsample_n: 512
          empirical:
            window_days: 30
            stride: "1d"
            max_snapshots: 50
            use_anomalies: true
    """
    # Accept either a top-level mapping or a nested "covariance" section.
    root = cfg.get("covariance", cfg)
    backend = str(root.get("backend", "kernel"))
    kernel_cfg = root.get("kernel", {})
    empirical_cfg = root.get("empirical", {})
    return CovarianceConfig(
        backend=backend,
        kernel=kernel_cfg,
        empirical=empirical_cfg,
    )


def build_covariance_backend(
    cfg: Mapping[str, Any],
    *,
    eval_grid: Optional[EvalGrid] = None,
    provider: Optional[object] = None,
) -> CovarianceBackend:
    """
    Factory for covariance backends.

    Parameters
    ----------
    cfg:
        Configuration mapping (either the full experiment config or the
        ``covariance`` subsection).
    eval_grid:
        Evaluation grid used to define the target set Y. Required for the
        empirical backend.
    provider:
        Optional data provider object; required when building the empirical
        backend that needs access to historical truth snapshots.
    """
    from .kernel_st import KernelSTCovariance
    from .empirical_cov import EmpiricalCovarianceBackend

    cov_cfg = parse_covariance_config(cfg)

    if cov_cfg.backend == "kernel":
        return KernelSTCovariance(config=cov_cfg.kernel)

    if cov_cfg.backend == "empirical":
        if eval_grid is None or provider is None:
            raise ValueError(
                "Empirical covariance backend requires both eval_grid and provider."
            )
        return EmpiricalCovarianceBackend.from_provider(
            eval_grid=eval_grid,
            provider=provider,
            config=cov_cfg.empirical,
        )

    raise ValueError(f"Unknown covariance backend: {cov_cfg.backend!r}")

