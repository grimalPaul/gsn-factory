from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .gsn_criterion import AbstractGSN


@dataclass
class IterefConfig:
    """Configuration for Iterative Refinement."""

    criterion: Optional[AbstractGSN] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    steps: Optional[List[int]] = None
    scale_range: Optional[List[float]] = None
    scale_factor: Optional[float] = None
    max_opti: Optional[int] = None
    optimizer_class: Optional[str] = None
    thresholds: Optional[Dict[int, float]] = None


@dataclass
class GsngConfig:
    """Configuration for GSN Guidance."""

    criterion: Optional[AbstractGSN] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    steps: Optional[List[int]] = None
    scale_range: Optional[List[float]] = None
    scale_factor: Optional[float] = None


@dataclass
class DistribConfig:
    """Configuration for Distribution-based SAGA."""

    criterion: Optional[AbstractGSN] = None
    one_image_per_distrib: bool = False
    batch_size_noise: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    step: Optional[int] = None
    optimizer_class: Optional[str] = "SGD"
    step_size: Optional[float] = 20
    max_opti: Optional[int] = None
    init_mu: Optional[str] = None
    block: Optional[int] = None
    log_var: bool = True
    per_channel: bool = False
    rescale: bool = False
    momentum_saga: float = 0.0
