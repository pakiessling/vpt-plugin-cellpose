from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CellposeSegProperties:
    model: str
    model_dimensions: str
    version: str
    custom_weights: Optional[str] = None


@dataclass(frozen=True)
class CellposeSegParameters:
    nuclear_channel: str
    entity_fill_channel: str
    diameter: int
    flow_threshold: float
    cellprob_threshold: float
    minimum_mask_size: int
    normalize: bool
    percentile_low: float
    percentile_high: float
