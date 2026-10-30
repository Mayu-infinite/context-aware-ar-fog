"""Label-free, training-free spatial value engine for context-aware FoG cueing."""

from cafog.spatial.config import EngineConfig, PatientProfile, make_profile
from cafog.spatial.engine import DepthFrame, EngineResult, SpatialCueEngine

__all__ = [
    "DepthFrame",
    "EngineConfig",
    "EngineResult",
    "PatientProfile",
    "SpatialCueEngine",
    "make_profile",
]
