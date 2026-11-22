"""RAC-ReID core package."""

from .image_signals import (
    ImageSignalExtractor,
    ImageSignals,
)
from .pipeline import (
    AdaptiveReIDPipeline,
    DetectorEvidence,
    PipelineResult,
    PipelineStage,
)
from .resource_controller import (
    DifficultyLevel,
    QuerySignals,
    ResourceController,
    ResourcePlan,
    RoutingDecision,
)
from .retrieval import (
    BudgetedVisualEncoder,
    EncodedImage,
    GalleryIndex,
    RetrievalCandidate,
)
from .token_pruning import (
    IdentityAwareTokenPruner,
    PreparedImage,
    TokenSelection,
    VisualToken,
)

__all__ = [
    "AdaptiveReIDPipeline",
    "BudgetedVisualEncoder",
    "DetectorEvidence",
    "DifficultyLevel",
    "EncodedImage",
    "GalleryIndex",
    "IdentityAwareTokenPruner",
    "ImageSignalExtractor",
    "ImageSignals",
    "PipelineResult",
    "PipelineStage",
    "PreparedImage",
    "QuerySignals",
    "ResourceController",
    "ResourcePlan",
    "RetrievalCandidate",
    "RoutingDecision",
    "TokenSelection",
    "VisualToken",
]

__version__ = "0.4.0"
