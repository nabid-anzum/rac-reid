"""RAC-ReID core package."""

from .image_signals import (
    ImageSignalExtractor,
    ImageSignals,
)
from .learned_reid import (
    LearnedCandidate,
    LearnedReIDResult,
    OSNetReIDEncoder,
)
from .person_detector import (
    NoPersonDetectedError,
    PersonDetection,
    PersonDetectionResult,
    PersonQueryPreprocessor,
    PreparedPersonQuery,
    YOLOPersonDetector,
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
    "LearnedCandidate",
    "LearnedReIDResult",
    "NoPersonDetectedError",
    "OSNetReIDEncoder",
    "PersonDetection",
    "PersonDetectionResult",
    "PersonQueryPreprocessor",
    "PipelineResult",
    "PipelineStage",
    "PreparedImage",
    "PreparedPersonQuery",
    "QuerySignals",
    "ResourceController",
    "ResourcePlan",
    "RetrievalCandidate",
    "RoutingDecision",
    "TokenSelection",
    "VisualToken",
    "YOLOPersonDetector",
]

__version__ = "0.7.0"
