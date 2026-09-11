try:
    import wcwidth

    if not hasattr(wcwidth, "wcswidth"):
        wcwidth.wcswidth = lambda text: sum(  # type: ignore[attr-defined]
            max(getattr(wcwidth, "wcwidth", lambda _: 1)(char), 0)
            for char in str(text)
        )
    if not hasattr(wcwidth, "wcwidth"):
        wcwidth.wcwidth = lambda char: 1  # type: ignore[attr-defined]
except Exception:
    pass

from .config import (
    GLiFormerConfig,
    GLiFormerAudioConfig,
    TaskHeadConfig,
    AnchorHeadConfig,
    BaseHeadConfig,
    NERHeadConfig,
    ClassificationHeadConfig,
    MediaClassificationHeadConfig,
    ImageClassificationHeadConfig,
    AudioClassificationHeadConfig,
    SetPredictionHeadConfig,
    GLiFormerLayoutConfig,
    GLiFormerOmniConfig,
    GLiFormerTextConfig,
    GLiFormerVisionConfig,
    ObjectDetectionHeadConfig,
    SegmentationHeadConfig,
    AudioSegmentationHeadConfig,
    JointRelexHeadConfig,
    OpenRelexHeadConfig,
    StructuringModeConfig,
    StructuringHeadConfig,
    CountHeadConfig,
    EmbeddingHeadConfig,
)
from .model import (
    BaseGLiFormerModel,
    GLiFormerAudioModel,
    GLiFormerAudioOutput,
    GLiFormerLayoutModel,
    GLiFormerLayoutOutput,
    GLiFormerOmniModel,
    GLiFormerOmniOutput,
    GLiFormerModel,
    GLiFormerOutput,
    GLiFormerTextModel,
    GLiFormerTextOutput,
    GLiFormerVisionModel,
    GLiFormerVisionOutput,
    resolve_gliformer_model_class,
)
from .processing.processor import GLiFormerProcessor
from .processing.pdf import GLiFormerPDFProcessor, PDFTableProcessor
from .processing.decoder import GLiFormerDecoder
from .processing.collator import (
    BaseGLiFormerDataCollator,
    GLiFormerAudioDataCollator,
    GLiFormerDataCollator,
    GLiFormerLayoutDataCollator,
    GLiFormerOmniDataCollator,
    GLiFormerTextDataCollator,
    GLiFormerVisionDataCollator,
    resolve_gliformer_collator_class,
)
from .processing.schema import GLiFormerSchema
from .processing.formatting import StructuringOutputFormatter, FieldType
from .processing.mappings import (
    BaseClassMapping,
    CatClassMapping,
    ExtractionItemMapping,
    ExtractionClassMapping,
    StructuringItemMapping,
    StructuringClassMapping,
    OpenRelexItemMapping,
    OpenRelexClassMapping,
    VisionItemMapping,
    VisionClassMapping,
    BatchClassesMapping,
)
from .gliformer import (
    BaseGLiFormer,
    GLiFormer,
    GLiFormerAudio,
    GLiFormerLayout,
    GLiFormerOmni,
    GLiFormerText,
    GLiFormerVision,
)
from .training import GLiFormerTrainer

# Backward compat alias
RelationsHeadConfig = JointRelexHeadConfig
