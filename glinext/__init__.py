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
    GLiNextConfig,
    GLiNextAudioConfig,
    TaskHeadConfig,
    AnchorHeadConfig,
    BaseHeadConfig,
    NERHeadConfig,
    ClassificationHeadConfig,
    MediaClassificationHeadConfig,
    ImageClassificationHeadConfig,
    AudioClassificationHeadConfig,
    SetPredictionHeadConfig,
    GLiNextLayoutConfig,
    GLiNextOmniConfig,
    GLiNextTextConfig,
    GLiNextVisionConfig,
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
    BaseGLiNextModel,
    GLiNExTAudioModel,
    GLiNExTAudioOutput,
    GLiNExTLayoutModel,
    GLiNExTLayoutOutput,
    GLiNExTOmniModel,
    GLiNExTOmniOutput,
    GLiNExTModel,
    GLiNExTOutput,
    GLiNExTTextModel,
    GLiNExTTextOutput,
    GLiNExTVisionModel,
    GLiNExTVisionOutput,
    resolve_glinext_model_class,
)
from .processing.processor import GLiNextProcessor
from .processing.pdf import GLiNextPDFProcessor, PDFTableProcessor
from .processing.decoder import GLiNExTDecoder
from .processing.collator import (
    BaseGLiNExTDataCollator,
    GLiNExTAudioDataCollator,
    GLiNExTDataCollator,
    GLiNExTLayoutDataCollator,
    GLiNExTOmniDataCollator,
    GLiNExTTextDataCollator,
    GLiNExTVisionDataCollator,
    resolve_glinext_collator_class,
)
from .processing.schema import GLiNExTSchema
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
from .glinext import (
    BaseGLiNeXT,
    BaseGLiNExT,
    GLiNExT,
    GLiNExTAudio,
    GLiNExTLayout,
    GLiNExTOmni,
    GLiNExTText,
    GLiNExTVision,
)
from .training import GLiNExTTrainer

# Backward compat alias
RelationsHeadConfig = JointRelexHeadConfig
