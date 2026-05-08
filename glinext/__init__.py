from .config import (
    GLiNextConfig,
    GLiNextAudioConfig,
    BaseHeadConfig,
    NERHeadConfig,
    ClassificationHeadConfig,
    ImageClassificationHeadConfig,
    AudioClassificationHeadConfig,
    GLiNextLayoutConfig,
    GLiNextOmniConfig,
    GLiNextTextConfig,
    GLiNextVisionConfig,
    ObjectDetectionHeadConfig,
    SegmentationHeadConfig,
    AudioSegmentationHeadConfig,
    JointRelexHeadConfig,
    OpenRelexHeadConfig,
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
