from .config import (
    GLiNextConfig,
    BaseHeadConfig,
    NERHeadConfig,
    ClassificationHeadConfig,
    ImageClassificationHeadConfig,
    AudioClassificationHeadConfig,
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
    GLiNExTOmniModel,
    GLiNExTModel,
    GLiNExTOutput,
    GLiNExTTextAudioModel,
    GLiNExTTextAudioUniEncoderModel,
    GLiNExTTextModel,
    GLiNExTTextVisionBiEncoderModel,
    GLiNExTTextVisionLayoutModel,
    GLiNExTTextVisionUniEncoderModel,
    resolve_glinext_model_class,
)
from .processing.processor import GLiNextProcessor
from .processing.decoder import GLiNExTDecoder
from .processing.collator import GLiNExTDataCollator
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
from .glinext import GLiNExT
from .training import GLiNExTTrainer

# Backward compat alias
RelationsHeadConfig = JointRelexHeadConfig
