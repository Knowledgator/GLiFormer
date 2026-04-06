from .config import (
    GLiNextConfig,
    NERHeadConfig,
    ClassificationHeadConfig,
    JointRelexHeadConfig,
    OpenRelexHeadConfig,
    StructuringHeadConfig,
    CountHeadConfig,
    EmbeddingHeadConfig,
)
from .model import GLiNExTModel, GLiNExTOutput
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
    BatchClassesMapping,
)
from .glinext import GLiNExT
from .training import GLiNExTTrainer

# Backward compat alias
RelationsHeadConfig = JointRelexHeadConfig
