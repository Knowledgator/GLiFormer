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
from .processor import GLiNextProcessor
from .decoder import GLiNExTDecoder
from .mappings import (
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

# Backward compat alias
RelationsHeadConfig = JointRelexHeadConfig
