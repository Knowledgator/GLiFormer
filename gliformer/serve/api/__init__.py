"""Unified analyze API (``POST /v1/analyze``) for GLiFormer serving.

See ``NEW_API.md`` for the contract. The modules are independent of Ray so
they can be used in-process and tested without a model:

* ``render``       input → one Markdown text with leaf offsets
* ``labels``       label forms and rendered label text
* ``jsonfields``   ``json`` task fields, model templates, output shaping
* ``request``      request validation into task specs
* ``engine``       row planning, execution, result assembly
* ``capabilities`` what the loaded checkpoint can run
"""

from .capabilities import Capabilities, capabilities_from_model
from .engine import AnalyzeEngine
from .errors import ERROR_STATUS, APIError
from .render import RenderedInput, render_input

__all__ = [
    "APIError",
    "AnalyzeEngine",
    "Capabilities",
    "ERROR_STATUS",
    "RenderedInput",
    "capabilities_from_model",
    "render_input",
]
