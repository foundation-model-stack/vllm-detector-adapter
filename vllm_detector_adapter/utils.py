# Standard
from enum import Enum, auto


class DetectorType(Enum):
    """Enum to represent different types of detectors"""

    TEXT_CONTENT = auto()
    TEXT_GENERATION = auto()
    TEXT_CHAT = auto()
    TEXT_CONTEXT_DOC = auto()