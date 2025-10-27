

### `src/qdl/__init__.py`

from .config import ConfigBundle, load_config
from .data import create_dataloaders
from .model import QuantumDeepLearningModel
from .trainer import QDLTrainer

__all__ = [
    "ConfigBundle",
    "load_config",
    "create_dataloaders",
    "QuantumDeepLearningModel",
    "QDLTrainer",
]