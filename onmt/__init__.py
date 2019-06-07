import onmt.inputters
import onmt.encoders
import onmt.decoders
import onmt.models
import onmt.utils
import onmt.modules
from onmt.trainer import Trainer

# For Flake
__all__ = [onmt.inputters, onmt.encoders, onmt.decoders, onmt.models,
           onmt.utils, onmt.modules, "Trainer"]

__version__ = "0.6.0"
