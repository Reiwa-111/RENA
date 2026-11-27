"""RENA-RUNE: mini GQA/SwiGLU language model for the Deltarune transcript."""

from .config import load_config, validate_config, config_hash
from .model import RENAForCausalLM, model_from_config
from .tokenizer import load_tokenizer, train_tokenizer

__all__ = [
    "load_config",
    "validate_config",
    "config_hash",
    "RENAForCausalLM",
    "model_from_config",
    "load_tokenizer",
    "train_tokenizer",
]

__version__ = "0.1"
