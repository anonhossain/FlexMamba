"""
FlexMamba training package.

This package contains only training-related functionality.

Entry points:
    src/run_train.py
    src/run_pipeline.py
"""

from training.setup import (
    build_dataloaders,
    build_model,
    choose_device,
    initialize_model,
    make_optimizer,
    make_scheduler,
    set_seed,
)

from training.trainer import train_model
from training.validation import validate_language_model


__all__ = [
    "build_dataloaders",
    "build_model",
    "choose_device",
    "initialize_model",
    "make_optimizer",
    "make_scheduler",
    "set_seed",
    "train_model",
    "validate_language_model",
]