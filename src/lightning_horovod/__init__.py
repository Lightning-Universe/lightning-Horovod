"""Root package info."""
import os

from lightning_horovod.__about__ import *  # noqa: F401, F403
from lightning_horovod.strategy import HorovodStrategy

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

__all__ = ["HorovodStrategy"]
