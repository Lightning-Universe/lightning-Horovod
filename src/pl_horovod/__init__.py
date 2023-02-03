"""Root package info."""
import contextlib
import os

from pl_horovod.__about__ import *  # noqa: F401, F403
from pl_horovod.strategy import _HOROVOD_AVAILABLE

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

_HOROVOD_NCCL_AVAILABLE = False
if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd

    with contextlib.suppress(AttributeError):
        # `nccl_built` returns an integer
        _HOROVOD_NCCL_AVAILABLE = bool(hvd.nccl_built())
        # AttributeError can be raised if MPI is not available:
        # https://github.com/horovod/horovod/blob/v0.23.0/horovod/torch/__init__.py#L33-L34
