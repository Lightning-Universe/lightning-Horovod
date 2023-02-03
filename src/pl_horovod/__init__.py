"""Root package info."""

import os

from pl_horovod.__about__ import *  # noqa: F401, F403

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from pl_horovod.strategy import _HOROVOD_AVAILABLE

_HOROVOD_NCCL_AVAILABLE = False
if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd

    try:

        # `nccl_built` returns an integer
        _HOROVOD_NCCL_AVAILABLE = bool(hvd.nccl_built())
    except AttributeError:
        # AttributeError can be raised if MPI is not available:
        # https://github.com/horovod/horovod/blob/v0.23.0/horovod/torch/__init__.py#L33-L34
        pass
