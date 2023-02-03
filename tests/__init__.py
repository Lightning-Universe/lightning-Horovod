"""Configure local testing."""

from pl_horovod import _HOROVOD_AVAILABLE

if not _HOROVOD_AVAILABLE:
    raise ModuleNotFoundError("Horovod is not installed!")
