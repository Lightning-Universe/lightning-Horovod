"""Configure local testing."""
from lightning_utilities import module_available

if not module_available("horovod"):
    raise ModuleNotFoundError("Horovod is not installed!")
