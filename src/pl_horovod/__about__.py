__version__ = "0.1.0dev"
__author__ = "Lightning-AI et al."
__author_email__ = "name@lightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2021-2023, {__author__}."
__homepage__ = "https://github.com/Lightning-Devel/PL-Horovod"
__docs__ = "PyTorch Lightning Strategy for Horovod."
# todo: consider loading Readme here...
__long_doc__ = """
Lightning Horovod
-----------------

Horovod allows the same training script to be used for single-GPU, multi-GPU, and multi-node training.

Like Distributed Data Parallel, every process in Horovod operates on a single GPU with a fixed subset of the data.
 Gradients are averaged across all GPUs in parallel during the backward pass,
 then synchronously applied before beginning the next step.

The number of worker processes is configured by a driver application (`horovodrun` or `mpirun`).
 In the training script, Horovod will detect the number of workers from the environment,
 and automatically scale the learning rate to compensate for the increased total batch size.

See the official Horovod documentation for details on installation and performance tuning.
"""

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__long_doc__",
    "__homepage__",
    "__license__",
    "__version__",
]
