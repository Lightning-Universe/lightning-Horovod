# Lightning extension: Horovod

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![PyPI Status](https://badge.fury.io/py/lightning-horovod.svg)](https://badge.fury.io/py/lightning-horovod)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-horovod)](https://pypi.org/project/lightning-horovod/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lightning-Horovod)](https://pepy.tech/project/lightning-horovod)
[![Deploy Docs](https://github.com/Lightning-AI/lightning-Horovod/actions/workflows/docs-deploy.yml/badge.svg)](https://lightning-ai.github.io/lightning-Horovod/)

[![General checks](https://github.com/Lightning-Universe/lightning-Horovod/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-Universe/lightning-Horovod/actions/workflows/ci-checks.yml)
[![CI testing](https://github.com/Lightning-Universe/lightning-Horovod/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-Universe/lightning-Horovod/actions/workflows/ci-testing.yml)
[![Build Status](https://dev.azure.com/Lightning-AI/compatibility/_apis/build/status%2Fstrategies%2FLightning-Universe.lightning-Horovod?branchName=main)](https://dev.azure.com/Lightning-AI/compatibility/_build/latest?definitionId=69&branchName=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-Universe/lightning-Horovod/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-Universe/lightning-Horovod/main)

[Horovod](http://horovod.ai) allows the same training script for single-GPU, multi-GPU, and multi-node training.

Like Distributed Data-Parallel, Horovod's processes operate on a single GPU with a fixed subset of the data.  Gradients are averaged across all GPUs in parallel during the backward pass, then synchronously applied before beginning the next step.

The number of worker processes is configured by a driver application (`horovodrun` or `mpirun`). Horovod will detect the number of workers from the environment in the training script and automatically scale the learning rate to compensate for the increased total batch size.

Horovod can be configured in the training script to run with any number of GPUs / processes as follows:

```py
from lightning import Trainer
from lightning_horovod import HorovodStrategy

# train Horovod on GPU (number of GPUs / machines provided on command-line)
trainer = Trainer(strategy="horovod", accelerator="gpu", devices=1)

# train Horovod on CPU (number of processes/machines provided on command-line)
trainer = Trainer(strategy=HorovodStrategy())
```

When starting the training job, the driver application will then be used to specify the total number of worker processes:

```bash
# run training with 4 GPUs on a single machine
horovodrun -np 4 python train.py

# run training with 8 GPUs on two machines (4 GPUs each)
horovodrun -np 8 -H hostname1:4,hostname2:4 python train.py
```

See the official [Horovod documentation](https://horovod.readthedocs.io/en/stable) for installation and performance tuning details.
