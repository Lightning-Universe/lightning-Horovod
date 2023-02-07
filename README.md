# Lightning extension: Horovod

[![CI testing](https://github.com/Lightning-AI/lightning-Horovod/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-Horovod/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-Horovod/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-Horovod/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-Horovod/badge/?version=latest)](https://lightning-Horovod.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-Horovod/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-Horovod/main)

[Horovod](http://horovod.ai) allows the same training script to be used for single-GPU, multi-GPU, and multi-node training.

Like Distributed Data Parallel, every process in Horovod operates on a single GPU with a fixed subset of the data.  Gradients are averaged across all GPUs in parallel during the backward pass, then synchronously applied before beginning the next step.

The number of worker processes is configured by a driver application (`horovodrun` or `mpirun`). In the training script, Horovod will detect the number of workers from the environment, and automatically scale the learning rate to compensate for the increased total batch size.

Horovod can be configured in the training script to run with any number of GPUs / processes as follows:

```py
# train Horovod on GPU (number of GPUs / machines provided on command-line)
trainer = Trainer(strategy="horovod", accelerator="gpu", devices=1)

# train Horovod on CPU (number of processes / machines provided on command-line)
trainer = Trainer(strategy="horovod")
```

When starting the training job, the driver application will then be used to specify the total number of worker processes:

```bash
# run training with 4 GPUs on a single machine
horovodrun -np 4 python train.py

# run training with 8 GPUs on two machines (4 GPUs each)
horovodrun -np 8 -H hostname1:4,hostname2:4 python train.py
```

See the official [Horovod documentation](https://horovod.readthedocs.io/en/stable) for details on installation and performance tuning.

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
