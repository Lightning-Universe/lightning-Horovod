# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import horovod.torch as hvd_torch
import numpy as np
import pytest
import torch
from lightning_utilities import module_available
from sklearn.metrics import accuracy_score
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy

from lightning_horovod.strategy import _HOROVOD_NCCL_AVAILABLE, HorovodStrategy
from tests.helpers import _run_horovod, run_model_test_without_loggers

if module_available("lightning"):
    from lightning.pytorch import Trainer
    from lightning.pytorch.accelerators import CUDAAccelerator
    from lightning.pytorch.demos.boring_classes import BoringModel
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer
    from pytorch_lightning.accelerators import CUDAAccelerator
    from pytorch_lightning.demos.boring_classes import BoringModel
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

if torch.cuda.is_available():
    assert _HOROVOD_NCCL_AVAILABLE, "for GPU tests we shall use Horovod with NCCL support..."


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test needs at least 2 GPUs.")
def test_nccl_is_available_on_gpu_environment():
    # the GPU environment should always install Horovod NCCL
    assert _HOROVOD_NCCL_AVAILABLE


@pytest.mark.xfail(raises=AssertionError, reason="unhandled cuda error")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.skipif(not _HOROVOD_NCCL_AVAILABLE, reason="This test requires NCCL support.")
def test_multi_gpu(tmpdir):
    """Test Horovod with multi-GPU support."""
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "gradient_clip_val": 1.0,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
        "accelerator": "gpu",
        "devices": 2,
    }
    _run_horovod(trainer_options)


@pytest.mark.xfail(raises=AssertionError, reason="unhandled cuda error")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.skipif(not _HOROVOD_NCCL_AVAILABLE, reason="This test requires NCCL support.")
def test_multi_gpu_accumulate_grad_batches(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 4,
        "limit_val_batches": 0,
        "accumulate_grad_batches": 2,
        "accelerator": "gpu",
        "devices": 2,
    }
    _run_horovod(trainer_options)


@pytest.mark.xfail(raises=AssertionError, reason="unhandled cuda error")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.skipif(not _HOROVOD_NCCL_AVAILABLE, reason="This test requires NCCL support.")
def test_multi_gpu_grad_by_value(tmpdir):
    """Test Horovod with multi-GPU support."""
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "value",
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
        "accelerator": "gpu",
        "devices": 2,
    }
    _run_horovod(trainer_options)


@pytest.mark.xfail(raises=AssertionError, reason="unhandled cuda error")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.skipif(not _HOROVOD_NCCL_AVAILABLE, reason="This test requires NCCL support.")
def test_amp(tmpdir):
    """Test Horovod with multi-GPU support using native amp."""
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "gradient_clip_val": 1.0,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
        "accelerator": "gpu",
        "devices": 2,
        "precision": 16,
    }
    _run_horovod(trainer_options)


@pytest.mark.xfail(raises=AssertionError, reason="unhandled cuda error")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.skipif(not _HOROVOD_NCCL_AVAILABLE, reason="This test requires NCCL support.")
def test_gather(tmpdir):
    """Test Horovod with multi-GPU support using native amp."""
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "gradient_clip_val": 1.0,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
        "accelerator": "gpu",
        "devices": 2,
    }
    _run_horovod(trainer_options)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.skipif(not _HOROVOD_NCCL_AVAILABLE, reason="This test requires NCCL support.")
def test_transfer_batch_to_gpu(tmpdir):
    class TestTrainingStepModel(BoringModel):
        automatic_optimization: bool = False

        def training_step(self, batch, *args, **kwargs):
            assert str(batch.device) != "cpu"
            return super().training_step(batch, *args, **kwargs)

        def validation_step(self, batch, *args, **kwargs):
            assert str(batch.device) != "cpu"
            return super().validation_step(batch, *args, **kwargs)

    model = TestTrainingStepModel()

    trainer_options = {
        "default_root_dir": str(tmpdir),
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
        "accelerator": "gpu",
        "devices": 2,
        "strategy": HorovodStrategy(),
    }
    run_model_test_without_loggers(trainer_options, model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.skipif(not module_available("sklearn"), reason="This tests scikit-learn accuracy.")
def test_accuracy_metric_horovod():
    import horovod

    num_batches = 20
    batch_size = 512
    threshold = 0.5

    def sk_metric(preds, target):
        sk_preds = (preds.view(-1).numpy() >= threshold).astype(np.uint8)
        sk_target = target.view(-1).numpy()
        return accuracy_score(y_true=sk_target, y_pred=sk_preds)

    preds = torch.rand(num_batches, batch_size)
    target = torch.randint(high=2, size=(num_batches, batch_size))

    def _compute_batch():
        trainer = Trainer(fast_dev_run=True, strategy=HorovodStrategy(), logger=False)

        assert isinstance(trainer.accelerator, CUDAAccelerator)
        # TODO: test that we selected the correct strategy based on horovod flags

        metric = Accuracy(
            compute_on_step=True,
            dist_sync_on_step=True,
            dist_sync_fn=trainer.strategy.all_gather,
            threshold=threshold,
        )

        for i in range(hvd_torch.rank(), num_batches, hvd_torch.size()):
            batch_result = metric(preds[i], target[i])
            if hvd_torch.rank() != 0:
                continue
            dist_preds = torch.stack([preds[i + r] for r in range(hvd_torch.size())])
            dist_target = torch.stack([target[i + r] for r in range(hvd_torch.size())])
            sk_batch_result = sk_metric(dist_preds, dist_target)
            assert torch.allclose(
                batch_result,
                torch.tensor(sk_batch_result, dtype=batch_result.dtype),
                atol=0.05,  # todo: this shall be almost zero
            ), f"with results: {batch_result}\n SK ref: {sk_batch_result}"

        # check on all batches on all ranks
        result = metric.compute()
        assert isinstance(result, Tensor)

        total_preds = torch.stack([preds[i] for i in range(num_batches)])
        total_target = torch.stack([target[i] for i in range(num_batches)])
        sk_result = sk_metric(total_preds, total_target)

        assert np.allclose(result.numpy(), sk_result, atol=0.05)

    horovod.run(_compute_batch, np=2)
