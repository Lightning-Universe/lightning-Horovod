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
import os
import sys
from unittest.mock import patch

import horovod
import horovod.torch as hvd
import pytest
import torch
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")


from lightning_horovod.strategy import HorovodStrategy
from tests.helpers import BasicGAN, _run_horovod


def test_simple(tmpdir):
    """Test Horovod running multi-process on CPU."""
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "gradient_clip_val": 1.0,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
    }
    _run_horovod(trainer_options)


def test_accumulate_grad_batches(tmpdir):
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 4,
        "limit_val_batches": 0,
        "accumulate_grad_batches": 2,
    }
    _run_horovod(trainer_options)


def test_clip_grad_by_value(tmpdir):
    """Test Horovod running multi-process on CPU."""
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "value",
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
    }
    _run_horovod(trainer_options)


def test_implicit(tmpdir):
    """Test Horovod without specifying a backend, inferring from env set by `horovodrun`."""
    trainer_options = {
        "default_root_dir": str(tmpdir),
        "gradient_clip_val": 1.0,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.2,
    }
    _run_horovod(trainer_options)


@pytest.mark.xfail(
    raises=RuntimeError, reason="Training with multiple optimizers is only supported with manual optimization"
)
def test_multi_optimizer(tmpdir):
    model = BasicGAN()

    # fit model
    trainer = Trainer(
        default_root_dir=str(tmpdir),
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        strategy=HorovodStrategy(),
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    assert len(trainer.optimizers) == 2
    for i, optimizer in enumerate(trainer.optimizers):
        assert hasattr(optimizer, "synchronize"), "optimizer has not been wrapped into DistributedOptimizer"

    def get_model_params(model):
        return set(model.parameters())

    def get_optimizer_params(optimizer):
        return {p for group in optimizer.param_groups for p in group.get("params", [])}

    assert get_model_params(model.generator) != get_model_params(model.discriminator)
    assert get_model_params(model.generator) == get_optimizer_params(trainer.optimizers[0])
    assert get_model_params(model.discriminator) == get_optimizer_params(trainer.optimizers[1])


@pytest.mark.skip(reason="TODO: CI agent.jobstatus=Succeeded: Permission denied")
def test_result_reduce_horovod(tmpdir):
    """Make sure result logging works with Horovod.

    This test mirrors tests/core/test_results.py::_ddp_test_fn
    """

    def hvd_test_fn():
        path_here = os.path.abspath(os.path.dirname(__file__))
        path_root = os.path.abspath(os.path.join(path_here, "..", ".."))
        sys.path.insert(0, os.path.abspath(path_root))

        class TestModel(BoringModel):
            def training_step(self, batch, batch_idx):
                self.training_step_called = True

                tensor = torch.tensor([1.0])
                self.log("test_tensor", tensor, sync_dist=True, reduce_fx="sum", on_step=True, on_epoch=True)

                res = self._results

                # Check that `tensor` is summed across all ranks automatically
                assert (
                    res["test_tensor"].item() == hvd.size()
                ), "Result-Log does not work properly with Horovod and Tensors"

            # ToDo: find alternative for this check
            # def training_epoch_end(self, outputs) -> None:
            #     assert len(outputs) == 0

        model = TestModel()
        model.val_dataloader = None

        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=2,
            limit_val_batches=2,
            max_epochs=1,
            log_every_n_steps=1,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model)

    horovod.run(hvd_test_fn, np=2)


@pytest.mark.xfail(  # for PL
    raises=RuntimeError, reason="Training with multiple optimizers is only supported with manual optimization"
)
@pytest.mark.xfail(raises=TypeError, reason="list indices must be integers or slices, not str")  # for LAi
def test_multi_optimizer_with_scheduling_stepping(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer1 = torch.optim.Adam(self.parameters(), lr=0.1)
            optimizer2 = torch.optim.Adam(self.parameters(), lr=0.1)
            lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    model = TestModel()

    num_workers = 8
    init_lr = 0.1 * num_workers

    with patch("horovod.torch.size", return_value=8):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            limit_val_batches=0.5,
            limit_train_batches=0.2,
            strategy=HorovodStrategy(),
        )
        trainer.fit(model)

    adjusted_lr1 = [pg["lr"] for pg in trainer.optimizers[0].param_groups][0]
    adjusted_lr2 = [pg["lr"] for pg in trainer.optimizers[1].param_groups][0]

    # Called ones after end of epoch with gamma=0.1
    assert pytest.approx(init_lr * 0.1) == adjusted_lr1

    # Called every 3 steps, meaning for 1 epoch of 11 batches, it is called 3 times with gamma=0.1
    assert pytest.approx(init_lr * 0.1) == adjusted_lr2
