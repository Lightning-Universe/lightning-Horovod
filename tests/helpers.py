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
import json
import os
import shlex
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

_PATH_TESTS_DIR = os.path.dirname(__file__)
_PATH_DATA_DIR = os.path.join(_PATH_TESTS_DIR, "_data")
os.makedirs(_PATH_DATA_DIR, exist_ok=True)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


class BasicGAN(LightningModule):
    """Implements a basic GAN for the purpose of illustrating multiple optimizers."""

    def __init__(
        self, hidden_dim: int = 128, learning_rate: float = 0.001, b1: float = 0.5, b2: float = 0.999, **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2

        # networks
        mnist_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=self.hidden_dim, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

        self.example_input_array = torch.rand(2, self.hidden_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def _generator_loss(self, imgs):
        # sample noise
        z = torch.randn(imgs.shape[0], self.hidden_dim)
        z = z.type_as(imgs)

        # generate images
        self.generated_imgs = self(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log("g_loss", loss, prog_bar=True, logger=True)
        return loss

    def _discriminator_loss(self, imgs):
        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(fake)

        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)

        # discriminator loss is the average of these
        loss = (real_loss + fake_loss) / 2
        self.log("d_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_idx == 0:
            loss = self._generator_loss(imgs)
        # train discriminator
        elif optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            loss = self._discriminator_loss(imgs)
        else:
            raise ValueError(f"invalid `optimizer_idx={optimizer_idx}` out of set [0, 1]")

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return DataLoader(MNIST(root=_PATH_DATA_DIR, train=True, download=True), batch_size=16)


@torch.no_grad()
def run_model_prediction(trained_model, dataloader, min_acc=0.50):
    orig_device = trained_model.device
    # run prediction on 1 batch
    trained_model.cpu()
    trained_model.eval()

    batch = next(iter(dataloader))
    x, y = batch
    x = x.flatten(1)

    y_hat = trained_model(x)
    acc = accuracy(y_hat.cpu(), y.cpu(), top_k=2).item()

    assert acc >= min_acc, f"This model is expected to get > {min_acc} in test set (it got {acc})"
    trained_model.to(orig_device)


def load_model_from_checkpoint(root_weights_dir, module_class=BoringModel):
    trained_model = module_class.load_from_checkpoint(root_weights_dir)
    assert trained_model is not None, "loading model failed"
    return trained_model


def run_model_test_without_loggers(
    trainer_options: dict, model: LightningModule, data: LightningDataModule = None, min_acc: float = 0.50
):
    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, datamodule=data)

    # correct result and ok accuracy
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    model2 = load_model_from_checkpoint(trainer.checkpoint_callback.best_model_path, type(model))

    # test new model accuracy
    test_loaders = model2.test_dataloader() if not data else data.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    if not isinstance(model2, BoringModel):
        for dataloader in test_loaders:
            run_model_prediction(model2, dataloader, min_acc=min_acc)


# This script will run the actual test model training in parallel
TEST_SCRIPT = os.path.join(os.path.dirname(__file__), "data", "horovod", "train_default_model.py")


def _run_horovod(trainer_options):
    """Execute the training script across multiple workers in parallel."""
    devices = trainer_options.get("devices", 1)
    # TODO: Find out why coverage breaks CI.
    # append = '-a' if '.coverage' in os.listdir(_PROJECT_ROOT) else ''
    # str(num_processes), sys.executable, '-m', 'coverage', 'run', '--source', 'pytorch_lightning', append,
    cmdline = [
        "horovodrun",
        "-np",
        str(devices),
        sys.executable,
        TEST_SCRIPT,
        "--trainer-options",
        shlex.quote(json.dumps(trainer_options)),
    ]
    if trainer_options.get("accelerator", "cpu") == "gpu":
        cmdline += ["--on-gpu"]
    if devices == 2:
        cmdline += ["--check-size"]
    exit_code = subprocess.call(" ".join(cmdline), shell=True, env=os.environ.copy())
    assert exit_code == 0
