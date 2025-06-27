import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Dict
from PIL import Image
from scripts.salt_pepper import salt_and_pepper_noise


class AddSaltPepper:
    def __init__(self, amount=0.05):
        self.amount = amount

    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        noisy = salt_and_pepper_noise(tensor, amount=self.amount)
        return noisy


class NoisyPairDataset:
    def __init__(self, dataset, clean_transform, noisy_transform):
        self.dataset = dataset
        self.clean_transform = clean_transform
        self.noisy_transform = noisy_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, _ = self.dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        clean_img = self.clean_transform(image)
        noisy_img = self.noisy_transform(image)
        return noisy_img, clean_img


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams: Dict, data_dir: str, noise_amount=0.05, image_size=(128, 128)):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_dir = data_dir
        self.noise_amount = noise_amount
        self.image_size = image_size

        self.transform_clean = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_noisy = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            AddSaltPepper(amount=self.noise_amount),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        base_train = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'train'),
        )
        self.train_ds = NoisyPairDataset(
            dataset=base_train,
            clean_transform=self.transform_clean,
            noisy_transform=self.transform_noisy
        )

        base_val = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'valid'),
        )
        self.val_ds = NoisyPairDataset(
            dataset=base_val,
            clean_transform=self.transform_clean,
            noisy_transform=self.transform_noisy
        )

        base_test = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'test'),
        )
        self.test_ds = NoisyPairDataset(
            dataset=base_test,
            clean_transform=self.transform_clean,
            noisy_transform=self.transform_noisy
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True
        )
