import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Dict
    
class DataModule(pl.LightningDataModule):
    def __init__(self, hparams: Dict, data_dir: str):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_dir = data_dir

        self.transform_clean = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        transform = self.transform_noisy if self.use_noise else self.transform_clean

        train_data = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'train'),
            transform=transform
        )

        total_size = len(train_data)
        indices = np.arange(total_size)
        np.random.seed(self.hparams.seed)
        np.random.shuffle(indices)

        label_count = int(total_size * self.hparams.label_pct)
        labeled_indices = indices[:label_count]
        unlabeled_indices = indices[label_count:]

        self.labeled_ds = Subset(train_data, labeled_indices)
        self.unlabeled_ds = Subset(train_data, unlabeled_indices)

        self.val_ds = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'valid'),
            transform=self.transform_clean
        )
        self.test_ds = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'test'),
            transform=self.transform_clean
        )

    def labeled_dataloader(self):
        return DataLoader(
            self.labeled_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def unlabeled_dataloader(self):
        return DataLoader(
            self.unlabeled_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.unlabeled_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True
        )