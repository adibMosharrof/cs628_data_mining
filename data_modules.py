import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from my_datasets import SiameseTrainDataset, SiameseValidationDataset


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=8) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trainset: Dataset
        self.validationset: Dataset
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    def setup(self, stage=None):

        self.trainset = datasets.FashionMNIST(
            "./data", download=True, train=True, transform=self.transform
        )
        self.validationset = datasets.FashionMNIST(
            "./data", download=True, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validationset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_base_train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_base_val_dataloader(self):
        return DataLoader(
            self.validationset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SiameseDataModule(BaseDataModule):
    def __init__(self, batch_size, num_workers=8) -> None:
        super().__init__(batch_size)
        self.siamese_trainset: Dataset
        self.siamese_validationset: Dataset

    def setup(self, stage=None):
        super().setup()
        # trainset = datasets.FashionMNIST(
        #     "./data", download=True, train=True, transform=self.transform
        # )
        # validationset = datasets.FashionMNIST(
        #     "./data", download=True, train=False, transform=self.transform
        # )
        self.siamese_trainset = SiameseTrainDataset(self.trainset)
        self.siamese_validationset = SiameseValidationDataset(self.validationset)

    def train_dataloader(self):
        return DataLoader(
            self.siamese_trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.siamese_validationset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
