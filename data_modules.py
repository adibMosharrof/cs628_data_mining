import importlib

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from my_datasets import (
    SiameseTrainDataset,
    SiameseValidationDataset,
    TripletTrainDataset,
    TripletValidationDataset,
)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset_name, num_workers=8) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trainset: Dataset
        self.validationset: Dataset
        self.transform = T.Compose(
            [
                T.Grayscale(),
                T.Resize([28, 28]),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        )
        self.dataset_class = getattr(
            importlib.import_module("torchvision.datasets"), dataset_name
        )

    def setup(self, stage=None):
        # self.trainset = datasets.FashionMNIST(
        self.trainset = self.dataset_class(
            "./data", download=True, train=True, transform=self.transform
        )
        # self.validationset = datasets.FashionMNIST(
        self.validationset = self.dataset_class(
            "./data", download=True, train=False, transform=self.transform
        )
        a = 1

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.validationset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #     )

    def val_dataloader(self):
        return DataLoader(
            SiameseValidationDataset(self.validationset),
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


class MetricDataModule(BaseDataModule):
    def __init__(
        self, batch_size, dataset_name, num_workers=8, metric_name: str = "siamese"
    ) -> None:
        super().__init__(batch_size, dataset_name)
        self.metric_trainset: Dataset
        self.metric_validationset: Dataset
        self.metric_name = metric_name

    def setup(self, stage=None):
        super().setup()
        if self.metric_name == "siamese":
            self.metric_trainset = SiameseTrainDataset(self.trainset)
            # self.metric_validationset = SiameseValidationDataset(self.validationset)
        elif self.metric_name == "triplet":
            self.metric_trainset = TripletTrainDataset(self.trainset)
            # self.metric_validationset = TripletValidationDataset(self.validationset)

    def train_dataloader(self):
        return DataLoader(
            self.metric_trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.metric_validationset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #     )
