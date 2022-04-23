import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as metrics

from my_losses import ContrastiveLoss


class EmbeddingNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class BaseModel(pl.LightningModule):
    def __init__(self, embedding_net: EmbeddingNet):
        super().__init__()
        self.embedding_net = embedding_net
        self.train_metrics = metrics.Accuracy()
        self.threshold = 0.5
        self.val_metrics = metrics.Accuracy()

    def validation_step(self, batch, batch_idx=None):
        imgs, labels = batch
        imgs_emb = self(imgs)

        similarities = F.cosine_similarity(imgs_emb[0], imgs_emb[1])
        preds = (similarities > self.threshold).float()
        acc = self.val_metrics(preds, labels).detach()
        self.log("acc", acc, prog_bar=True)


class ClassificationNet(pl.LightningModule):
    def __init__(self, embedding_net: EmbeddingNet, n_classes=10):
        super().__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = metrics.Accuracy()
        self.val_metrics = metrics.Accuracy()
        self.test_metrics = metrics.Accuracy()

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="train", metric=self.train_metrics)

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="val", metric=self.val_metrics)

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="test", metric=self.test_metrics)

    def _shared_step(self, batch, batch_idx=None, step="train", metric=None):
        data, labels = batch
        logits = self(data)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = metric(preds, labels).detach()
        self.log(f"{step}/acc", accuracy, prog_bar=True)
        self.log(f"{step}/loss", loss, prog_bar=True)
        return loss


class MetricNet(BaseModel):
    def __init__(
        self,
        embedding_net: EmbeddingNet,
        loss_func: nn.Module = None,
    ):
        super().__init__(embedding_net)
        self.save_hyperparameters(ignore=["embedding_net"])
        self.criterion = loss_func

    def forward(self, imgs):
        return [self.embedding_net(img) for img in imgs]

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="train", metric=self.train_metrics)

    # def validation_step(self, batch, batch_idx=None):
    #     return self._shared_step(batch, step="val", metric=self.val_metrics)

    def _shared_step(self, batch, batch_idx=None, step="train", metric=None):
        imgs, labels = batch
        imgs_emb = self(imgs)
        loss = self.criterion(imgs_emb, labels)

        # similarities = F.cosine_similarity(**imgs_emb)
        # preds = (similarities > self.threshold).float()
        # accuracy = metric(preds, labels).detach()
        # self.log(f"{step}/acc", accuracy)
        self.log(f"{step}/loss", loss)
        return loss

    def get_embedding(self, x):
        return self.embedding_net(x)
