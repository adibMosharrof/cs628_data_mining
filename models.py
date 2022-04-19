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


class SiameseNet(pl.LightningModule):
    def __init__(
        self, embedding_net: EmbeddingNet, threshold: float = 0.5, margin: int = 1
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedding_net"])
        self.embedding_net = embedding_net
        self.threshold = threshold
        self.criterion = ContrastiveLoss(margin)
        self.train_metrics = metrics.Accuracy()
        self.val_metrics = metrics.Accuracy()
        self.test_metrics = metrics.Accuracy()

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="train", metric=self.train_metrics)

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="val", metric=self.val_metrics)

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="test", metric=self.test_metrics)

    def _shared_step(self, batch, batch_idx=None, step="train", metric=None):
        img1, img2, labels = batch
        img1_emb, img2_emb = self(img1, img2)
        loss = self.criterion(img1_emb, img2_emb, labels)

        similarities = F.cosine_similarity(img1_emb, img2_emb)
        preds = (similarities > self.threshold).float()
        accuracy = metric(preds, labels).detach()
        self.log(f"{step}/acc", accuracy)
        self.log(f"{step}/loss", loss)
        return loss

    def get_embedding(self, x):
        return self.embedding_net(x)
