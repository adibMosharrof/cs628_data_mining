import os

import matplotlib.pyplot as plt
import numpy as np
import torch

cuda = torch.cuda.is_available()

fashion_mnist_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
mnist_classes = fashion_mnist_classes


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        if cuda:
            model = model.cuda()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k : k + len(images)] = (
                model.get_embedding(images).data.cpu().numpy()
            )
            labels[k : k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


def plot_embeddings(
    embeddings, targets, xlim=None, ylim=None, out_dir="", plot_file_name="val"
):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(
            embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i]
        )
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.savefig(os.path.join(out_dir, f"{plot_file_name}.jpg"), bbox_inches="tight")
