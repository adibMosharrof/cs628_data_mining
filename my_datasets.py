from typing import List

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    def __init__(self, labels, data, transform) -> None:
        super().__init__()
        self.transform = transform

        self.labels = labels
        self.data = data
        self.labels_set = set(labels.numpy())
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }

        self.random_state = np.random.RandomState(69)

    def read_images(self, imgs) -> List[any]:
        out = []
        for i in imgs:
            img = Image.fromarray(i.numpy(), mode="L")
            if self.transform:
                img = self.transform(img)
            out.append(img)
        return out

    def __len__(self):
        return len(self.data)


class SiameseTrainDataset(BaseDataset):
    def __init__(self, mnist_dataset) -> None:
        super().__init__(
            mnist_dataset.train_labels,
            mnist_dataset.train_data,
            mnist_dataset.transform,
        )

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.data[index], self.labels[index].item()
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.data[siamese_index]
        img1, img2 = self.read_images([img1, img2])

        # return img1, img2, target
        return (img1, img2), target


class SiameseValidationDataset(BaseDataset):
    def __init__(self, mnist_dataset) -> None:
        # super().__init__()
        super().__init__(
            mnist_dataset.test_labels,
            mnist_dataset.test_data,
            mnist_dataset.transform,
        )
        positive_pairs = [
            [
                i,
                self.random_state.choice(self.label_to_indices[self.labels[i].item()]),
                1,
            ]
            for i in range(0, len(self.data), 2)
        ]
        negative_pairs = [
            [
                i,
                self.random_state.choice(
                    self.label_to_indices[
                        np.random.choice(
                            list(self.labels_set - set([self.labels[i].item()]))
                        )
                    ]
                ),
                0,
            ]
            for i in range(1, len(self.data), 2)
        ]
        self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        img1 = self.data[self.test_pairs[index][0]]
        img2 = self.data[self.test_pairs[index][1]]

        img1, img2 = self.read_images([img1, img2])
        # return (img1, img2), self.test_pairs[index][2]
        return (img1, img2), self.test_pairs[index][2]


class TripletTrainDataset(BaseDataset):
    def __init__(self, mnist_dataset) -> None:
        super().__init__(
            mnist_dataset.train_labels,
            mnist_dataset.train_data,
            mnist_dataset.transform,
        )

    def __getitem__(self, index: int):
        img1, label1 = self.data[index], self.labels[index].item()
        pos_index = index
        while pos_index == index:
            pos_index = np.random.choice(self.label_to_indices[label1])
        neg_label = np.random.choice(list(self.labels_set - set([label1])))
        neg_index = np.random.choice(self.label_to_indices[neg_label])
        img2 = self.data[pos_index]
        img3 = self.data[neg_index]
        img1, img2, img3 = self.read_images([img1, img2, img3])
        return (img1, img2, img3), []


class TripletValidationDataset(BaseDataset):
    def __init__(self, mnist_dataset) -> None:
        super().__init__(
            mnist_dataset.train_labels,
            mnist_dataset.train_data,
            mnist_dataset.transform,
        )
        self.triplets = [
            [
                i,
                self.random_state.choice(self.label_to_indices[self.labels[i].item()]),
                self.random_state.choice(
                    self.label_to_indices[
                        np.random.choice(
                            list(self.labels_set - set([self.labels[i].item()]))
                        )
                    ]
                ),
            ]
            for i in range(len(self.data))
        ]

    def __getitem__(self, index: int):
        img1, img2, img3 = self.read_images(
            [
                self.data[self.triplets[index][0]],
                self.data[self.triplets[index][1]],
                self.data[self.triplets[index][2]],
            ]
        )
        return (img1, img2, img3), []
