import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class SiameseTrainDataset(Dataset):
    def __init__(self, mnist_dataset) -> None:
        super().__init__()
        self.mnist_dataset = mnist_dataset
        self.transform = mnist_dataset.transform

        self.train_labels = mnist_dataset.train_labels
        self.train_data = mnist_dataset.train_data
        self.labels_set = set(mnist_dataset.train_labels.numpy())
        self.label_to_indices = {
            label: np.where(self.train_labels.numpy() == label)[0]
            for label in self.labels_set
        }

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.train_data[index], self.train_labels[index].item()
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.train_data[siamese_index]
        img1 = Image.fromarray(img1.numpy(), mode="L")
        img2 = Image.fromarray(img2.numpy(), mode="L")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, target

    def __len__(self):
        return len(self.train_data)


class SiameseValidationDataset(Dataset):
    def __init__(self, mnist_dataset) -> None:
        super().__init__()
        self.mnist_dataset = mnist_dataset

        self.transform = mnist_dataset.transform

        self.test_labels = mnist_dataset.test_labels
        self.test_data = mnist_dataset.test_data
        self.labels_set = set(mnist_dataset.test_labels.numpy())
        self.label_to_indices = {
            label: np.where(self.test_labels.numpy() == label)[0]
            for label in self.labels_set
        }
        random_state = np.random.RandomState(69)
        positive_pairs = [
            [
                i,
                random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                1,
            ]
            for i in range(0, len(self.test_data), 2)
        ]
        negative_pairs = [
            [
                i,
                random_state.choice(
                    self.label_to_indices[
                        np.random.choice(
                            list(self.labels_set - set([self.test_labels[i].item()]))
                        )
                    ]
                ),
                0,
            ]
            for i in range(1, len(self.test_data), 2)
        ]
        self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        img1 = self.test_data[self.test_pairs[index][0]]
        img2 = self.test_data[self.test_pairs[index][1]]
        img1 = Image.fromarray(img1.numpy(), mode="L")
        img2 = Image.fromarray(img2.numpy(), mode="L")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, self.test_pairs[index][2]

    def __len__(self):
        return len(self.test_data)
