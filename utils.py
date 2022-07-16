import numpy as np
import torch

from torch.utils.data import Dataset


class RotMNISTDataset(Dataset):

    def __init__(self, dataset, transform=None):
        super().__init__()
        self.transform = transform
        mnist_dir = './data/ROT-MNIST'
        if dataset == 'train':
            data = np.load(mnist_dir + '/rotated_train.npz')
        elif dataset == 'validation':
            data = np.load(mnist_dir + '/rotated_valid.npz')
        elif dataset == 'test':
            data = np.load(mnist_dir + '/rotated_test.npz')
        else:
            data = None

        self.image_set = data['x'].reshape((data['x'].shape[0], 28, 28))
        self.label_set = data['y']

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        image_sample = torch.from_numpy(self.image_set[idx])
        label_sample = torch.from_numpy(np.asarray(self.label_set[idx]))

        if self.transform:
            image_sample = self.transform(image_sample)
        return image_sample[None, :, :], label_sample
