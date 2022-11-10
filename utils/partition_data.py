import os
import argparse
import numpy as np

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms

from PIL import Image


def generate_partitions(n_partitions, dataset, malicious=False):
    torch.manual_seed(0)
    os.makedirs('/tmp/app/data/train', exist_ok=True)
    os.makedirs('/tmp/app/data/test', exist_ok=True)

    if dataset == "cifar10":
        dataset_inst = CIFAR10
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    if dataset == "mnist":
        dataset_inst = MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )

    # Load train & test sets
    trainset = dataset_inst(
        root="/tmp/app/data", train=True, download=True
    )

    testset = dataset_inst(
        root="/tmp/app/data", train=False, download=False
    )

    # Define splits
    n_train_samples = len(trainset)
    n_test_samples = len(testset)
    train_samples_per_user = n_train_samples/n_partitions
    test_sampels_per_user = n_test_samples/n_partitions

    train_distribution = [int(train_samples_per_user) for _ in range(n_partitions)]
    test_distribution = [int(test_sampels_per_user) for _ in range(n_partitions)]

    train_remaining_samples = n_train_samples - np.sum(train_distribution)
    if(train_remaining_samples>0):
        for i in range(train_remaining_samples):
            train_distribution[i] += 1
    test_remaining_samples = n_test_samples - np.sum(test_distribution)
    if(test_remaining_samples>0):
        for i in range(test_remaining_samples):
            test_distribution[i] += 1

    # Split dataset
    print(f"trainset: {len(trainset)} samples")
    print(f"testset: {len(testset)} samples")
    train_splits = random_split(trainset, train_distribution)
    test_splits = random_split(testset, test_distribution)

    # Generate subset for each user
    train_data_subsets = []
    train_target_subsets = []
    test_data_subsets = []
    test_target_subsets = []
    for i, train_split in enumerate(train_splits):
        train_split_indices = train_split.indices

        train_data = np.array(trainset.data)[train_split_indices]
        train_target = np.array(trainset.targets)[train_split_indices]
        train_data_subsets.append(train_data)
        train_target_subsets.append(train_target)

    for i, test_split in enumerate(test_splits):
        test_split_indices = test_split.indices

        test_data = np.array(testset.data)[test_split_indices]
        test_target = np.array(testset.targets)[test_split_indices]
        test_data_subsets.append(test_data)
        test_target_subsets.append(test_target)

    # Permute labels of the subset to simulate malicious updates
    if malicious:
        train_target_subsets[3] = np.random.permutation(train_target_subsets[3])


    # Save subsets
    for i in range(n_partitions):
        print(f"subset {i}: {len(train_data_subsets[i])} train data, {len(test_data_subsets[i])} test data")
        train_subset = Partition(
            torch.from_numpy(train_data_subsets[i]),
            train_target_subsets[i],
            transform=transform
        )
        test_subset = Partition(
            torch.from_numpy(test_data_subsets[i]),
            test_target_subsets[i],
            transform=transform
        )

        torch.save(train_subset, f"/tmp/app/data/train/train_subset-{i}.pth")
        torch.save(test_subset, f"/tmp/app/data/test/test_subset-{i}.pth")


class Partition(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.array(img))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_partitions", type=int, required=True, help="number of partitions"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset"
    )
    parser.add_argument(
        "--malicious", action='store_true'
    )
    args = parser.parse_args()

    generate_partitions(
        dataset=args.dataset,
        n_partitions=args.n_partitions,
        malicious=args.malicious
    )
