"""Preprocessing shared by the Fashion-MNIST classifiers."""

from torchvision import transforms


def classification_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
