import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


def get_color_extractor(color):
    """Given 'red', 'green', 'blue', or 'all', returns a normalizer
    that zeros out the other two colors.
    Pre-normalized torchvision datas are PILImages of range [0,1].
    Normalized data are tensors of range [-1,1].
    """
    color_center = (0.5, 0.5, 0.5)
    if color == "red":
        color_range = (0.5, float("Inf"), float("Inf"))
    if color == "green":
        color_range = (float("Inf"), 0.5, float("Inf"))
    if color == "blue":
        color_range = (float("Inf"), float("Inf"), 0.5)
    if color == "all":
        color_range = (0.5, 0.5, 0.5)
    return transforms.Normalize(color_center, color_range)


def get_train_and_test_loaders(color):
    """Given 'red', 'green', 'blue', or 'all', return (train_ldr, test_ldr)"""
    if color == "fuser": # quick hack
        color = "all"
    normalizer = transforms.Compose(
        [transforms.ToTensor(),
         get_color_extractor(color)])
    train_data = CIFAR10("./cifar10_data",
                         train=True,
                         download=False,
                         transform=normalizer)
    test_data = CIFAR10("./cifar10_data",
                        train=False,
                        download=False,
                        transform=normalizer)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)
    return train_loader, test_loader


def imshow(img):
    """Plot an image"""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    for color in ["red", "green", "blue", "all"]:
        trainer, tester = get_train_and_test_loaders(color)
        train_iter = iter(trainer)
        images, labels = train_iter.next()
        imshow(torchvision.utils.make_grid(images))


if __name__ == "__main__":
    main()
