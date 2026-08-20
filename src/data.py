"""CIFAR-10 loaders. Augmentation is a flag so we can ablate it."""
import torch
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def build_transforms(augment: bool, img_size: int = 32):
    base = [transforms.Resize(img_size)] if img_size != 32 else []
    train = base + ([transforms.RandomCrop(img_size, padding=img_size // 8),
                     transforms.RandomHorizontalFlip()] if augment else [])
    train += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    test = base + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(train), transforms.Compose(test)


def get_loaders(batch_size=128, augment=True, img_size=32, root="./data", workers=2):
    tf_train, tf_test = build_transforms(augment, img_size)
    train = datasets.CIFAR10(root, train=True, download=True, transform=tf_train)
    test = datasets.CIFAR10(root, train=False, download=True, transform=tf_test)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                    num_workers=workers, pin_memory=True),
        torch.utils.data.DataLoader(test, batch_size=256, shuffle=False,
                                    num_workers=workers, pin_memory=True),
    )
