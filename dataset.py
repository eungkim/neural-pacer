import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random


def build_dataset(name_dataset, num_meta, corruption_prob, corruption_type, batch_size, seed):
    normalize = transforms.Normalize(
        mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
        std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if name_dataset=='cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True
        )
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True, seed=seed
        )
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)
    
    elif name_dataset=='cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True
        )
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=num_meta, corruption_prob=corruption_prob,
            corruption_type=corruption_type, transform=train_transform, download=True, seed=seed
        )
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(train_data_meta, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, train_meta_loader, test_loader

