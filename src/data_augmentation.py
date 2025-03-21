import torch
from torchvision.transforms import transforms


def data_augmentation(rot=25, crop=True, hflip=True, vflip=True):
    transform_list = []

    if rot:
        transform_list.append(transforms.RandomRotation(rot))
    if crop:
        transform_list.append(transforms.RandomResizedCrop(28, scale=(0.9, 1.0)))
    if hflip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if vflip:
        transform_list.append(transforms.RandomVerticalFlip())

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_list)