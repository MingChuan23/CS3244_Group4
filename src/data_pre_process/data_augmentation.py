import torch
from torchvision.transforms import transforms


def data_augmentation(img, rot = 25, crop = True, hflip = True, vflip = True):
    # input arg:
    # img = torch.randint(0, 256, size=(28, 28), dtype=torch.uint8)
    transform_list = []
    
    if rot != 0 :
        transform_list.append(transforms.RandomRotation(25))  # randomly rotate images in the range (degrees)
    if crop == True:
        transform_list.append(transforms.RandomResizedCrop(224, scale=(0.9, 1.0)))  # simulates width/height shift)
    if hflip == True:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5)) # disable horizontal flip (p=0 means no flip)
    if vflip == True:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5)) # disable horizontal flip (p=0 means no flip)
    transform_list.append(transforms.ToTensor())  # Convert image to tensor
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))  # Normalize with arbitrary mean/std (adjust as needed))

    transform = transforms.Compose(transform_list)
    
    img = transform(img)
    return img