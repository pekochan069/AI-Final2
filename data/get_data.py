from os import listdir, makedirs, remove
from os.path import join, exists, join, basename
from shutil import copyfile
import urllib
import tarfile

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data
from datasets import load_dataset, load_from_disk, get_dataset_split_names
from PIL import Image
import os


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)
        ]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class HuggingFaceDataset(data.Dataset):
    def __init__(self, name, split, input_transform=None, target_transform=None):
        super(HuggingFaceDataset, self).__init__()
        self.dataset = load_dataset(name, "bicubic_x2", split=split).with_format(
            "torch"
        )
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        pass


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.Resize(crop_size // upscale_factor),
            transforms.ToTensor(),
        ]
    )


def target_transform(crop_size):
    return transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )


def get_cifar10(batch_size=4, upscale=False):
    normalize = transforms.Normalize(
        mean=[0.49139968, 0.48215841, 0.44653091],
        std=[0.24703223, 0.24348513, 0.26158784],
    )

    train_transform = [transforms.RandomHorizontalFlip()]
    if upscale:
        train_transform.append(transforms.Resize((224, 224)))
    else:
        train_transform.append(transforms.RandomCrop(32, 4))
    train_transform.extend([transforms.ToTensor(), normalize])

    # Prepare train dataset
    train_data = torchvision.datasets.CIFAR10(
        root="data/",
        train=True,
        download=True,
        transform=transforms.Compose(train_transform),
    )
    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Prepare test dataset
    test_transform = []
    if upscale:
        test_transform.append(transforms.Resize((224, 224)))
    test_transform.extend([transforms.ToTensor(), normalize])

    test_data = torchvision.datasets.CIFAR10(
        root="data/",
        train=False,
        download=True,
        transform=transforms.Compose(test_transform),
    )
    test_loader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


def get_cifar10_normalize():
    train_data = torchvision.datasets.CIFAR10(root="data/", train=True, download=True)

    data = train_data.data / 255

    train_mean = data.mean(axis=(0, 1, 2))
    train_std = data.std(axis=(0, 1, 2))

    test_data = torchvision.datasets.CIFAR10(root="data/", train=False, download=True)

    data = test_data.data / 255

    test_mean = data.mean(axis=(0, 1, 2))
    test_std = data.std(axis=(0, 1, 2))

    return train_mean, train_std, test_mean, test_std


def download_bsds300(dest="data"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        if not exists(dest):
            makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, "wb") as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def get_bsds300(upscale_factor: int, batch_size=20):
    root_dir = download_bsds300()
    train_dir = join(root_dir, "train")
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    train_data = DatasetFromFolder(
        train_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_data = DatasetFromFolder(
        test_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )

    test_loader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


def download_bsd100(dest="data"):
    output_image_dir = join(dest, "BSD100_SR/images")

    if not exists(output_image_dir):
        if not exists(dest):
            makedirs(dest)

        file_path = join(dest, "BSD100-images.tgz")

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def get_bsd100(upscale_factor: int, batch_size=20):
    root_dir = download_bsd100()
    test_dir = join(root_dir, "targets")

    crop_size = calculate_valid_crop_size(256, upscale_factor)

    test_data = DatasetFromFolder(
        test_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )

    test_loader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return test_loader


def get_div2k(batch_size=8):
    train_data = load_dataset(
        "eugenesiow/Div2k", "bicubic_x2", split="train"
    ).with_format("torch")

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_data = load_dataset(
        "eugenesiow/Div2k", "bicubic_x2", split="validation"
    ).with_format("torch")

    test_loader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader
