import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from scipy.io import loadmat
from PIL import Image

def get_MNIST(aug=False, root="../../"):
    input_size = 28
    num_classes = 10
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root + "data/MNIST", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root + "data/MNIST", train=False, download=True, transform=transform)
    return input_size, num_classes, train_dataset, test_dataset, None

def get_FashionMNIST(aug=False, root="../../"):
    input_size = 28
    num_classes = 10

    transform_list = [transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))]
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.FashionMNIST(root + "data/FashionMNIST", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root + "data/FashionMNIST", train=False, download=True, transform=transform)
    return input_size, num_classes, train_dataset, test_dataset, None

def get_SVHN(aug=False, root="../../"):
    input_size = 32
    num_classes = 10
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.SVHN(root + "data/svhn-data", split="train", transform=transform, download=True)
    test_dataset = datasets.SVHN(root + "data/svhn-data", split="test", transform=transform, download=True)
    return input_size, num_classes, train_dataset, test_dataset, None

def get_CIFAR10(aug=False, root="../../"):
    input_size = 32
    num_classes = 10
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(root + "data/cifar10", train=True, transform=train_transform, download=True)
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_dataset = datasets.CIFAR10(root + "data/cifar10", train=False, transform=test_transform, download=True)
    return input_size, num_classes, train_dataset, test_dataset, None

def get_RRUseCase(aug=False, root="../../"):
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.46840793, 0.23778377, 0.19240856),
                                std=(0.12404595643681854, 0.08136763306617903, 0.07868825907965848)),
            transforms.Resize(64)])
    if aug:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.46840793, 0.23778377, 0.19240856), 
                                std=(0.12404595643681854, 0.08136763306617903, 0.07868825907965848)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(7, sigma=(0.1, 0.5)),
            transforms.ColorJitter(brightness=0.2, hue=0.5),
            transforms.Resize(64)])
    else:
        transform = test_transform
    datapath = root + "data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean"
    traindata = ImageFolder(datapath+'/train/', transform=transform)
    valdata = ImageFolder(datapath+'/val/', transform=transform)
    test_dataset = ImageFolder(datapath+'/test/', transform=test_transform)
    ood_dataset = ImageFolder(datapath+'/ood/', transform=test_transform)
    mini_ood, _ = torch.utils.data.random_split(ood_dataset, [len(test_dataset), len(ood_dataset)-len(test_dataset)]) #RR: 2265 samples
    
    # change splits according to implementation with "final model" in train script (more train data)
    combined_trainset = []
    combined_trainset.append(traindata)
    combined_trainset.append(valdata)
    train_dataset = torch.utils.data.ConcatDataset(combined_trainset)

    num_classes = len(test_dataset.class_to_idx)
    sample_img, _ = train_dataset.__getitem__(0)
    input_size = sample_img.shape[-1]
    
    return input_size, num_classes, train_dataset, test_dataset, mini_ood

def get_EmblemUseCase(aug=False, root="../../"):
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3987, 0.4262, 0.4706), 
                                std=(0.2505, 0.2414, 0.2466)),
            transforms.Resize((64,64))])
    if aug == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3987, 0.4262, 0.4706), 
                                std=(0.2505, 0.2414, 0.2466)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(7, sigma=(0.1, 0.5)),
            transforms.ColorJitter(brightness=0.2, hue=0.5),
            transforms.Resize((64,64))])
    else:
        transform = test_transform
    datapath = root + "data/EmblemUseCase"
    traindata = ImageFolder(datapath+'/train/', transform=transform)
    valdata = ImageFolder(datapath+'/val/', transform=transform)
    test_dataset = ImageFolder(datapath+'/test/', transform=test_transform)
    ood_dataset = ImageFolder(datapath+'/ood/', transform=test_transform)

    # change splits according to implementation with "final model" in train script (more train data)
    combined_trainset = []
    combined_trainset.append(traindata)
    combined_trainset.append(valdata)
    train_dataset = torch.utils.data.ConcatDataset(combined_trainset)

    num_classes = len(test_dataset.class_to_idx)
    sample_img, _ = train_dataset.__getitem__(0)
    input_size = sample_img.shape[-1]
    
    return input_size, num_classes, train_dataset, test_dataset, ood_dataset


all_datasets = {
    "MNIST": get_MNIST,
    "notMNIST": get_notMNIST,
    "FashionMNIST": get_FashionMNIST,
    "SVHN": get_SVHN,
    "cifar": get_CIFAR10,
    "RR": get_RRUseCase,
    "Emblem": get_EmblemUseCase, 
    "Part": get_RRPartUseCase
}

class NotMNIST(Dataset):
    def __init__(self, root, transform=None):
        root = os.path.expanduser(root)
        self.transform = transform

        data_dict = loadmat(os.path.join(root, "notMNIST_small.mat"))
        self.data = torch.tensor(data_dict["images"].transpose(2, 0, 1), dtype=torch.uint8).unsqueeze(1)
        self.targets = torch.tensor(data_dict["labels"], dtype=torch.int64)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            img = Image.fromarray(img.squeeze().numpy(), mode="L")
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class FastFashionMNIST(datasets.FashionMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(0.2861).div_(0.3530)
        self.data, self.targets = self.data.to("cuda"), self.targets.to("cuda")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target