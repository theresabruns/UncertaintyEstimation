import numpy as np
from torch.utils.data import Dataset
import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import os

def get_CIFAR10loaders(datapath, batch_size, aug=False):
    if aug:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
            transforms.RandomCrop(32, padding=4),
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
            ])
    test_transform = transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])
    # load the CIFAR10 dataset
    cifar10_train = datasets.CIFAR10(datapath, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(datapath, train=False, download=True, transform=test_transform)
    test_img, _ = cifar10_test[0]
    input_size = test_img.size()[-1]
    num_classes = 10

    # create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, pin_memory=pinmem)
    test_loader = data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, pin_memory=pinmem)

    return train_loader, test_loader, num_classes, input_size

def get_SVHNloaders(datapath, batch_size, aug=False):
    if aug:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                std=[x / 255.0 for x in [50.1, 50.6, 50.8]]),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
            ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                                            std=[x / 255.0 for x in [50.1, 50.6, 50.8]])])
    # load the SVHN dataset
    svhn_train = datasets.SVHN(datapath, split='train', download=True, transform=transform)
    svhn_test = datasets.SVHN(datapath, split='test', download=True, transform=transform)
    svhn_test, _ = data.random_split(svhn_test, [10000, len(svhn_test)-10000])
    test_img, _ = svhn_test[0]
    input_size = test_img.size()[-1]
    num_classes = 10

    # create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True, pin_memory=pinmem)
    test_loader = data.DataLoader(svhn_test, batch_size=batch_size, shuffle=False, pin_memory=pinmem)

    return train_loader, test_loader, num_classes, input_size
    
def get_Elementloaders(datapath, batch_size, aug=False):
    if aug == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.46840793, 0.23778377, 0.19240856), 
                                std=(0.12404595643681854, 0.08136763306617903, 0.07868825907965848)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(64, padding=4)
    ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.46840793, 0.23778377, 0.19240856), std=(0.12404595643681854, 0.08136763306617903, 0.07868825907965848)),
            transforms.Resize(64)
        ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.46840793, 0.23778377, 0.19240856), 
                                                            std=(0.12404595643681854, 0.08136763306617903, 0.07868825907965848)),
                                        transforms.Resize(64)])
    traindata = ImageFolder(datapath+'/train/', transform=transform)
    valdata = ImageFolder(datapath+'/val/', transform=transform)
    testdata = ImageFolder(datapath+'/test/', transform=test_transform)
    ood_data = ImageFolder(datapath+'/ood/', transform=test_transform) #RR: 12496 samples

    # change splits according to implementation in train script (only train/test split)
    combined_trainset = []
    combined_trainset.append(traindata)
    combined_trainset.append(valdata)
    train_dataset = torch.utils.data.ConcatDataset(combined_trainset)

    mini_ood, _ = data.random_split(ood_data, [len(testdata), len(ood_data)-len(testdata)]) #RR: 2265 samples
    test_test, _ = data.random_split(testdata, [900, len(testdata)-900])
    test_ood, _ = data.random_split(mini_ood, [900, len(mini_ood)-900])
    num_classes = len(traindata.class_to_idx)
    sample_img, _ = testdata.__getitem__(0)
    input_size = sample_img.shape[-1]

    # Create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pinmem)
    test_loader = data.DataLoader(testdata, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    ood_loader = data.DataLoader(mini_ood, batch_size=batch_size, shuffle=False, pin_memory=pinmem) #ood_data
    # additional small dataloaders for test purposes
    minitest_loader = data.DataLoader(test_test, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    miniood_loader = data.DataLoader(test_ood, batch_size=batch_size, shuffle=False, pin_memory=pinmem) #ood_data
    
    return train_loader, minitest_loader, miniood_loader, num_classes, input_size

def get_Emblemloaders(datapath, batch_size, aug=False):
    if aug == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3987, 0.4262, 0.4706), std=(0.2505, 0.2414, 0.2466)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(7, sigma=(0.1, 0.5)),
            transforms.ColorJitter(brightness=0.2, hue=0.5),
            transforms.Resize((64,64))
    ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3987, 0.4262, 0.4706), 
                                std=(0.2505, 0.2414, 0.2466)),
            transforms.Resize((64,64))
        ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.3987, 0.4262, 0.4706), 
                                                            std=(0.2505, 0.2414, 0.2466)),
                                        transforms.Resize((64,64))])
    traindata = ImageFolder(datapath+'/train/', transform=transform)
    valdata = ImageFolder(datapath+'/val/', transform=transform)
    testdata = ImageFolder(datapath+'/test/', transform=test_transform)
    ood_data = ImageFolder(datapath+'/ood/', transform=test_transform)
    num_classes = len(traindata.class_to_idx)
    sample_img, _ = testdata.__getitem__(0)
    input_size = sample_img.shape[-1]

    # change splits according to implementation in train script (only train/test split)
    combined_trainset = []
    combined_trainset.append(traindata)
    combined_trainset.append(valdata)
    train_dataset = torch.utils.data.ConcatDataset(combined_trainset)

    # Create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pinmem)
    test_loader = data.DataLoader(testdata, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    ood_loader = data.DataLoader(ood_data, batch_size=batch_size, shuffle=False, pin_memory=pinmem)

    return train_loader, test_loader, ood_loader, num_classes, input_size
