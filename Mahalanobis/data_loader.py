# original code is from https://github.com/aaron-xichen/pytorch-playground
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        testset = datasets.SVHN(root=data_root, split='test', download=True,transform=TF)
        sample_img, _ = testset.__getitem__(0)
        input_size = sample_img.shape[-1]
        subset, _ = data.random_split(testset, [10000, len(testset)-10000])
        test_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds, input_size

def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=TF)
    valset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=TF)
    sample_img, _ = trainset.__getitem__(0)
    input_size = sample_img.shape[-1]
    if train:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds, input_size

def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    trainset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=TF)
    valset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=TF)
    sample_img, _ = trainset.__getitem__(0)
    input_size = sample_img.shape[-1]
    if train:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds, input_size

def get_Elementloaders(data_root, batch_size, aug=False):
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
    traindata = ImageFolder(data_root+'/train/', transform=transform)
    valdata = ImageFolder(data_root+'/val/', transform=test_transform)
    testdata = ImageFolder(data_root+'/test/', transform=test_transform)
    ood_data = ImageFolder(data_root+'/ood/', transform=test_transform)
    mini_ood, _ = data.random_split(ood_data, [len(testdata), len(ood_data)-len(testdata)]) #RR: 2265 samples
    num_classes = len(traindata.class_to_idx)
    sample_img, _ = testdata.__getitem__(0)
    input_size = sample_img.shape[-1]

    # Create dataloaders
    train_loader = data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(valdata, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
    ood_loader = data.DataLoader(mini_ood, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, ood_loader, num_classes, input_size

def get_Emblemloaders(data_root, batch_size, aug=False):
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
    traindata = ImageFolder(data_root+'/train/', transform=transform)
    valdata = ImageFolder(data_root+'/val/', transform=test_transform)
    testdata = ImageFolder(data_root+'/test/', transform=test_transform)
    ood_data = ImageFolder(data_root+'/ood/', transform=test_transform)
    num_classes = len(traindata.class_to_idx)
    sample_img, _ = testdata.__getitem__(0)
    input_size = sample_img.shape[-1]

    # Create dataloaders
    train_loader = data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(valdata, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
    ood_loader = data.DataLoader(ood_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, ood_loader, num_classes, input_size

def getTargetDataSet(data_type, batch_size, input_TF, dataroot, aug):
    if data_type == 'cifar10':
        train_loader, test_loader, input_size = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader, input_size = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader, input_size = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'RRusecase':
        train_loader, _, test_loader, _, _, input_size = get_Elementloaders(data_root=dataroot, batch_size=batch_size, aug=aug)
    elif data_type == 'Emblemusecase':
        train_loader, _, test_loader, _, _, input_size = get_Emblemloaders(data_root=dataroot, batch_size=batch_size, aug=aug)
    return train_loader, test_loader, input_size

def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader, _ = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader, _ = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader, _ = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'RRusecase':
        _, _, _, test_loader, _, _ = get_Elementloaders(data_root=dataroot, batch_size=batch_size)
    elif data_type == 'Emblemusecase':
        _, _, _, test_loader, _, _ = get_Emblemloaders(data_root=dataroot, batch_size=batch_size)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader


