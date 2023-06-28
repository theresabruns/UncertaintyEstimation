import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import os

def get_MNISTloaders(datapath, batch_size, aug=False):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if aug:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((28,28), scale=(0.8, 1.0))
            ])
    else:
        transform = test_transform
    # Load the MNIST dataset
    mnist_train = datasets.MNIST(datapath, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(datapath, train=False, download=True, transform=test_transform)
    miniset, _ = data.random_split(mnist_test, [1000, 9000])
    test_img, _ = mnist_test[0]
    input_size = test_img.size()[-1]
    num_classes = 10

    # Split the training data into a training set and a validation set
    train_size = int(0.9 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = data.random_split(mnist_train, [train_size, val_size])
    minitrain_size = int(0.02 * len(mnist_train))
    minitrainset, _ = data.random_split(mnist_train, [minitrain_size, len(mnist_train) - minitrain_size])

    # Create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=pinmen)
    val_loader = data.DataLoader(mnist_val, batch_size=batch_size, shuffle=False, pin_memory=pinmen)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=pinmen)
    #additional small dataloaders for test purposes
    miniloader = data.DataLoader(miniset, batch_size=batch_size, shuffle=False, pin_memory=pinmen)
    minitrain_loader = data.DataLoader(minitrainset, batch_size=batch_size, shuffle=False, pin_memory=pinmen)
    return train_loader, val_loader, test_loader, num_classes, input_size

def get_FashionMNISTloaders(datapath, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    mnist_test = datasets.FashionMNIST(datapath, train=False, download=True, transform=transform)
    miniset, _ = data.random_split(mnist_test, [1000, 9000])
    test_img, _ = mnist_test[0]
    input_size = test_img.size()[-1]

    # Create dataloader
    ood_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False) #for 10.000 images
    miniloader = data.DataLoader(miniset, batch_size=batch_size, shuffle=False) #for 1.000 images
    return miniloader, input_size

def get_NotMNISTloaders(datapath, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ExtractChannel(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    num_classes = 10
    #using this requires a previous manual download of the notMNIST dataset!
    notmnist = datasets.ImageFolder(root=os.path.join(datapath, 'notMNIST_small'), transform=transform)
    notmnist_test, _ = data.random_split(notmnist, [10000, len(notmnist)-10000])
    test_img, _ = notmnist_test[0]
    input_size = test_img.size()[-1]
    print("Size test: ",test_img.size())

    # Create dataloader
    ood_loader = data.DataLoader(notmnist_test, batch_size=batch_size, shuffle=False)
    return ood_loader, num_classes, input_size

def get_CIFAR10loaders(datapath, batch_size, rgb_flag=None):
    if rgb_flag: #in case this dataset is used as OOD with SVHN as ID set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), 
                                (0.19803012, 0.20101562, 0.19703614)) #svhn norm. values
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.CenterCrop(28)
            ])
    # Load the CIFAR10 dataset
    cifar10_train = datasets.CIFAR10(datapath, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(datapath, train=False, download=True, transform=transform)
    if not rgb_flag: 
        cifar10_test, _ = data.random_split(cifar10_test, [1000, 9000])
    test_img, _ = cifar10_test[0]
    input_size = test_img.size()[-1]
    num_classes = 10

    # Split the training data into a training set and a validation set
    train_size = int(0.9 * len(cifar10_train))
    val_size = len(cifar10_train) - train_size
    cifar10_train, cifar10_val = data.random_split(cifar10_train, [train_size, val_size])

    # Create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, pin_memory=pinmen)
    val_loader = data.DataLoader(cifar10_val, batch_size=batch_size, shuffle=False, pin_memory=pinmen)
    ood_loader = data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, pin_memory=pinmen)

    return train_loader, val_loader, ood_loader, num_classes, input_size

def get_SVHNloaders(datapath, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), 
                            (0.19803012, 0.20101562, 0.19703614)), #svhn norm. values
        ])
    # Load the SVHN dataset
    svhn_train = datasets.SVHN(datapath, split='train', download=True, transform=transform)
    svhn_test = datasets.SVHN(datapath, split='test', download=True, transform=transform)
    svhn_test, _ = data.random_split(svhn_test, [10000, len(svhn_test)-10000])
    test_img, _ = svhn_test[0]
    input_size = test_img.size()[-1]
    num_classes = 10

    # Split the training data into a training set and a validation set
    train_size = int(0.9 * len(svhn_train))
    val_size = len(svhn_train) - train_size
    svhn_train, svhn_val = data.random_split(svhn_train, [train_size, val_size])

    # Create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True, pin_memory=pinmem)
    val_loader = data.DataLoader(svhn_val, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    test_loader = data.DataLoader(svhn_test, batch_size=batch_size, shuffle=False, pin_memory=pinmem)

    return train_loader, val_loader, test_loader, num_classes, input_size

def get_Elementloaders(datapath, batch_size, aug=False):
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
    traindata = ImageFolder(datapath+'/train/', transform=transform)
    valdata = ImageFolder(datapath+'/val/', transform=test_transform)
    testdata = ImageFolder(datapath+'/test/', transform=test_transform)
    ood_data = ImageFolder(datapath+'/ood/', transform=test_transform)
    mini_ood, _ = data.random_split(ood_data, [len(testdata), len(ood_data)-len(testdata)]) #RR: 2265 samples
    test_test, _ = data.random_split(testdata, [200, len(testdata)-200])
    test_ood, _ = data.random_split(mini_ood, [200, len(mini_ood)-200])
    num_classes = len(traindata.class_to_idx)
    sample_img, _ = testdata.__getitem__(0)
    input_size = sample_img.shape[-1]

    # Create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=pinmem)
    val_loader = data.DataLoader(valdata, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    test_loader = data.DataLoader(testdata, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    ood_loader = data.DataLoader(mini_ood, batch_size=batch_size, shuffle=False, pin_memory=pinmem) #ood_data
    #additional small dataloaders for test purposes
    minitest_loader = data.DataLoader(test_test, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    miniood_loader = data.DataLoader(test_ood, batch_size=batch_size, shuffle=False, pin_memory=pinmem) #ood_data

    return train_loader, val_loader, test_loader, ood_loader, num_classes, input_size

def get_Emblemloaders(datapath, batch_size, aug=False):
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
    traindata = ImageFolder(datapath+'/train/', transform=transform)
    valdata = ImageFolder(datapath+'/val/', transform=test_transform)
    testdata = ImageFolder(datapath+'/test/', transform=test_transform)
    ood_data = ImageFolder(datapath+'/ood/', transform=test_transform)
    num_classes = len(traindata.class_to_idx)
    sample_img, _ = testdata.__getitem__(0)
    input_size = sample_img.shape[-1]

    # Create dataloaders
    if torch.cuda.is_available():
        pinmen = True
    else:
        pinmem = False
    train_loader = data.DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=pinmem)
    val_loader = data.DataLoader(valdata, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    test_loader = data.DataLoader(testdata, batch_size=batch_size, shuffle=False, pin_memory=pinmem)
    ood_loader = data.DataLoader(ood_data, batch_size=batch_size, shuffle=False, pin_memory=pinmem)

    return train_loader, val_loader, test_loader, ood_loader, num_classes, input_size

class ExtractChannel(object):
    def __call__(self, image):
        image = image[0,:,:].unsqueeze(0)    
        return image