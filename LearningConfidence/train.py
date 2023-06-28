import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

import seaborn as sns  # import this after torch or it will break everything

from models.vgg import VGG16
from models.linear import Linear_2L
from models.densenet import DenseNet121
from models.wideresnet import WideResNet
from models.resnet import ResNet50, ResNet34
from utils.utils import encode_onehot, CSVLogger, plot_histograms
from utils.datasets import *

dataset_options = ['cifar10', 'svhn', 'RR', 'Emblem']
model_options = ['wideresnet', 'densenet', 'vgg16', 'resnet50', 'resnet34', 'linear']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', default='cifar10', choices=dataset_options)
parser.add_argument('--model', default='linear', choices=model_options)
parser.add_argument('--batch_size', type=int, default=64) 
parser.add_argument('--epochs', type=int, default=50) 
parser.add_argument('--seed', type=int, default=0, help='model/run identifier')
parser.add_argument('--learning_rate', type=float, default=1e-4) 
parser.add_argument('--budget', type=float, default=0.3, metavar='N',
                    help='the budget for how often the network can get hints')
parser.add_argument('--aug', action='store_true', default=False, help='augment data')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, choice: 0/1, default: 0')
parser.add_argument('--outdir', type=str, default='RRElement_linear')
parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='\'../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\' or \'../../data/EmblemUseCase\'')
parser.add_argument('--baseline', action='store_true', default=False, help='train model without confidence branch')
args = parser.parse_args()
print(args)

cudnn.benchmark = True  # should make training go faster for large models
if args.baseline:
    args.budget = 0.
if args.dataset == 'svhn' and args.model == 'wideresnet':
    args.model = 'wideresnet16_8'
np.random.seed(0)

# set device
if torch.cuda.is_available():
    use_cuda = True
    device = torch.device("cuda:"+str(args.gpu))
else:
    use_cuda = False
    device = torch.device("cpu")
print("Device used: ", device)

filename = args.dataset + '_' + args.model + '_budget_' + str(args.budget) + '_seed_' + str(args.seed)
modeldir = 'checkpoints/' + args.outdir
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
logdir = 'logs/' + args.outdir
if not os.path.exists(logdir):
    os.makedirs(logdir)

# ------------------------------------------------------------------------------------------------------
# dataset
datapath = '../../data'
if args.dataset == 'cifar10':
    train_loader, test_loader, num_classes, img_size = get_CIFAR10loaders(datapath, batch_size=args.batch_size, aug=args.aug)
    data = args.dataset
elif args.dataset == 'svhn':
    train_loader, test_loader, num_classes, img_size = get_SVHNloaders(datapath, batch_size=args.batch_size, aug=args.aug)
    data = args.dataset
elif args.dataset == 'RR':
    datapath = args.uc_datapath    
    train_loader, test_loader, _, num_classes, img_size = get_Elementloaders(datapath, batch_size=args.batch_size, aug=args.aug)
    data = 'RRusecase'
elif args.dataset == 'Emblem':
    datapath = args.uc_datapath
    train_loader, test_loader, _, num_classes, img_size = get_Emblemloaders(datapath, batch_size=args.batch_size, aug=args.aug)
    data = 'Emblemusecase'

# ------------------------------------------------------------------------------------------------------
# testing helper function
def test(loader, device):
    cnn.eval()    # change model to 'eval' mode (BN uses moving mean/var).

    correct = []
    probability = []
    confidence = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        pred, conf = cnn(images)
        pred = F.softmax(pred, dim=-1)
        sig = nn.Sigmoid()
        conf = sig(conf).data.view(-1)

        pred_value, pred = torch.max(pred.data, 1)
        correct.extend((pred == labels).cpu().numpy())
        probability.extend(pred_value.cpu().numpy())
        confidence.extend(conf.cpu().numpy())

    correct = np.array(correct).astype(bool)
    probability = np.array(probability)
    confidence = np.array(confidence)

    if args.baseline:
        plot_histograms(id_confidences=probability, data=data, model=args.model, seed=args.seed, corrects=correct)
    else:
        plot_histograms(id_confidences=confidence, data=data, model=args.model, seed=args.seed, corrects=correct)

    val_acc = np.mean(correct)
    conf_min = np.min(confidence)
    conf_max = np.max(confidence)
    conf_avg = np.mean(confidence)

    cnn.train()
    return val_acc, conf_min, conf_max, conf_avg

# ------------------------------------------------------------------------------------------------------
# initialize model
if args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10).to(device)
elif args.model == 'wideresnet16_8':
    cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8).to(device)
elif args.model == 'linear':
    cnn = Linear_2L(in_channels=3, out_channels=num_classes, input_size=img_size, n_hid=200).to(device)
elif args.model == 'densenet':
    cnn = DenseNet121(num_classes=num_classes, input_size=img_size).to(device)
elif args.model == 'vgg16':
    cnn = VGG16(in_channels=3, out_channels=num_classes, input_size=img_size).to(device)
elif args.model == 'resnet50':
    cnn = ResNet50(num_classes=num_classes, img_size=img_size).to(device)
elif args.model == 'resnet34':
    cnn = ResNet34(num_classes=num_classes, img_size=img_size).to(device)

# optimizer and scheduler
prediction_criterion = nn.NLLLoss().to(device)
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[int(args.epochs*(1/3)), int(args.epochs*(2/3))], gamma=0.1) #80/120
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[int(args.epochs*(1/4)), int(args.epochs*(2/4)), int(args.epochs*(3/4))], gamma=0.2) #60/120/160

if args.model == 'densenet':
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                    momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = MultiStepLR(cnn_optimizer, milestones=[150, 225], gamma=0.1)

# ---------------------------------------------------------------------------------------------------------------------
# Network training
csv_logger = CSVLogger(args=args, filename=logdir+'/'+filename+'.csv',
                       fieldnames=['epoch', 'train_acc', 'test_acc'])

# start with a reasonable guess for lambda
lmbda = 0.1

for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    confidence_loss_avg = 0.
    correct_count = 0.
    total = 0.

    # ------------- TRAINING LOOP -------------
    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images, labels = images.to(device), labels.to(device)
        labels_onehot = encode_onehot(labels, num_classes, device)

        cnn.zero_grad()

        pred_original, confidence = cnn(images)

        pred_original = F.softmax(pred_original, dim=-1)
        sig = nn.Sigmoid()
        confidence = sig(confidence)

        # make sure we don't have any numerical instability
        eps = 1e-12
        pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
        confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

        if not args.baseline:
            # randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(device)
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
            pred_new = torch.log(pred_new)
        else:
            pred_new = torch.log(pred_original)

        xentropy_loss = prediction_criterion(pred_new, labels)
        confidence_loss = torch.mean(-torch.log(confidence))

        if args.baseline:
            total_loss = xentropy_loss
        else:
            total_loss = xentropy_loss + (lmbda * confidence_loss)

            if args.budget > confidence_loss.item(): 
                lmbda = lmbda / 1.01
            elif args.budget <= confidence_loss.item(): 
                lmbda = lmbda / 0.99

        total_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()
        confidence_loss_avg += confidence_loss.item() 

        pred_idx = torch.max(pred_original.data, 1)[1]
        total += labels.size(0)
        correct_count += (pred_idx == labels.data).sum()
        accuracy = correct_count / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            confidence_loss='%.3f' % (confidence_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    # ------------- VALIDATION LOOP -------------
    test_acc, conf_min, conf_max, conf_avg = test(test_loader, device)
    tqdm.write('test_acc: %.3f, conf_min: %.3f, conf_max: %.3f, conf_avg: %.3f' % (test_acc, conf_min, conf_max, conf_avg))

    scheduler.step()

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

    # save model
    torch.save(cnn.state_dict(), modeldir+'/' + filename + '.pt')

csv_logger.close()
