import pdb
import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.autograd import Variable

import seaborn as sns

from models.vgg import VGG16
from models.linear import Linear_2L
from models.densenet import DenseNet121
from models.wideresnet import WideResNet
from models.resnet import ResNet50, ResNet34
from utils.datasets import *
from utils.utils import *

ind_options = ['cifar10', 'svhn', 'RR', 'Emblem']
ood_options = ['tinyImageNet_crop', 'tinyImageNet_resize', 'LSUN_crop', 'LSUN_resize', 
                'iSUN', 'Uniform', 'Gaussian', 'all', 'RR', 'Emblem']
model_options = ['densenet', 'wideresnet', 'linear', 'vgg16', 'resnet50', 'resnet34']
process_options = ['baseline', 'ODIN', 'confidence', 'confidence_scaling']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--ind_dataset', default='cifar10', choices=ind_options)
parser.add_argument('--ood_dataset', default='tinyImageNet_resize', choices=ood_options)
parser.add_argument('--model', default='vgg13', choices=model_options)
parser.add_argument('--seed', type=int, default=0, help='Seed for stored model')
parser.add_argument('--id', type=int, help='Additional identifier for results in different experiments on same model')
parser.add_argument('--budget', type=float, default=0.3, metavar='N',
                    help='the hint budget used for network training')
parser.add_argument('--process', default='confidence', choices=process_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--T', type=float, default=1000., help='Scaling temperature')
parser.add_argument('--epsilon', type=float, default=0.001, help='Noise magnitude')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, choice: 0/1, default: 0')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='checkpoints',
                    help='Where to get learnt models from. Default: \'checkpoints\'.')
parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='\'../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\' or \'../../data/EmblemUseCase\'')
parser.add_argument('--save_uncimgs', action='store_true', help='whether to store the images sorted by their assigned uncertainty')
parser.add_argument('--save_txt', action='store_true', help='whether to store results in txt-file')
parser.add_argument('--final_model', action='store_true', help='for final model, store arrays for integrative comparison plots')

args = parser.parse_args()
print(args)

cudnn.benchmark = True  # should make training go faster for large models

# set device
if torch.cuda.is_available():
    use_cuda = True
    device = torch.device("cuda:"+str(args.gpu))
else:
    use_cuda = False
    device = torch.device("cpu")
print("Device used: ", device)

filename = args.ind_dataset + '_' + args.model + '_budget_' + str(args.budget) + '_seed_' + str(args.seed) 

if args.ind_dataset == 'svhn' and args.model == 'wideresnet':
    args.model = 'wideresnet16_8'

# Load the data
datapath = '../../data'
if args.ind_dataset == 'RR' and args.ood_dataset == 'RR':
    datapath = args.uc_datapath
    _, ind_loader, ood_loader, num_classes, img_size = get_Elementloaders(datapath, batch_size=args.batch_size)
    data = 'RRusecase'
elif args.ind_dataset == 'Emblem' and args.ood_dataset == 'Emblem':
    datapath = args.uc_datapath
    _, ind_loader, ood_loader, num_classes, img_size = get_Emblemloaders(datapath, batch_size=args.batch_size)
    data = 'Emblemusecase'
else:
    if args.ind_dataset == 'svhn':
        _, ind_loader, num_classes, img_size = get_SVHNloaders(datapath, batch_size=args.batch_size)
    elif args.ind_dataset == 'cifar10':
        _, ind_loader, num_classes, img_size = get_CIFAR10loaders(datapath, batch_size=args.batch_size)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    crop_transform = transforms.Compose([transforms.CenterCrop(size=(32, 32)),
                                        transforms.ToTensor(),
                                        normalize])

    if args.ood_dataset == 'tinyImageNet_crop':
        ood_dataset = datasets.ImageFolder(root=datapath + '/TinyImageNet_crop', transform=crop_transform)
    elif args.ood_dataset == 'tinyImageNet_resize':
        ood_dataset = datasets.ImageFolder(root=datapath + '/TinyImagenet_resize', transform=transform)
    elif args.ood_dataset == 'LSUN_crop':
        ood_dataset = datasets.ImageFolder(root=datapath + '/LSUN_crop', transform=crop_transform)
    elif args.ood_dataset == 'LSUN_resize':
        ood_dataset = datasets.ImageFolder(root=datapath + '/LSUN_resize', transform=transform)
    elif args.ood_dataset == 'iSUN':
        ood_dataset = datasets.ImageFolder(root=datapath + 'iSUN', transform=transform)
    elif args.ood_dataset == 'Uniform':
        ood_dataset = UniformNoise(size=(3, 32, 32), n_samples=10000, low=0., high=1.)
    elif args.ood_dataset == 'Gaussian':
        ood_dataset = GaussianNoise(size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0)
    elif args.ood_dataset == 'all':
        ood_dataset = torch.utils.data.ConcatDataset([
            datasets.ImageFolder(root=datapath + 'TinyImageNet_crop', transform=crop_transform),
            datasets.ImageFolder(root=datapath + 'TinyImagenet_resize', transform=transform),
            datasets.ImageFolder(root=datapath + 'LSUN_crop', transform=crop_transform),
            datasets.ImageFolder(root=datapath + 'LSUN_resize', transform=transform),
            datasets.ImageFolder(root=datapath + 'iSUN', transform=transform)])
    data = args.ind_dataset + '_' + args.ood_dataset

    ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=2)

# Load model
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

model_dict = cnn.state_dict()

pretrained_dict = torch.load(args.models_dir + '/' + filename + '.pt')
cnn.load_state_dict(pretrained_dict)
cnn = cnn.to(device)

cnn.eval()

# --------------------------------------------------------------------------------------
# helper function for model evaluation
def evaluate(data_loader, mode, num_classes):
    xent = nn.CrossEntropyLoss()
    for j, data in enumerate(data_loader):
        images, labels = data
        images = Variable(images, requires_grad=True).to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]

        images.retain_grad()
        sig = nn.Sigmoid()

        # this is the case relevant for Learning Confidence evaluation
        if mode == 'confidence':
            pred, confidence = cnn(images)

            probs = F.softmax(pred, dim=-1)
            batch_preds = probs.max(dim=1, keepdim=False)[1]
            batch_error = batch_preds.ne(labels).sum()
            batch_error = batch_error / batch_size

            batch_preds = batch_preds.detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()
    
            confidence = sig(confidence)
            # NLL score
            batch_nll = F.nll_loss(probs, labels)
            batch_nll = -batch_nll / batch_size

            confidence = confidence.data.cpu().numpy()
            # Brier score
            one_hot = np.zeros((batch_size, num_classes))
            for i in range(labels.shape[0]):
                one_hot[i][labels[i]] = 1
            diff = np.power((one_hot - confidence), 2)
            batch_brier = np.sum(diff, axis=1) / num_classes
            batch_brier = np.sum(batch_brier) / batch_size
        
            if j == 0: # first batch
                all_labels = labels
                all_images = images
                out = confidence
                predictions = batch_preds.reshape((batch_preds.shape[0], -1))
                error = batch_error
                nll = batch_nll
                brier = batch_brier
            else: # stack batch results in columns
                all_labels = torch.cat((all_labels, labels))
                all_images = torch.cat((all_images, images))
                out = np.concatenate((out, confidence.reshape((confidence.shape[0], -1))))
                predictions = np.concatenate((predictions, batch_preds.reshape((batch_preds.shape[0], -1))))
                error = (error + batch_error) / 2
                nll = (nll + batch_nll) / 2
                brier = (brier + batch_brier) / 2

        elif mode == 'confidence_scaling':
            epsilon = args.epsilon

            cnn.zero_grad()
            _, confidence = cnn(images)
            confidence = sig(confidence).view(-1)
            loss = torch.mean(-torch.log(confidence))
            loss.backward()

            images = images - args.epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            if j == 0:
                out = confidence
            else:
                out = np.concatenate((out, confidence.reshape((confidence.shape[0], -1))))

        elif mode == 'baseline':
            # https://arxiv.org/abs/1610.02136
            pred, _ = cnn(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            if j == 0:
                out = pred
            else:
                out = np.concatenate((out, pred.reshape((pred.shape[0], -1))))

        elif mode == 'ODIN':
            # https://arxiv.org/abs/1706.02690
            T = args.T
            epsilon = args.epsilon

            cnn.zero_grad()
            pred, _ = cnn(images)
            _, pred_idx = torch.max(pred.data, 1)
            labels = Variable(pred_idx)
            pred = pred / T
            loss = xent(pred, labels)
            loss.backward()

            images = images - epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            pred, _ = cnn(images)

            pred = pred / T
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            if j == 0:
                out = pred
            else:
                out = np.concatenate((out, pred.reshape((pred.shape[0], -1))))

    return out, all_labels, all_images, predictions, error, nll, brier

# Inference on ID data
ind_scores, all_labels, all_images, predictions, error, nll, brier = evaluate(ind_loader, args.process, num_classes)
# Inference on OOD data
ood_scores, _, all_ood_images, _, _, _, _ = evaluate(ood_loader, args.process, num_classes)
ind_scores = ind_scores.flatten()
ood_scores = ood_scores.flatten()
flat_confs = ind_scores.flatten()
flat_preds = predictions.flatten()

# ------------------------------------------------------ ID detection metrics -----------------------------------------------------------
avg_acc = 1.0 - error

# out-of-the box calibration via ECE and reliability diagram
np_labels = all_labels.detach().cpu().numpy()
num_bins = 20 # bins for calibration curve and ECE
bin_confidences, bin_accuracies, ece = plot_calibration_diagram(flat_confs, flat_preds, np_labels, num_bins, args.model, args.seed, args.id, data)

# ------------------------------------------------------ OOD detection metrics -----------------------------------------------------------
# plot and save histogram
kl_div = plot_histograms(ind_scores, data, args.model, args.seed, id=args.id, ood_confidences=ood_scores)

# perform binary ID/OOD classification via measure 
tpr, fpr, thresholds, auroc, UA_values = calc_OODmetrics(ind_scores, ood_scores)
# calculate AUROC and plot ROC curve
plot_roc(tpr, fpr, args.model, data, auroc, UA_values, args.seed, args.id)

# plot accuracy-rejection curves
fractions, accuracies, ideal, filtered_imgs, filtered_uncs, filtered_lbls = calc_rejection_curve(all_labels, flat_preds, ind_scores, ood_scores, device, all_images, all_ood_images)
plot_accrej_curve(fractions, accuracies, ideal, args.model, data, args.seed, args.id)

# save results to txt.file
if args.save_txt:
    file_path = "result_files"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    results_file = file_path+'/%s_%s_%s_%s_id%s.txt'%(args.model, args.process, data, str(args.seed), str(args.id))
    g = open(results_file, 'w+')

    g.write("ID Detection Metrics: \n\n")
    g.write("Labels: \t\t{};".format(all_labels))
    g.write("\nPredictions: \t{};".format(flat_preds))
    g.write("\nConfidences: \t{};".format(flat_confs))
    g.write("\nError: \t\t\t{};".format(error))
    g.write("\nAccuracy: \t\t{};".format(avg_acc))
    g.write("\nNLL Score: \t\t{};".format(nll))
    g.write("\nBrier Score: \t{};".format(brier))
    g.write("\nCalibration metrics: \t")
    g.write("\nECE: \t{};".format(ece))
    g.write("\nBin Confidences: \t{};".format(bin_confidences))
    g.write("\nBin Accuracies: \t{};".format(bin_accuracies))

    g.write("\n\nOOD Detection Metrics: \n\n")
    g.write("ID Confidences: \n{};".format(ind_scores.flatten().flatten()))
    g.write("\nOOD Confidences: \n{};".format(ood_scores.flatten().flatten()))
    g.write("\nKL-divergence: \t{};".format(kl_div))
    g.write("\nUncertainty-balanced Accuracies: \n{};".format(UA_values.flatten().flatten()))
    g.write("\nUA Thresholds: \n{};".format(thresholds))
    g.write("\nAUROC: \t\t{};".format(auroc))
    g.write("\nACR Accuracies: \t{};".format(accuracies))

    g.close()

# for best results, set final_model flag 
# -> saves results to Integration directory for comparative visualization to other methods
if args.final_model:
    plot_dir =  "../Integration/LearningConfidence"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_file = plot_dir+'/%s_%s_%s_%s.npz'%(args.model, args.process, data, str(args.seed))
    np.savez(plot_file, arr0=ind_scores, 
                        arr1=ood_scores, 
                        arr2=fractions, 
                        arr3=accuracies, 
                        arr4=ideal, 
                        arr5=bin_confidences,
                        arr6=bin_accuracies, 
                        arr7=tpr,
                        arr8=fpr,
                        compress=True)

# save images named and sorted by their uncertainty value
if args.save_uncimgs:
    img_path = "OUT_images/"+data+'_'+args.model+'_'+str(args.seed)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in range(filtered_imgs.size(0)):
        img = filtered_imgs[i]
        print("IMG: ", img.shape)
        # revert normalization to get original image
        if args.usecase == "RR":
            mean = torch.tensor([0.46840793, 0.23778377, 0.19240856]).to(device)
            std = torch.tensor([0.12404595643681854, 0.08136763306617903, 0.07868825907965848]).to(device)
        elif: args.usecase == "Emblem":
            mean = torch.tensor([0.3987, 0.4262, 0.4706]).to(device)
            std = torch.tensor([0.2505, 0.2414, 0.2466]).to(device)
        img = img * std[:, None, None] + mean[:, None, None]

        uncertainty_score = filtered_uncs[i].detach().cpu().numpy()
        class_label = filtered_lbls[i].detach().cpu().numpy()
        utils.save_image(img, img_path+'/'+str(uncertainty_score)+'_label'+str(class_label)+'.png')
