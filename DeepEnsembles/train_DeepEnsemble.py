from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
import matplotlib
from src.Deep_Ensemble.model import *
from prepare_data import *
import numpy as np
import timm 
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Bayesian Neural Network with a Deep Ensemble')

parser.add_argument('--weight_decay', type=float, nargs='?', action='store', default=5e-4,
                    help='Specify the precision of an isotropic Gaussian prior. Default: 1.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=40,
                    help='How many epochs to train. Default: 40.')
parser.add_argument('--batch_size', type=int, nargs='?', action='store', default=100,
                    help='batch size for the data. Default: 100.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=0.1,
                    help='learning rate. Default: 0.1.')
parser.add_argument('--model', type=str, default='linear', choices=['linear', 'vgg16', 'resnet34', 'resnet50', 'densenet'], 
                    help='Base model architecture for the ensemble members')
parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar10', 'svhn', 'usecase'], 
                    help='Dataset to train the Ensemble models on')
parser.add_argument('--usecase', type=str, default='RR', choices=['RR', 'Emblem'], 
                        help='Which use case data the model was trained on')
parser.add_argument('--aug', action='store_true', help='whether to use advanced augmentation on images')
parser.add_argument('--num', type=int, default=5, help='Number of ensemble members to train')
parser.add_argument('--seed', type=int, default=0, help='Seed for storing the model')
parser.add_argument('--gpu', type=int, default=1, help='Which GPU to use, choice: 0/1, default: 0')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='Ensemble_models',
                    help='Where to save learnt weights and train vectors. Default: \'Ensemble_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='Ensemble_results',
                    help='Where to save learnt training plots. Default: \'Ensemble_results\'.')
parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='\'../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\' or \'../../data/EmblemUseCase\'')
args = parser.parse_args()

# Where to save models weights
models_dir = args.models_dir
mkdir(models_dir)

# Where to save plots and error, accuracy vectors
results_dir = args.results_dir
mkdir(results_dir)
writer = SummaryWriter(log_dir=f"{results_dir}/seed{str(args.seed)}_lr{str(args.lr)}_wd{str(args.weight_decay)}", comment="seed"+str(args.seed))

# train config
batch_size = args.batch_size
nb_epochs = args.epochs
lr = args.lr
num = args.num

#--------------------------------------------------------------------------------------------------------------
# Dataset
cprint('c', '\nData:')

# load data
datapath = '../../data'
if args.data == 'mnist':
    train_loader, val_loader, test_loader, num_classes, input_size = get_MNISTloaders(datapath, batch_size=args.batch_size, aug=args.aug)
    dataset = 'mnist'
    channels_in = 1
elif args.data == 'cifar10':
    train_loader, val_loader, test_loader, num_classes, input_size = get_CIFAR10loaders(datapath, batch_size=args.batch_size)
    dataset = 'cifar10'
    channels_in = 3
elif args.data == 'svhn':
    train_loader, val_loader, test_loader, num_classes, input_size = get_SVHNloaders(datapath, batch_size=args.batch_size)
    dataset = 'svhn'
    channels_in = 3
elif args.data == 'usecase':
    datapath = args.uc_datapath
    if args.usecase == 'RR':
        train_loader, val_loader, test_loader, _, num_classes, input_size = get_Elementloaders(datapath, batch_size=args.batch_size, aug=args.aug)
        dataset = 'RRusecase'
    elif args.usecase == 'Emblem':
        train_loader, val_loader, test_loader, _, num_classes, input_size = get_Emblemloaders(datapath, batch_size=args.batch_size, aug=args.aug)
        dataset = 'Emblemusecase'
    channels_in = 3

# set device
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")
print("Using device: ", device)


## ---------------------------------------------------------------------------------------------------------------------
# Network training
cprint('c', '\nNetworks:')
cprint('c', '\nUsing model: '+ args.model)

# initialize models and combine to ensemble
ensemble = []
for _ in range(num):
    net = Ens_net(lr=lr, channels_in=channels_in, side_in=input_size, cuda=use_cuda, device=device, 
                        classes=num_classes, batch_size=batch_size, weight_decay=args.weight_decay, 
                        n_hid=200, data=args.data, model=args.model)
    net.set_mode_train(True)
    net.model = net.model.to(device)
    ensemble.append(net)
ensemble = torch.nn.ModuleList(ensemble)

cprint('c', '\nTrain:')

tic0 = time.time()
for i, network in enumerate(ensemble):
    
    print('  init cost variables:')
    pred_cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    acc_train = np.zeros(nb_epochs)

    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    acc_dev = np.zeros(nb_epochs)
    best_err = np.inf
    best_acc = 0.0
    nb_its_dev = 1 # frequency of validation phase
    cprint('g',"---------------- Training Model "+str(i+1)+"/"+str(len(ensemble))+"------------------------------")
    tic1 = time.time()

    for j in range(nb_epochs):
        tic = time.time()
        nb_samples = 0

        # ------------- TRAINING LOOP -------------
        for x, y in tqdm(train_loader):
            cost_pred, err = network.fit(x, y)

            err_train[j] += err
            acc_train[j] += (len(x) - err)
            pred_cost_train[j] += cost_pred
            nb_samples += len(x)

        pred_cost_train[j] /= nb_samples
        err_train[j] /= nb_samples
        acc_train[j] /= nb_samples

        toc = time.time()
        network.epoch = j
        
        if i == 0: # show train curves in tensorboard only for first ensemble member
            writer.add_scalar("Loss/train", pred_cost_train[j], j+1)
            writer.add_scalar("Error/train", err_train[j], j+1)
            writer.add_scalar("Accuracy/train", acc_train[j], j+1)
        print("it %d/%d, Jtr_pred = %f, err = %f, acc = %f, " % (j+1, nb_epochs, pred_cost_train[j], err_train[j], acc_train[j]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))


        # ------------- VALIDATION LOOP -------------
        if j % nb_its_dev == 0:
            network.set_mode_train(False)
            nb_samples = 0
            for x, y in tqdm(val_loader):
                cost, err, probs = network.eval(x, y)

                cost_dev[j] += cost
                err_dev[j] += err
                acc_dev[j] += (len(x) - err)
                nb_samples += len(x)

            cost_dev[j] /= nb_samples
            err_dev[j] /= nb_samples
            acc_dev[j] /= nb_samples

            if i == 0: # show val curves in tensorboard only for first ensemble member
                writer.add_scalar("Loss/val", cost_dev[j], j+1)
                writer.add_scalar("Error/val", err_dev[j], j+1)
                writer.add_scalar("Accuracy/val", acc_dev[j], j+1)
            cprint('g', '    Jdev = %f, err = %f, acc = %f\n' % (cost_dev[j], err_dev[j], acc_dev[j]))

            # save model with lowest validation error
            if err_dev[j] < best_err:
                best_err = err_dev[j]
                cprint('b', 'best val error')
                network.save(models_dir+'/model'+str(i)+'_'+args.model+'_'+dataset+'_seed'+str(args.seed)+'.pt')

            if acc_dev[j] < best_acc:
                best_acc = acc_dev[j]
                cprint('b', 'best val accuracy')

    toc1 = time.time()
    runtime_per_it = (toc1 - tic1) / float(nb_epochs)
    cprint('r', '   Average epoch time model %i: %f seconds\n' % (i, runtime_per_it))

writer.close()
toc0 = time.time()
runtime_per_model = (toc0 - tic0) / float(num)
cprint('r', '   Average training time per model: %f seconds\n' % runtime_per_model)


## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c', '\nBEST MODEL RESULTS:')
nb_parameters = network.get_nb_parameters()
best_cost_dev = np.min(cost_dev)
best_cost_train = np.min(pred_cost_train)
best_acc_dev = np.max(acc_dev)
best_acc_train = np.max(acc_train)
err_dev_min = err_dev[::nb_its_dev].min()

print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
print('  accuracy_dev: %f (accuracy_train %f)' % (best_acc_dev, best_acc_train))
print('  err_dev: %f' % (err_dev_min))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))