from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
import matplotlib
from src.Bayes_By_Backprop.model import *
from src.Bayes_By_Backprop_Local_Reparametrization.model import *
from prepare_data import *
import numpy as np
import timm
from tqdm import tqdm 

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Bayesian Neural Net with BBB Variational Inference')
parser.add_argument('--weight_decay', type=float, nargs='?', action='store', default=1,
                    help='Specify the precision of an isotropic Gaussian prior. Default: 1.')
parser.add_argument('--model', type=str, nargs='?', action='store', default='Local_Reparam',
                    help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\', \'GMM_prior\','
                         ' \'Local_Reparam\'. Default: \'Local_Reparam\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of weight prior. Default: 0.1.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=50,
                    help='How many epochs to train. Default: 50.')
parser.add_argument('--batch_size', type=int, nargs='?', action='store', default=64,
                    help='batch size for the data. Default: 64.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
parser.add_argument('--nhid', type=int, nargs='?', action='store', default=200,
                    help='number of hidden units.')
parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'svhn', 'cifar10', 'usecase'], 
                    help='Dataset to train the BBB model on')
parser.add_argument('--usecase', type=str, default='RR', choices=['RR', 'Emblem'], 
                    help='Which use case data the model was trained on')
parser.add_argument('--aug', action='store_true', help='whether to use advanced augmentation on images')
parser.add_argument('--seed', type=int, default=0, help='Seed for storing the model')
parser.add_argument('--gpu', type=int, default=1, help='Which GPU to use, choice: 0/1, default: 1')
parser.add_argument('--n_samples', type=float, nargs='?', action='store', default=50,
                    help='How many MC samples to take when approximating the ELBO. Default: 50.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='BBP_models',
                    help='Where to save learnt weights and train vectors. Default: \'BBP_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='BBP_results',
                    help='Where to save learnt training plots. Default: \'BBP_results\'.')
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
writer = SummaryWriter(log_dir=f"{results_dir}/{args.model}_seed{str(args.seed)}_lr{str(args.lr)}_wd{str(args.weight_decay)}_prior{str(args.prior_sig)}")

# train config
batch_size = args.batch_size
nb_epochs = args.epochs
lr = args.lr
nhid = args.nhid
nsamples = int(args.n_samples)  # How many samples to estimate ELBO with at each iteration

# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c', '\nData:')

# load data
datapath = '../../data'
if args.data == 'mnist':
    train_loader, val_loader, test_loader, num_classes, input_size = get_MNISTloaders(datapath, batch_size=args.batch_size)
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
cprint('c', '\nNetwork:')
cprint('c', '\nUsing model: '+args.model)

# initialize model
if args.model == 'Local_Reparam':
    net = BBP_Bayes_Net_LR(lr=lr, channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        Nbatches=len(train_loader), weight_decay=args.weight_decay, nhid=nhid, prior_sig=args.prior_sig)
elif args.model == 'Laplace_prior':
    net = BBP_Bayes_Net(lr=lr, channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        Nbatches=len(train_loader), weight_decay=args.weight_decay, nhid=nhid,
                        prior_instance=laplace_prior(mu=0, b=args.prior_sig))
elif args.model == 'Gaussian_prior':
    net = BBP_Bayes_Net(lr=lr, channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        Nbatches=len(train_loader), weight_decay=args.weight_decay, nhid=nhid,
                        prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))
elif args.model == 'GMM_prior':
    net = BBP_Bayes_Net(lr=lr, channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        Nbatches=len(train_loader), weight_decay=args.weight_decay, nhid=nhid,
                        prior_instance=spike_slab_2GMM(mu1=0, mu2=0, sigma1=args.prior_sig, sigma2=0.0005, pi=0.75))
else:
    print('Invalid model type')
    exit(1)

net.model = net.model.to(device)

cprint('c', '\nTrain:')

print('  init cost variables:')
kl_cost_train = np.zeros(nb_epochs)
pred_cost_train = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)
acc_train = np.zeros(nb_epochs)

cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
acc_dev = np.zeros(nb_epochs)
best_err = np.inf
best_acc = 0.0

nb_its_dev = 1 # frequency of validation phase

tic0 = time.time()
for i in range(nb_epochs):
    # We draw more samples on the first epoch in order to ensure convergence
    if i == 0:
        ELBO_samples = 10
    else:
        ELBO_samples = nsamples

    net.set_mode_train(True)
    tic = time.time()
    nb_samples = 0

    # ------------- TRAINING LOOP -------------
    for x, y in tqdm(train_loader):
        cost_dkl, cost_pred, err = net.fit(x, y, samples=ELBO_samples)

        err_train[i] += err
        acc_train[i] += (len(x) - err)
        kl_cost_train[i] += cost_dkl
        pred_cost_train[i] += cost_pred
        nb_samples += len(x)

    # Normalise by number of samples in order to get comparable number to the -log like
    kl_cost_train[i] /= nb_samples  
    pred_cost_train[i] /= nb_samples
    err_train[i] /= nb_samples
    acc_train[i] /= nb_samples

    toc = time.time()
    net.epoch = i

    # print and log train results
    writer.add_scalar("Loss/train", pred_cost_train[i], i+1)
    writer.add_scalar("Error/train", err_train[i], i+1)
    writer.add_scalar("Accuracy/train", acc_train[i], i+1)
    print("it %d/%d, Jtr_KL = %f, Jtr_pred = %f, err = %f, acc = %f, " % (
    i+1, nb_epochs, kl_cost_train[i], pred_cost_train[i], err_train[i], acc_train[i]), end="")
    cprint('r', '   time: %f seconds\n' % (toc - tic))

    # ------------- VALIDATION LOOP -------------
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        for j, (x, y) in enumerate(tqdm(val_loader)):
            cost, err, probs = net.eval(x, y)  # This takes the expected weights to save time, not proper inference

            cost_dev[i] += cost
            err_dev[i] += err
            acc_dev[i] += (len(x) - err)
            nb_samples += len(x)

        cost_dev[i] /= nb_samples
        err_dev[i] /= nb_samples
        acc_dev[i] /= nb_samples

        # print and log val results
        writer.add_scalar("Loss/val", cost_dev[i], i+1)
        writer.add_scalar("Error/val", err_dev[i], i+1)
        writer.add_scalar("Accuracy/val", acc_dev[i], i+1)
        cprint('g', '    Jdev = %f, err = %f, acc = %f\n' % (cost_dev[i], err_dev[i], acc_dev[i]))

        # save model with lowest validation error
        if err_dev[i] <= best_err:
            best_err = err_dev[i]
            cprint('b', 'best test error')
            net.save(models_dir+'/'+args.model+'_'+dataset+'_seed'+str(args.seed)+'.pt')

        if acc_dev[i] < best_acc:
            best_acc = acc_dev[i]
            cprint('b', 'best val accuracy')

writer.close()
toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)

## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c', '\nRESULTS:')
nb_parameters = net.get_nb_parameters()
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