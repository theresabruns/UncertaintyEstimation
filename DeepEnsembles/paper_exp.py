from __future__ import division, print_function
import time
from tqdm import tqdm
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Deep_Ensemble.model import *
from src.utils import *
from prepare_data import *
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Script to reproduce Deep Ensemble experiments from Lak+17 paper')
parser.add_argument('--id', type=int, help='Identifier for histogram storage in different experiments on same model')
parser.add_argument('--data', type=str, default='mnist', choices=['svhn', 'mnist', 'usecase'], help='ID data used for model training')
parser.add_argument('--model', type=str, default='linear', choices=['conv', 'linear'], 
                    help='Base model architecture for the ensemble members')
parser.add_argument('--batch_size', type=int, nargs='?', action='store', default=64,
                    help='batch size for the test data. Default: 64.')
parser.add_argument('--num_runs', type=int, default=10, help='#runs with different amounts of members')
parser.add_argument("--num_bins", type=int, default=20, help="bin number for ECE")
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, choice: 0/1, default: 0')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='Ensemble_models/paper_experiments',
                    help='Where to get learnt models from. Default: \'Ensemble_models/paper_experiments\'')
parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\'')
args = parser.parse_args()

models_dir = args.models_dir
mkdir(models_dir)

# set device
if torch.cuda.is_available():
    use_cuda = True
    device = torch.device("cuda:"+str(args.gpu))
else:
    use_cuda = False
    device = torch.device("cpu")
print("Device used: ", device)

# Load the data
datapath = '../../data'
if args.data == 'mnist':
    _, _, test_loader, num_classes, input_size = get_MNISTloaders(datapath, batch_size=args.batch_size)
    ood_loader, _, _ = get_NotMNISTloaders(datapath, batch_size=args.batch_size)
    ood_data = 'notmnist'
    channels_in=1
elif args.data == 'svhn':
    _, _, test_loader, num_classes, input_size = get_SVHNloaders(datapath, batch_size=args.batch_size)
    _, _, ood_loader, _, _ = get_CIFAR10loaders(datapath, batch_size=args.batch_size, rgb_flag=True)
    ood_data = 'cifar10'
    channels_in=3
elif args.data == 'usecase':
    datapath = args.uc_datapath
    _, _, test_loader, ood_loader, num_classes, input_size = get_UseCaseloaders(datapath, batch_size=args.batch_size)
    ood_data = 'usecase'
    channels_in=3

# Load model
net = Ens_net(channels_in=channels_in, side_in=input_size, cuda=use_cuda, device=device, 
            classes=num_classes, batch_size=args.batch_size, data=args.data, model=args.model)

batch_errors = np.zeros(args.num_runs)
batch_nlls = np.zeros(args.num_runs)
batch_briers = np.zeros(args.num_runs)

cprint('g', '\nGetting predictions on ID data:')
for j, (images, labels) in enumerate(tqdm(test_loader)):
    images = images.to(device)
    labels = labels.to(device)
    
    for i in range(args.num_runs):
        error, nll, brier, probs, _ = net.get_sample_scores(images, labels, num_classes, modeldir=models_dir, model=args.model, seed=i, data=args.data)
        probs = probs.detach().cpu().numpy()
        batch_errors[i] = error
        batch_nlls[i] = nll
        batch_briers[i] = brier
        batch_confs = np.max(probs, 1) #(batch_size,) conf values for predicted class (i.e. highest)
        batch_preds = np.argmax(probs, 1) #(batch_size,) index for predicted class
        id_entropy = entropy(probs, axis=1) #(batch_size, )
        if i == 0:
            confs = batch_confs.reshape((-1, batch_confs.shape[0])) #confidences for batch j
            preds = batch_preds.reshape((-1, batch_preds.shape[0])) #predictions for batch j
            batch_id_entropies = id_entropy.reshape((-1, id_entropy.shape[0])) #ID entropies for batch j
        else:
            confs = np.concatenate((confs, batch_confs.reshape((-1, batch_confs.shape[0]))), axis=0) #stack seed-wise confs in rows -> (num_runs, batch_size)
            preds = np.concatenate((preds, batch_preds.reshape((-1, batch_preds.shape[0]))), axis=0) #stack seed-wise preds in rows -> (num_runs, batch_size)
            batch_id_entropies = np.concatenate((batch_id_entropies, id_entropy.reshape((-1, id_entropy.shape[0]))), axis=0) #stack seed-wise entr in rows -> (num_runs, batch_size)
    labels = labels.detach().cpu().numpy()

    if j == 0: #first batch
        confidences = confs.reshape((confs.shape[0], -1))
        predictions = preds.reshape((preds.shape[0], -1))
        id_entropies = batch_id_entropies.reshape((batch_id_entropies.shape[0], -1))
        all_labels = labels
        errors = batch_errors
        nlls = batch_nlls
        briers = batch_briers
    else: 
        #stack batches after each other in columns, rows: different seeds -> (num_runs, total_num_images)
        confidences = np.concatenate((confidences, confs.reshape((confs.shape[0], -1))), axis=1)
        predictions = np.concatenate((predictions, preds.reshape((preds.shape[0], -1))), axis=1)
        id_entropies = np.concatenate((id_entropies, batch_id_entropies.reshape((batch_id_entropies.shape[0], -1))), axis=1)
        all_labels = np.concatenate((all_labels, labels))
        errors = (errors + batch_errors) / 2
        nlls = (nlls + batch_nlls) / 2
        briers = (briers + batch_briers) / 2
    
cprint('g', '\nGetting predictions on OOD data:')
for j, (ood_images, ood_labels) in enumerate(tqdm(ood_loader)):
    ood_images = ood_images.to(device)
    ood_labels = ood_labels.to(device)

    for i in range(args.num_runs):
        _, _, _, ood_probs, _ = net.get_sample_scores(ood_images, ood_labels, num_classes, modeldir=models_dir, model=args.model, seed=i, data=args.data)
        ood_probs = ood_probs.detach().cpu().numpy()
        ood_entropy = entropy(ood_probs, axis=1) #(batch_size, )

        if i == 0:
            batch_ood_entropies = ood_entropy.reshape((-1, ood_entropy.shape[0])) #ID entropies for batch j
        else:
            batch_ood_entropies = np.concatenate((batch_ood_entropies, ood_entropy.reshape((-1, ood_entropy.shape[0]))), axis=0) #stack seed-wise entr in rows -> (num_runs, batch_size)

    if j == 0: #first batch
        ood_entropies = batch_ood_entropies.reshape((batch_ood_entropies.shape[0], -1))
    else:
        ood_entropies = np.concatenate((ood_entropies, batch_ood_entropies.reshape((batch_ood_entropies.shape[0], -1))), axis=1)

print("Errors: ", errors, errors.shape)
print("NLLs: ", nlls, nlls.shape)
print("Brier Scores: ", briers, briers.shape)
print("Confidences: ", confidences, confidences.shape)
print("Predictions: ", predictions, predictions.shape)
print("ID Entropies: ", id_entropies, id_entropies.shape)
print("OOD Entropies: ", ood_entropies, ood_entropies.shape)
print("Labels: ", all_labels, all_labels.shape)

#plot and save ID classification metric curces
plot_idclassif_metrics(args.num_runs, errors, nlls, briers, args.data, args.id)

#plot and save ID vs OOD detection metric curces
plot_idood_metrics(id_entropies, ood_entropies, args.data, ood_data, args.id, idxs=[0,4,9])