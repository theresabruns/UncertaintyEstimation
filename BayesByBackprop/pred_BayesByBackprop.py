from __future__ import division, print_function
import time
from tqdm import tqdm
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bayes_By_Backprop.model import *
from src.Bayes_By_Backprop_Local_Reparametrization.model import *
from src.utils import *
from prepare_data import *
import numpy as np
from scipy.stats import entropy
import gc

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Sample Bayesian Neural Net with Bayes by Backprop (BBB) Variational Inference')

parser.add_argument('--batch_size', type=int, nargs='?', action='store', default=1,
                    help='batch size for the test data. Default: 1.')
parser.add_argument('--model', type=str, nargs='?', action='store', default='Local_Reparam',
                    help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\', \'GMM_prior\','
                         ' \'Local_Reparam\'. Default: \'Local_Reparam\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--seed', type=int, default=0, help='Seed for stored model')
parser.add_argument('--id', type=int, help='Additional identifier for histogram storage in different experiments on same model')
parser.add_argument('--n_samples', type=int, nargs='?', action='store', default=20,
                    help='How many MC samples to take when approximating the ELBO. Default: 20.')
parser.add_argument('--nhid', type=int, nargs='?', action='store', default=600,
                    help='number of hidden units.')
parser.add_argument('--measure', type=str, default='entropy', choices=['entropy', 'mutualinfo', 'conf', 'avg-entropy'], help='Uncertainty measure')
parser.add_argument('--data', type=str, default='usecase', choices=['svhn', 'mnist', 'usecase'], help='ID data used for model training')
parser.add_argument('--ood_data', type=str, default='usecase', choices=['cifar10', 'notmnist', 'fashionmnist', 'usecase'], help='OOD data to test model on')
parser.add_argument('--usecase', type=str, default='RR', choices=['RR', 'Emblem'], 
                    help='Which use case data the model was trained on')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, choice: 0/1, default: 0')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='BBP_models',
                    help='Where to get learnt models from. Default: \'BBP_models\'.')
parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='\'../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\' or \'../../data/EmblemUseCase\'')
parser.add_argument('--save_uncimgs', action='store_true', help='whether to store the images sorted by their assigned uncertainty')
parser.add_argument('--save_txt', action='store_true', help='whether to store results in txt-file')
parser.add_argument('--final_model', action='store_true', help='for final model, store numpy arrays for integrative comparison plots')
args = parser.parse_args()
print(args)

gc.collect()
torch.cuda.empty_cache()

models_dir = args.models_dir
Nsamples = args.n_samples
nhid = args.nhid
batch_size = args.batch_size
num_bins = 20 # bins for calibration curve and ECE

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
if args.ood_data == 'usecase' and args.data == 'usecase':
    datapath = args.uc_datapath
    if args.usecase == 'RR':
        _, _, test_loader, ood_loader, num_classes, input_size = get_Elementloaders(datapath, batch_size=args.batch_size)
        data = 'RRusecase'
    elif args.usecase == 'Emblem':
        _, _, test_loader, ood_loader, num_classes, input_size = get_Emblemloaders(datapath, batch_size=args.batch_size)
        data = 'Emblemusecase'
    channels_in=3
elif args.data == 'mnist':
    _, _, test_loader, num_classes, input_size = get_MNISTloaders(datapath, batch_size=args.batch_size)
    channels_in=1
    data = 'mnist'
    if args.ood_data == "fashionmnist":
        ood_loader, _ = get_FashionMNISTloaders(datapath, batch_size=args.batch_size)
    elif args.ood_data == "notmnist":
        ood_loader, _ = get_NotMNISTloaders(datapath, batch_size=args.batch_size)
    else:
        _, _, ood_loader, num_classes, input_size = get_CIFAR10loaders(datapath, batch_size=args.batch_size)
elif args.data == 'svhn':
    _, _, test_loader, num_classes, input_size = get_SVHNloaders(datapath, batch_size=args.batch_size)
    _, _, ood_loader, _, _ = get_CIFAR10loaders(datapath, batch_size=args.batch_size, rgb_flag=True)
    data = 'svhn'
    channels_in=3


# Load model
if args.model == 'Local_Reparam':
    net = BBP_Bayes_Net_LR(channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        nhid=nhid, prior_sig=args.prior_sig)
elif args.model == 'Laplace_prior':
    net = BBP_Bayes_Net(channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        nhid=nhid, prior_instance=laplace_prior(mu=0, b=args.prior_sig))
elif args.model == 'Gaussian_prior':
    net = BBP_Bayes_Net(channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        nhid=nhid, prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))
elif args.model == 'GMM_prior':
    net = BBP_Bayes_Net(channels_in=channels_in, side_in=input_size, cuda=use_cuda, 
                        device=device, classes=num_classes, batch_size=batch_size, 
                        nhid=nhid, prior_instance=spike_slab_2GMM(mu1=0, mu2=0, sigma1=args.prior_sig, sigma2=0.0005, pi=0.75))
else:
    print('Invalid model type')
    exit(1)

net.load(models_dir+'/'+args.model+'_'+data+'_seed'+str(args.seed)+'.pt')
net.model = net.model.to(device)


# ------------------------------- Inference on ID data -------------------------------
cprint('g', '\nGetting predictions on ID data:')
for j, (images, labels) in enumerate(tqdm(test_loader)):
    images, labels = images.to(device), labels.to(device)

    # results from average of samples
    batch_error, batch_nll, batch_brier, probs, mean_entropy = net.sample_eval(images, labels, num_classes, Nsamples)

    batch_confs, batch_preds = probs.max(dim=1, keepdim=False) # returns (values, indices) of shape (batch_size,)
    batch_confs = batch_confs.detach().cpu().numpy()
    batch_preds = batch_preds.detach().cpu().numpy()
    batch_id_entropies = entropy(probs.detach().cpu().numpy(), axis=1) # (batch_size, )
    mutualinfos = batch_id_entropies - mean_entropy.detach().cpu().numpy()

    if j == 0: # first batch
        confidences = batch_confs.reshape((batch_confs.shape[0], -1))
        predictions = batch_preds.reshape((batch_preds.shape[0], -1))
        id_entropies = batch_id_entropies.reshape((batch_id_entropies.shape[0], -1))
        all_labels = labels
        all_images = images
        error = batch_error
        nll = batch_nll
        brier = batch_brier
    else: # stack batch results in columns
        confidences = np.concatenate((confidences, batch_confs.reshape((batch_confs.shape[0], -1))))
        predictions = np.concatenate((predictions, batch_preds.reshape((batch_preds.shape[0], -1))))
        id_entropies = np.concatenate((id_entropies, batch_id_entropies.reshape((batch_id_entropies.shape[0], -1))))
        all_labels = torch.cat((all_labels, labels))
        all_images = torch.cat((all_images, images))
        error = (error + batch_error) / 2
        nll = (nll + batch_nll) / 2
        brier = (brier + batch_brier) / 2

# ------------------------------- Inference on OOD data -------------------------------
cprint('g', '\nGetting predictions on OOD data:')
for j, (ood_images, ood_labels) in enumerate(tqdm(ood_loader)):
    ood_images, ood_labels = ood_images.to(device), ood_labels.to(device)

    #results from average of samples
    _, _, _, ood_probs, ood_mean_entropy = net.sample_eval(ood_images, ood_labels, num_classes, Nsamples)

    batch_ood_confs, _ = ood_probs.max(dim=1, keepdim=False) #returns (value, indice)
    batch_ood_confs = batch_ood_confs.detach().cpu().numpy()
    batch_ood_entropies = entropy(ood_probs.detach().cpu().numpy(), axis=1) #(batch_size, )
    ood_mutualinfos = batch_ood_entropies - ood_mean_entropy.detach().cpu().numpy()
    if j == 0: #first batch
        all_ood_images = ood_images
        ood_confidences = batch_ood_confs.reshape((batch_ood_confs.shape[0], -1))
        ood_entropies = batch_ood_entropies.reshape((batch_ood_entropies.shape[0], -1))
    else: # stack batch results in columns
        all_ood_images = torch.cat((all_ood_images, ood_images))
        ood_confidences = np.concatenate((ood_confidences, batch_ood_confs.reshape((batch_ood_confs.shape[0], -1))))
        ood_entropies = np.concatenate((ood_entropies, batch_ood_entropies.reshape((batch_ood_entropies.shape[0], -1))))

# ------------------------------------------------------ ID detection metrics -----------------------------------------------------------
avg_acc = 1.0 - error # sum(accuracies) / len(accuracies)
flat_confs = confidences.flatten()
flat_preds = predictions.flatten()

# out-of-the box calibration via ECE and reliability diagram
np_labels = all_labels.detach().cpu().numpy()
bin_confidences, bin_accuracies, ece = plot_calibration_diagram(flat_confs, flat_preds, np_labels, num_bins, args.model, args.seed, args.id, data, args.ood_data, Nsamples)

# ------------------------------------------------------ OOD detection metrics -----------------------------------------------------------
# choose uncertainty measure among entropy, max. softmax confidence and mutual information
if args.measure == 'entropy':
    id_array = np.array(id_entropies).flatten()
    ood_array = np.array(ood_entropies).flatten()
elif args.measure == 'conf':
    id_array = np.array(confidences).flatten()
    ood_array = np.array(ood_confidences).flatten()
elif args.measure == 'mutualinfo':
    id_array = np.array(mutualinfos).flatten()
    ood_array = np.array(ood_mutualinfos).flatten()

# plot and save histogram
kl_div = plot_histogram(id_array, ood_array, args.model, args.measure, args.seed, args.id, data, args.ood_data, Nsamples)

# perform binary ID/OOD classification via measure 
tpr, fpr, thresholds, auroc, UA_values = calc_OODmetrics(id_array, ood_array)

# calculate AUROC and plot ROC curve
plot_roc(tpr, fpr, args.model, data, args.ood_data, auroc, UA_values, args.seed, args.id)

# plot accuracy-rejection curves
fractions, accuracies, ideal, filtered_imgs, filtered_uncs = calc_rejection_curve(all_labels, flat_preds, id_array, ood_array, device, all_images, all_ood_images)
plot_accrej_curve(fractions, accuracies, ideal, args.model, data, args.ood_data, args.seed, args.id)

# save results to txt.file
if args.save_txt:
    file_path = "result_files"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    results_file = file_path+'/%s_%s_%s_%s_id%s.txt'%(args.model, args.measure, data, str(args.seed), str(args.id))
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
    g.write("ID Confidences: \n{};".format(confidences.flatten().flatten()))
    g.write("\nOOD Confidences: \n{};".format(ood_confidences.flatten().flatten()))
    g.write("\nID Entropies: \n{};".format(id_entropies.flatten().flatten()))
    g.write("\nOOD Entropies: \n{};".format(ood_entropies.flatten().flatten()))
    g.write("\nKL-divergence: \t{};".format(kl_div))
    g.write("\nUncertainty-balanced Accuracies: \n{};".format(UA_values.flatten().flatten()))
    g.write("\nUA Thresholds: \n{};".format(thresholds))
    g.write("\nAUROC: \t\t{};".format(auroc))
    g.write("\nACR Accuracies: \t{};".format(accuracies))

    g.close()

# for best results, set final_model flag 
# -> saves results to Integration directory for comparative visualization to other methods
if args.final_model:
    plot_dir =  "../Integration/BBB"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_file = plot_dir+'/%s_%s_%s_%s.npz'%(args.model, args.measure, data, str(args.seed))
    np.savez(plot_file, arr0=id_array, 
                        arr1=ood_array, 
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
        #revert normalization to get original image
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
