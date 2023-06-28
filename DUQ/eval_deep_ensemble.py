import os
import argparse
from tqdm import tqdm 
import numpy as np
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torchvision.models import resnet18
from utils.cnn_duq import SoftmaxModel as CNN
from utils.datasets import all_datasets
from utils.evaluate_ood import *


class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.resnet = resnet18(num_classes=num_classes)
        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, n_hid):
        super().__init__()
        input_dim = in_channels * input_size * input_size
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, out_channels)
        self.bn = nn.BatchNorm1d(n_hid) 

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.bn(self.fc1(x)))
        x = F.relu(self.bn(self.fc2(x)))
        x = F.relu(self.bn(self.fc3(x)))
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)

def batch_eval(x, y, models, num_classes):
    batch_size = x.shape[0]
    logits = x.data.new(len(models), batch_size, num_classes)
    entropies = x.data.new(len(models), batch_size)

    for i, model in enumerate(models):
        logits[i] = model(x)
        prob = F.softmax(logits[i], dim=1)

        single_entropy = entropy(prob.detach().cpu().numpy().T)
        single_entropy = torch.from_numpy(single_entropy)
        entropies[i] = single_entropy

    mean_logit = logits.mean(dim=0, keepdim=False)
    mean_entropy = entropies.mean(dim=0, keepdim=False)
    probs = F.softmax(mean_logit, dim=1).data # (batch, classes)
    # classification error
    # get the indexes of the max log-probability
    pred = probs.max(dim=1, keepdim=False)[1] # (batch, )
    batch_error = pred.ne(y.data).sum() #int
    error = batch_error / batch_size #int
    # NLL score
    batch_nll = F.nll_loss(probs, y)
    nll = -batch_nll / batch_size
    # Brier score
    one_hot = np.zeros((batch_size, num_classes))
    for i in range(y.shape[0]):
        one_hot[i][y[i]] = 1
    probs_arr = probs.detach().cpu().numpy()
    diff = np.power((one_hot - probs_arr), 2)
    batch_brier = np.sum(diff, axis=1) / num_classes # sum up over classes -> (batch, )
    brier = np.sum(batch_brier) / batch_size
    return error, nll, brier, probs, mean_entropy

def test(models, dataloader, num_classes, device):
    models.eval()
    for j, (data, target) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            batch_error, batch_nll, batch_brier, probs, mean_entropy = batch_eval(data, target, models, num_classes)
            
            batch_confs, batch_preds = probs.max(dim=1, keepdim=False) # returns (values, indices) of shape (batch_size,)
            batch_confs = batch_confs.detach().cpu().numpy()
            batch_preds = batch_preds.detach().cpu().numpy()
            batch_entropies = entropy(probs.detach().cpu().numpy(), axis=1) # (batch_size, )

            if j == 0: # first batch
                confidences = batch_confs.reshape((batch_confs.shape[0], -1))
                predictions = batch_preds.reshape((batch_preds.shape[0], -1))
                entropies = batch_entropies.reshape((batch_entropies.shape[0], -1))
                all_labels = target
                all_images = data
                error = batch_error
                nll = batch_nll
                brier = batch_brier
            else: # stack batch results in columns
                confidences = np.concatenate((confidences, batch_confs.reshape((batch_confs.shape[0], -1))))
                predictions = np.concatenate((predictions, batch_preds.reshape((batch_preds.shape[0], -1))))
                entropies = np.concatenate((entropies, batch_entropies.reshape((batch_entropies.shape[0], -1))))
                all_labels = torch.cat((all_labels, target))
                all_images = torch.cat((all_images, data))
                error = (error + batch_error) / 2
                nll = (nll + batch_nll) / 2
                brier = (brier + batch_brier) / 2
    results = {
        "error": error,
        "nll": nll,
        "brier": brier,
        "confidences": confidences,
        "predictions": predictions,
        "entropies": entropies,
        "images": all_images,
        "labels": all_labels
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar",  "RR", "Emblem", "Part"], help="Select a dataset")
    parser.add_argument('--batch_size', type=int, nargs='?', action='store', default=64,
                    help='batch size for the test data. Default: 1.')
    parser.add_argument("--exp_dir", type=str, default="results", help="experiment folder used for training")
    parser.add_argument("--ensemble", type=int, default=5, help="Ensemble size (default: 5)")
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "linear", "conv"], help="architecture used (default: resnet18)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for loading trained model")
    parser.add_argument('--id', type=int, help='Additional identifier for results in different experiments on same model')
    parser.add_argument('--measure', type=str, default='entropy', choices=['entropy', 'mutualinfo', 'conf', 'avg-entropy'], help='Uncertainty measure')
    parser.add_argument('--gpu', type=int, default=1, help='Which GPU to use, choice: 0/1, default: 1')
    parser.add_argument('--save_txt', action='store_true', help='whether to store results in txt-file')
    args = parser.parse_args()
    print(args)

    device = device = torch.device("cuda:"+str(args.gpu))
    print("Device used: ", device)

    # get datasets and dataloader
    ds = all_datasets[args.dataset]()
    if args.dataset == "cifar": #load ID cifar + OOD svhn
        input_size, num_classes, _, test_dataset, _ = ds
        _, _, _ , ood_dataset, _ = all_datasets["SVHN"]()
    else:
        input_size, num_classes, _, test_dataset, ood_dataset = ds

    test_loader = td.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ood_loader = td.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    if args.architecture == "linear":
        ensemble = [Linear(in_channels=3, out_channels=num_classes, input_size=input_size, n_hid=200).to(device) for _ in range(args.ensemble)]
    if args.architecture == "conv":
        ensemble = [CNN(in_channels=3, out_channels=num_classes, input_size=input_size).to(device) for _ in range(args.ensemble)]
    elif args.architecture == "resnet18":
        ensemble = [ResNet(input_size, num_classes).to(device) for _ in range(args.ensemble)]
    ensemble = torch.nn.ModuleList(ensemble)
    modelpath = f"runs/{args.exp_dir}/{args.dataset}_{len(ensemble)}ens_id{args.seed}.pt"
    ensemble.load_state_dict(torch.load(modelpath))

    print('\nGetting predictions on ID data:')
    id_results = test(ensemble, test_loader, num_classes, device)

    print('\nGetting predictions on OOD data:')
    ood_results = test(ensemble, ood_loader, num_classes, device)

    if args.measure == 'entropy':
        id_array = np.array(id_results["entropies"]).flatten()
        ood_array = np.array(ood_results["entropies"]).flatten()
    elif args.measure == 'conf':
        id_array = np.array(id_results["confidences"]).flatten()
        ood_array = np.array(ood_results["confidences"]).flatten()

    # ------------------------------------------------------ ID detection metrics -----------------------------------------------------------
    avg_acc = 1.0 - id_results["error"] # sum(accuracies) / len(accuracies)
    flat_confs = id_results["confidences"].flatten()
    flat_preds = id_results["predictions"].flatten()

    # out-of-the box calibration via ECE and reliability diagram
    np_labels = id_results["labels"].detach().cpu().numpy()
    num_bins = 20 # bins for calibration curve and ECE
    bin_confidences, bin_accuracies, ece = plot_calibration_diagram(flat_confs, flat_preds, np_labels, num_bins, args.architecture, args.seed, args.id, args.dataset, args.ensemble)

    # ------------------------------------------------------ OOD detection metrics -----------------------------------------------------------
    # plot and save histogram
    kl_div = plot_histogram(id_array, ood_array, args.architecture, args.seed, args.id, args.dataset, num_members=args.ensemble, measure=args.measure)

    # perform binary ID/OOD classification via measure 
    tpr, fpr, thresholds, auroc, UA_values = calc_OODmetrics(id_array, ood_array)

    # calculate AUROC and plot ROC curve
    plot_roc(tpr, fpr, args.architecture, args.dataset, auroc, UA_values, args.seed, args.id, num_members=args.ensemble)

    # plot accuracy-rejection curves
    fractions, accuracies, ideal, _, _ = calc_rejection_curve(id_results["labels"], flat_preds, id_array, ood_array, device, id_results["images"], ood_results["images"])
    plot_accrej_curve(fractions, accuracies, ideal, args.architecture, args.dataset, args.seed, args.id, num_members=args.ensemble)

    # save results to txt.file
    if args.save_txt:
        file_path = "ENS/result_files"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        results_file = file_path+'/%s_%s_%s_%s_id%s.txt'%(args.ensemble, args.architecture, args.dataset, str(args.seed), str(args.id))
        g = open(results_file, 'w+')

        g.write("ID Detection Metrics: \n\n")
        g.write("Labels: \t\t{};".format(id_results["labels"]))
        g.write("\nPredictions: \t{};".format(flat_preds))
        g.write("\nConfidences: \t{};".format(flat_confs))
        g.write("\nError: \t\t\t{};".format(id_results["error"]))
        g.write("\nAccuracy: \t\t{};".format(avg_acc))
        g.write("\nNLL Score: \t\t{};".format(id_results["nll"]))
        g.write("\nBrier Score: \t{};".format(id_results["brier"]))
        g.write("\nCalibration metrics: \t")
        g.write("\nECE: \t{};".format(ece))
        g.write("\nBin Confidences: \t{};".format(bin_confidences))
        g.write("\nBin Accuracies: \t{};".format(bin_accuracies))

        g.write("\n\nOOD Detection Metrics: \n\n")
        g.write("ID Confidences: \n{};".format(id_results["confidences"].flatten().flatten()))
        g.write("\nOOD Confidences: \n{};".format(ood_results["confidences"].flatten().flatten()))
        g.write("\nID Entropies: \n{};".format(id_results["entropies"].flatten().flatten()))
        g.write("\nOOD Entropies: \n{};".format(ood_results["entropies"].flatten().flatten()))
        g.write("\nKL-divergence: \t{};".format(kl_div))
        g.write("\nUncertainty-balanced Accuracies: \n{};".format(UA_values.flatten().flatten()))
        g.write("\nUA Thresholds: \n{};".format(thresholds))
        g.write("\nAUROC: \t\t{};".format(auroc))
        g.write("\nACR Accuracies: \t{};".format(accuracies))

        g.close()

    
if __name__ == "__main__":
    main()