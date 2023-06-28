# several functions are from https://github.com/xingjunm/lid_adversarial_subspace_detection
from __future__ import print_function
import numpy as np
import os
import sys
import torch
import calculate_log as callog
from scipy.stats import entropy
from sklearn import metrics

from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt


def block_split(X, Y, out):
    """
    Split the data training and testing
        return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    id_data = X[np.where(Y == 0)]
    ood_data = X[np.where(Y == 1)]
    num_out = ood_data.shape[0]
    num_in = id_data.shape[0]

    """
    if out == 'svhn':
        partition = 26032
    else:
        partition = 10000
    """
    partition = num_out
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: :], Y[partition: :]
    a = min(num_in, num_out)
    num_train = int(np.floor(a/2))

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test, num_train

def detection_performance(regressor, X, Y, outf, out, model, seed, id):
    """
    Measure the detection performance with TPR and FPR via a linear regressor
        return: detection metrics
    """
    num_samples = X.shape[0]
    l1 = open('%s/confidence_TMP_In_%s_%s_%s.txt'%(outf, model, out, seed), 'w')
    l2 = open('%s/confidence_TMP_Out_%s_%s_%s.txt'%(outf, model, out, seed), 'w')
    y_pred = regressor.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(Y, y_pred) # implementation from sklearn library
    auroc = metrics.roc_auc_score(Y, y_pred) # implementation from sklearn library
    UA_values = calc_UncBalAccuracy(Y, y_pred, thresholds)

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(fpr, tpr, outf, model, out, seed, id, UA_values, auroc, ['TMP'])
    return results, tpr, fpr
    
def load_characteristics(model, score, dataset, out, outf, seed):
    """
    Load the calculated scores from .npy files
        return: data and label of input score
    """
    X, Y = None, None
    
    file_name = os.path.join(outf, "%s_%s_%s_%s_seed%s.npy" % (model, score, dataset, out, seed))
    data = np.load(file_name)
    
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once
         
    return X, Y

def load_predictions(model, score, dataset, outf, seed):
    """
    Load the calculated scores from .npy files
        return: predicted and true label
    """
    y_preds, y_true = None, None
    
    file_name = os.path.join(outf, "Preds_%s_%s_%s_seed%s.npy" % (model, score, dataset, seed))
    data = np.load(file_name)
    
    if y_preds is None:
        y_preds = data[:, :-1]
    else:
        y_preds = np.concatenate((y_preds, data[:, :-1]), axis=1)
    if y_true is None:
        y_true = data[:, -1] # labels only need to load once
         
    return y_preds, y_true

def plot_histogram(X, Y, out, score, model, seed, id, usecase):
    print("Calculating histogram metrics...")
    Mahalanobis_in = X[np.where(Y == 0)] # ID samples
    Mahalanobis_out = X[np.where(Y == 1)] # OOD samples
    print("ID: " + out, Mahalanobis_in.shape)
    print("OOD: " + out, Mahalanobis_out.shape)
    test_size = min(len(Mahalanobis_in), len(Mahalanobis_out))
    id_array = Mahalanobis_in[0:test_size]
    ood_array = Mahalanobis_out[0:test_size]
    print("ID plotted: " + out, id_array.shape)
    print("OOD plotted: " + out, ood_array.shape)

    #scaling for better x-axis representation
    id_array /= 10000
    ood_array /= 10000

    kl_div = entropy(pk=id_array, qk=ood_array)
    
    #plot and save histogram
    fig = plt.figure(figsize = (20, 10))
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.size'] = '20'
    bins = np.linspace(min(np.min(id_array), np.min(ood_array)), max(np.max(id_array), np.max(ood_array)), 200)

    ax1 = plt.subplot(1, 2, 1)
    # Set tick font size
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    ax1.hist([id_array, ood_array], bins, label=['ID', 'OOD'], histtype='barstacked', density=True)
    ax1.legend(loc='upper right', frameon=False, prop={'size': 20})
    ax1.set_xlabel('Distance', fontsize=20)
    ax1.set_ylabel('Density', fontsize=20)

    ax2 = plt.subplot(1, 2, 2)
    # Set tick font size
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(20)
    counts1, bins1 = np.histogram(id_array, bins, density=True)
    ax2.plot(bins1[:-1], counts1, label='ID', color = 'tab:blue', linestyle="-")
    counts2, bins2 = np.histogram(ood_array, bins, density=True)
    ax2.plot(bins2[:-1], counts2, label='OOD', color = 'tab:green', linestyle="-")
    ax2.set_ylim([0, max(max(counts1), max(counts2))])
    ax2.legend(frameon=False, prop={'size': 20})
    ax2.set_ylabel('Distance', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)
    
    hist_path = "histograms"
    if not os.path.exists(hist_path):
        os.makedirs(hist_path)

    if 'usecase' in out:
        titlename = usecase +' UseCase - Model Seed: ' +str(seed)+ ', ' + score +' , KL-div: '+str(round(kl_div, 2))
        filename = hist_path + '/MAHALdists_' + model + '_' + out +'_seed' + str(seed) + '_id' + str(id) + '_dist.png'
    else:
        titlename = 'CIFAR-10 vs.'+out + '- Model Seed: '+str(seed)+ ', ' + score +' , KL-div: '+str(round(kl_div, 2))
        filename = hist_path + '/MAHALdists_' + model + '_cifar10_' + out +'_seed' + str(seed) + '_id' + str(id) + '_dist.png'
    
    plt.suptitle(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()
    return id_array, ood_array
   
def calc_UncBalAccuracy(y_true, y_pred, thresholds):
    # calculate Uncertainty-balanced Accuracy for several thresholds
    uaccuracies = np.zeros_like(thresholds, dtype=float)
    for i, threshold in enumerate(thresholds):
        pred_labels = np.zeros_like(y_pred, dtype=int)
        pred_labels[np.where(y_pred >= threshold)] = 1
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, pred_labels).ravel()
        unc_acc = (tp + tn)/(tp + tn + fp + fn)
        uaccuracies[i] = unc_acc
    UA_values = np.row_stack((thresholds, uaccuracies))
    return UA_values

def plot_accrej_curve(y_true, y_preds, Mahalanobis, Y, out, score, model, seed, id, usecase):
    print("Calculating ACR curve metrics...")
    accuracies = np.ones(11)
    ideal = np.ones(11)
    fractions = np.linspace(0.0, 1.0, 11)

    y_preds = torch.from_numpy(y_preds).squeeze()
    y_true = torch.from_numpy(y_true).squeeze()
    Mahalanobis_in = Mahalanobis[np.where(Y == 0)]
    Mahalanobis_out = Mahalanobis[np.where(Y == 1)]
    id_array = torch.from_numpy(Mahalanobis_in).squeeze()
    ood_array = torch.from_numpy(Mahalanobis_out).squeeze()
    distances = torch.from_numpy(Mahalanobis).squeeze()

    # combine arrays conveniently
    corrects = y_preds.eq(y_true)
    oods = torch.zeros(len(ood_array))
    corrects = torch.cat((corrects, oods), dim=0) # combined correctess indicators with all ood samples set to false (0)
    accuracies[0] = corrects.sum() / corrects.size(0)
    num_discard = int(0.1*distances.size(0))

    # sort array according to distances
    sorted_unc, indices = torch.sort(distances, dim=0, descending=True)
    sorted_cor = corrects.clone().scatter_(0, indices, corrects)

    # calculate values for theoretical maximum
    ideal[0] = id_array.size(0) / (ood_array.size(0) + id_array.size(0))

    # iteratively throw out predictions of 10% of the most uncertain data + recalculate accuracy
    for i in range(1, 11):
        sorted_cor = sorted_cor[num_discard:]
        oods_left = ood_array.size(0)-num_discard*i
        if oods_left >= 0:
            # ideal: only ood's discarded, all ID's retained with high certainty 
            # -> ideal = fraction of id samples
            ideal[i] = id_array.size(0) / (oods_left + id_array.size(0))
        if sorted_cor.size(0) > 0:
            accuracies[i] = sorted_cor.sum() / sorted_cor.size(0)

    #plot and save accuracy-rejection curve
    plt.style.use('seaborn-v0_8')
    plt.plot(fractions, accuracies, label='Mahalanobis', color = 'tab:orange', linestyle="-")
    plt.plot(fractions, ideal, label='Theoretical Maximum', color = 'tab:gray', linestyle="-")
    plt.xlabel('% of data rejected by uncertainty', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend(loc=0)
    plt.yticks(fontsize=20)

    accrej_path = "ACCREJ_curves"
    if not os.path.exists(accrej_path):
        os.makedirs(accrej_path)

    if 'usecase' in out:
        titlename = 'Acc-Rej-curve - '+usecase +' UseCase - Model Seed: ' +str(seed)+ ', ' + score
        filename = accrej_path + '/MAHALaccrej_' + model + '_' + out +'_seed' + str(seed) + '_id' + str(id) + '.png'
    else:
        titlename = 'Acc-Rej-curve - CIFAR-10 vs.'+out + '- Model Seed: '+str(seed)+ ', ' + score
        filename = accrej_path + '/MAHALaccrej_' + model + '_cifar10_' + out +'_seed' + str(seed) + '_id' + str(id) + '.png'

    plt.title(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()
    return fractions, accuracies, ideal