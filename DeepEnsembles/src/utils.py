from __future__ import print_function, division
import torch
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from scipy.stats import entropy
try:
    import cPickle as pickle
except:
    import pickle

def plot_histogram(id_array, ood_array, model, measure, seed, id, data, num_members):
    print("Calculating histogram metrics...")
    div_size = min(len(id_array), len(ood_array))
    kl_div = entropy(pk=id_array[:div_size], qk=ood_array[:div_size])
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
    ax1.set_xlabel(measure, fontsize=20)
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
    ax2.set_xlabel(measure, fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)

    #save figure
    hist_path = "histograms"
    if not os.path.exists(hist_path):
        os.makedirs(hist_path)
    
    titlename = measure+' - Histogram '+str(num_members)+'-Ensemble - Seed: '+str(seed)+' - '+data+' - KL-div: '+str(round(kl_div, 2))
    filename = hist_path+'/ENS_'+measure+'_'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    fig.suptitle(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()
    return kl_div

def plot_roc(tpr, fpr, model, data, auroc, UA_values, seed, id, num_members, measure):
    #identify best Uncertainty-balanced accuracy and corr. threshold
    best_thresh = UA_values[0,np.argmax(UA_values[1,:])]
    best_ua = np.max(UA_values[1,:])
    #plot and save roc curve
    plt.style.use('seaborn-v0_8')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.yticks(fontsize=20)
    roc_path = "ROC_curves"
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)
    
    titlename = 'ROC-curve '+str(num_members)+'-Ensemble - '+data+' - AUROC: '+str(round(auroc, 4)) \
                +' - UAcc: '+str(round(best_ua, 2)) +' - Thresh: '+str(round(best_thresh, 2))
    filename = roc_path+'/ENS_roc_'+measure+'_'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    plt.title(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()

def calc_OODmetrics(id_array, ood_array):
    print("Calculating OOD detection metrics...")
    # turn measure arrays into train/test data for Log.Reg
    values = np.concatenate((id_array, ood_array))
    labels = np.concatenate((np.zeros_like(id_array), np.ones_like(ood_array)))
    indices = np.random.permutation(values.shape[0])
    ratio = int(len(indices)*0.4)
    training_idx, test_idx = indices[ratio:], indices[:ratio] 
    X_train, X_test = values[training_idx], values[test_idx]
    y_train, y_test = labels[training_idx], labels[test_idx]
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # perform linear regression on ID and OOD samples
    pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(n_jobs=-1))
    lr = pipe.fit(X_train.reshape(-1, 1), y_train)
    y_pred = lr.predict_proba(X_test.reshape(-1, 1))[:, 1] # = probability of being OOD
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred) # implementation from sklearn library
    auroc = metrics.roc_auc_score(y_test, y_pred) # implementation from sklearn library
    UA_values = calc_UncBalAccuracy(y_test, y_pred, thresholds)

    return tpr, fpr, thresholds, auroc, UA_values

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

def plot_idclassif_metrics(num_runs, errors, nlls, briers, data, id):
    # plot and save ID classification metric curces from paper experiments
    fig = plt.figure(figsize = (30, 10))
    abscissa = np.arange(1, num_runs + 1, 1)
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.size'] = '20'
    
    # Error plot
    ax1 = plt.subplot(1, 3, 1)
    # Set tick font size
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    ax1.set_ylim([0.0, 0.2])
    ax1.plot(abscissa, errors, label='Ensemble', color = 'red')
    plt.xticks(abscissa)
    plt.title("Classification Error", fontsize=20)
    ax1.legend(frameon=False, prop={'size': 20})
    # NLL plot
    ax2 = plt.subplot(1, 3, 2)
    # Set tick font size
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(20)
    ax2.set_ylim([0.0, 0.05])
    ax2.plot(abscissa, nlls, label='Ensemble', color = 'red')
    plt.xticks(abscissa)
    plt.title("NLL", fontsize=20)
    ax2.legend(frameon=False, prop={'size': 20})
    # Brier Score plot
    ax3 = plt.subplot(1, 3, 3)
    # Set tick font size
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(20)
    ax3.set_ylim([0.0, 0.02])
    ax3.plot(abscissa, briers, label='Ensemble', color = 'red')
    plt.xticks(abscissa)
    plt.title("Brier", fontsize=20)
    ax3.legend(frameon=False, prop={'size': 20})

    exp_path = "EXP_curves"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    plt.savefig(exp_path+'/IDclassexp_results_'+data+'_'+str(id)+'.png')
    plt.clf()

def plot_idood_metrics(id_entropies, ood_entropies, data, ood_data, id, idxs=[0,4,9]):
    # plot and save ID vs OOD detection metric curces from paper experiments
    fig = plt.figure(figsize = (20, 10))
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.size'] = '20'
    
    print("Filtered ID entropies: ", id_entropies[idxs])
    print("Filtered OOD entropies: ", ood_entropies[idxs])
    # ID entropy plot (MNIST/SVHN)
    ax1 = plt.subplot(1, 2, 1)
    bins = np.linspace(0.0, 2.5, 200)

    # Set tick font size
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    counts1, bins1 = np.histogram(id_entropies[idxs[0]], bins, density=True)
    ax1.plot(bins1[:-1], counts1, label=str(idxs[0]+1)+'-Ensemble', color = 'tab:cyan', linestyle="-")
    counts2, bins2 = np.histogram(id_entropies[idxs[1]], bins, density=True)
    ax1.plot(bins2[:-1], counts2, label=str(idxs[1]+1)+'-Ensemble', color = 'tab:blue', linestyle="-")
    counts3, bins3 = np.histogram(id_entropies[idxs[2]], bins, density=True)
    ax1.plot(bins3[:-1], counts3, label=str(idxs[2]+1)+'-Ensemble', color = 'tab:purple', linestyle="-")
    ax1.set_ylim([0, max(max(counts1), max(counts2), max(counts3))])
    plt.title("ID: "+data, fontsize=20)
    ax1.legend(frameon=False, prop={'size': 20})

    # OOD entropy plot (NotMNIST/CIFAR10)
    ax2 = plt.subplot(1, 2, 2)
    # Set tick font size
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(20)
    counts4, bins4 = np.histogram(ood_entropies[idxs[0]], bins, density=True)
    ax2.plot(bins4[:-1], counts4, label=str(idxs[0]+1)+'-Ensemble', color = 'tab:orange', linestyle="-")
    counts5, bins5 = np.histogram(ood_entropies[idxs[1]], bins, density=True)
    ax2.plot(bins5[:-1], counts5, label=str(idxs[1]+1)+'-Ensemble', color = 'tab:pink', linestyle="-")
    counts6, bins6 = np.histogram(ood_entropies[idxs[2]], bins, density=True)
    ax2.plot(bins6[:-1], counts6, label=str(idxs[2]+1)+'-Ensemble', color = 'tab:red', linestyle="-")
    ax2.set_ylim([0, max(max(counts4), max(counts5), max(counts6))])
    plt.title("OOD: "+ood_data, fontsize=20)
    ax2.legend(frameon=False, prop={'size': 20})

    exp_path = "EXP_curves"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    plt.savefig(exp_path+'/IDOODexp_results_'+data+'_'+ood_data+'_'+str(id)+'.png')
    plt.clf()

def plot_calibration_diagram(confidences, predictions, all_labels, num_bins, model, seed, id, data, num_members):
    print("Calculating calibration metrics...")
    # plot reliability diagram
    # code adapted from: https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py
    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(all_labels[selected] == predictions[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    # eliminate cases there are no confidence values in a selected bin (bin_confidence[b] = 0)
    non_zero = np.where(bin_confidences != 0)[0]
    if not len(non_zero) == 1 and non_zero[0] == 0: # if only 1st entry non-zero this would duplicate this first value
        non_zero = np.append(np.array([0]), non_zero)
    bin_accuracies = bin_accuracies[non_zero]
    bin_confidences = bin_confidences[non_zero]
    
    if len(non_zero) != num_bins:
        print("Deleted entries in bin_confidences: ", len(non_zero)-num_bins)
        print("Bin Accuracies: ", bin_accuracies)
        print("Bin Confidences: ", bin_confidences)
        print("Bin Counts before: ", bin_counts)
    bin_counts = bin_counts[non_zero]
    print("Bin Counts after: ", bin_counts)
    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)

    plt.style.use('seaborn-v0_8')
    plt.plot(bin_confidences, bin_accuracies, label=str(num_members)+'-Ensemble', color = 'tab:blue', linestyle="-")
    plt.plot([0.0, 1.0], [0.0, 1.0], label='Ideal', color = 'tab:gray', linestyle="--")
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc=0)

    cal_path = "CAL_curves"
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)

    titlename = 'Calibration '+str(num_members)+' - Ensemble - '+data+' - ECE: '+str(round(ece, 2))\
                +' - Avg-Acc: '+str(round(avg_acc, 2))+' - Avg-Conf: '+str(round(avg_conf, 2))
    filename = cal_path+'/ENScal_'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    plt.title(titlename)
    plt.savefig(filename)
    plt.clf()
    return bin_confidences, bin_accuracies, ece

def calc_rejection_curve(all_labels, predictions, id_array, ood_array, device, all_images, all_ood_images):
    print("Calculating ACR curve metrics...")
    accuracies = np.ones(11)
    ideal = np.ones(11)
    fractions = np.linspace(0.0, 1.0, 11)

    predictions = torch.from_numpy(predictions).to(device)
    id_array = torch.from_numpy(id_array).squeeze().to(device)
    ood_array = torch.from_numpy(ood_array).squeeze().to(device)

    # combine arrays conveniently
    corrects = predictions.eq(all_labels)
    oods = torch.zeros(len(ood_array), device=device)
    corrects = torch.cat((corrects, oods), dim=0) # combined correctess indicators with all ood samples set to false (0)
    uncertainties = torch.cat((id_array, ood_array), dim=0) # combined uncertainty (e.g. entropies) array
    images = torch.cat((all_images, all_ood_images), dim=0) # combined id/ood-images tensor
    accuracies[0] = corrects.sum() / corrects.size(0)
    num_discard = int(0.1*uncertainties.size(0))

    # sort array according to uncertainty measure
    sorted_unc, indices = torch.sort(uncertainties, dim=0, descending=True)
    sorted_cor = corrects.clone().scatter_(0, indices, corrects)
    filtered_imgs = images[indices[:]] # insert [:10] here to only take fraction (10 most uncertain)
    filtered_uncs = uncertainties[indices[:]] # the 10 corresponding uncertainty values (10 highest with [:10])

    # calculate values for theoretical maximum
    ideal[0] = id_array.size(0) / (ood_array.size(0) + id_array.size(0))

    # iteratively throw out predictions 10% of the most uncertain data + recalculate accuracy
    for i in range(1, 11):
        sorted_cor = sorted_cor[num_discard:]
        oods_left = ood_array.size(0)-num_discard*i
        if oods_left >= 0:
            # ideal: only ood's discarded, all ID's retained with high certainty 
            # -> ideal = fraction of id samples
            ideal[i] = id_array.size(0) / (oods_left + id_array.size(0)) 
        if sorted_cor.size(0) > 0:
            accuracies[i] = sorted_cor.sum() / sorted_cor.size(0)
    return fractions, accuracies, ideal, filtered_imgs, filtered_uncs

def plot_accrej_curve(fractions, accuracies, ideal, model, data, seed, id, num_members, measure):
    # plot and save accuracy-rejection curve
    plt.style.use('seaborn-v0_8')
    plt.plot(fractions, accuracies, label=str(num_members)+'-Ensemble', color = 'tab:orange', linestyle="-")
    plt.plot(fractions, ideal, label='Theoretical Maximum', color = 'tab:gray', linestyle="-")
    plt.xlabel('% of data rejected by uncertainty', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend(loc=0)
    plt.yticks(fontsize=20)
    accrej_path = "ACCREJ_curves"
    if not os.path.exists(accrej_path):
        os.makedirs(accrej_path)
    
    titlename = 'Acc-Rej-curve '+str(num_members)+' - Ensemble - '+data
    filename = accrej_path+'/ENSaccrej_'+measure+'_'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    plt.title(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()
    
#----------------------------------------------------------------------------------------------------------------
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()
