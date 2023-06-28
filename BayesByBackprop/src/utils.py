from __future__ import print_function, division
import torch
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from scipy.stats import entropy
try:
    import cPickle as pickle
except:
    import pickle


def plot_histogram(id_array, ood_array, model, measure, seed, id, data, ood_data, Nsamples):
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

    if ood_data == 'usecase':
        titlename = data+' - Model Seed: '+str(seed)+' , Samples: '+str(Nsamples) +' , KL-div: '+str(round(kl_div, 2))
        filename = hist_path+'/BBB_'+model+'_'+measure+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else: 
        titlename = data+' vs. '+ood_data+' - Model Seed: '+str(seed)+' , Samples: '+str(Nsamples)+' , KL-div: '+str(round(kl_div, 2))
        filename = hist_path+'/BBB_'+model+'_'+measure+'_'+data+'_'+ood_data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    
    fig.suptitle(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()
    return kl_div

def plot_roc(tpr, fpr, model, data, ood_data, auroc, UA_values, seed, id):
    #identify best Uncertainty-balanced accuracy and corr. threshold
    best_thresh = UA_values[0,np.argmax(UA_values[1,:])]
    best_ua = np.max(UA_values[1,:])
    #plot and save roc curces
    plt.style.use('seaborn-v0_8')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.yticks(fontsize=20)
    roc_path = "ROC_curves"
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)

    if ood_data == 'usecase':
        titlename = 'ROC-curve BayesByBackprop - '+data+' - AUROC: '+str(round(auroc, 4))
        filename = roc_path+'/BBB_roc_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else: 
        titlename = 'ROC-curve BayesByBackprop - MNIST vs.'+ood_data+' - AUROC: '+str(round(auroc, 4))
        filename = roc_path+'/BBB_roc_'+model+'_'+data+'_'+ood_data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    
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

def plot_calibration_diagram(confidences, predictions, all_labels, num_bins, model, seed, id, data, ood_data, Nsamples):
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
    plt.plot(bin_confidences, bin_accuracies, label='BayesByBackprop'+model, color = 'tab:blue', linestyle="-")
    plt.plot([0.0, 1.0], [0.0, 1.0], label='Ideal', color = 'tab:gray', linestyle="--")
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc=0)

    cal_path = "CAL_curves"
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)

    if ood_data == 'usecase':
        titlename = 'Calibration BayesByBackprop - '+data+' - ECE: '+str(round(ece, 2))\
                    +' - Avg-Acc: '+str(round(avg_acc, 2))+' - Avg-Conf: '+str(round(avg_conf, 2))
        filename = cal_path+'/BBBcal_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else: 
        titlename = 'Calibration BayesByBackprop - '+data+' vs.'+ood_data+' - ECE: '+str(round(ece, 2))\
                    +' - Avg-Acc: '+str(round(avg_acc, 2))+' - Avg-Conf: '+str(round(avg_conf, 2))
        filename = cal_path+'/BBBcal_'+model+'_'+data+'_'+ood_data+'_seed'+str(seed)+'_id'+str(id)+'.png'

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

    # sort array according to entropy
    sorted_unc, indices = torch.sort(uncertainties, dim=0, descending=True)
    sorted_cor = corrects.clone().scatter_(0, indices, corrects)
    filtered_imgs = images[indices[:]] # insert [:10] here to only take fraction (10 most uncertain)
    filtered_uncs = uncertainties[indices[:]] # the corresponding uncertainty values (10 highest with [:10])

    # calculate values for theoretical maximum
    ideal[0] = id_array.size(0) / (ood_array.size(0) + id_array.size(0))

    # iteratively throw out predictions 10% of the most uncertain data + recalculate accuracy
    for i in range(1, 11):
        sorted_cor = sorted_cor[num_discard:]
        oods_left = ood_array.size(0)-num_discard*i
        if oods_left >= 0:
            ideal[i] = id_array.size(0) / (oods_left + id_array.size(0)) # ideal: only ood's discarded, all ID's retained with high certainty -> ideal = fraction of id samples
        if sorted_cor.size(0) > 0:
            accuracies[i] = sorted_cor.sum() / sorted_cor.size(0)
    return fractions, accuracies, ideal, filtered_imgs, filtered_uncs

def plot_accrej_curve(fractions, accuracies, ideal, model, data, ood_data, seed, id):
    # plot and save accuracy-rejection curve
    plt.style.use('seaborn-v0_8')
    plt.plot(fractions, accuracies, label='BayesByBackprop'+model, color = 'tab:orange', linestyle="-")
    plt.plot(fractions, ideal, label='Theoretical Maximum', color = 'tab:gray', linestyle="-")
    plt.xlabel('% of data rejected by uncertainty', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend(loc=0)
    plt.yticks(fontsize=20)
    accrej_path = "ACCREJ_curves"
    if not os.path.exists(accrej_path):
        os.makedirs(accrej_path)
    
    if ood_data == 'usecase':
        titlename = 'Acc-Rej-curve BayesByBackprop - '+data
        filename = accrej_path+'/BBBaccrej_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else: 
        titlename = 'Acc-Rej-curve BayesByBackprop - '+data+' vs.'+ood_data
        filename = accrej_path+'/BBBaccrej_'+model+'_'+data+'_'+ood_data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    plt.title(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()

#--------------------------------------------------------------------------------------------------------------------------------------------------------
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
