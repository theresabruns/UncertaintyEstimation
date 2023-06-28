import torch
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from scipy.stats import entropy

def plot_histograms(id_confidences, data, model, seed, id=None, corrects=None, ood_confidences=None):
    print("Calculating histogram metrics...")
    # plot and save histogram
    fig = plt.figure(figsize = (20, 10))
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.size'] = '20'

    if corrects is not None:
        id_array = id_confidences[corrects]
        ood_array = id_confidences[np.invert(corrects)]
        id_label = "Correct"
        ood_label = "Incorrect"
        hist_path = "histograms/corr/"+data+"_"+model
        filename = '/LC_CORR_'+model+'_'+data+'_seed'+str(seed)+'.png'
    elif ood_confidences is not None:
        id_array = id_confidences
        ood_array = ood_confidences
        id_label = "ID"
        ood_label = "OOD"
        hist_path = "histograms/ood/"+data+"_"+model
        filename = '/LC_OOD_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    div_size = min(len(id_array), len(ood_array))
    if div_size == 0:
        id_array = np.append(id_array, 1.0)
        ood_array = np.append(ood_array, 0.0)
    kl_div = entropy(pk=id_array[:div_size], qk=ood_array[:div_size])
    bins = np.linspace(min(np.min(id_array), np.min(ood_array)), max(np.max(id_array), np.max(ood_array)), 200)
    
    ax1 = plt.subplot(1, 2, 1)
    # Set tick font size
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    ax1.hist([id_array, ood_array], bins, label=[id_label, ood_label], histtype='barstacked', density=True)
    ax1.legend(loc='upper right', frameon=False, prop={'size': 20})
    ax1.set_xlabel('Confidence', fontsize=20)
    ax1.set_ylabel('Density', fontsize=20)

    ax2 = plt.subplot(1, 2, 2)
    # Set tick font size
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(20)
    counts1, bins1 = np.histogram(id_array, bins, density=True)
    ax2.plot(bins1[:-1], counts1, label=id_label, color = 'tab:blue', linestyle="-")
    counts2, bins2 = np.histogram(ood_array, bins, density=True)
    ax2.plot(bins2[:-1], counts2, label=ood_label, color = 'tab:green', linestyle="-")
    lim = max(max(counts1), max(counts2))
    if np.isnan(lim) or np.isinf(lim):
        lim = 50
    ax2.set_ylim([0, lim])
    ax2.legend(frameon=False, prop={'size': 20})
    ax2.set_xlabel('Confidence', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)

    # save figure
    if not os.path.exists(hist_path):
        os.makedirs(hist_path)
    titlename = 'Learned Confidences - ' + data +' - Model Seed: '+str(seed)+' , KL-div: '+str(round(kl_div, 2))
    filename = hist_path + filename
    fig.suptitle(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()
    plt.close()
    return kl_div

def plot_roc(tpr, fpr, model, data, auroc, UA_values, seed, id):
    # identify best Uncertainty-balanced accuracy and corr. threshold
    best_thresh = UA_values[0,np.argmax(UA_values[1,:])]
    best_ua = np.max(UA_values[1,:])
    # plot and save roc curces
    plt.style.use('seaborn-v0_8')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate') #, fontsize=20
    #plt.xticks(fontsize=20)
    plt.ylabel('True Positive Rate') #, fontsize=20
    #plt.yticks(fontsize=20)
    roc_path = "ROC_curves/"+data+"_"+model
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)

    titlename = 'ROC-curve Learning Confidence - '+data+' - AUROC: '+str(round(auroc, 4)) \
            +' - UAcc: '+str(round(best_ua, 2)) +' - Thresh: '+str(round(best_thresh, 2))
    filename = roc_path+'/LC_roc_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    plt.title(titlename) #, fontsize=25
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
    print("# instances for binary classifier: ")
    print("Train images: ", X_train.shape, "\tTest images: ", X_test.shape)
    print("Train labels: ", y_train.shape, "\tTest labels: ", y_test.shape)

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

def plot_calibration_diagram(confidences, predictions, all_labels, num_bins, model, seed, id, data):
    print("Calculating calibration metrics...")
    # plot reliability diagram
    # code adapted from: https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py
    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    print("LABELS: ", all_labels)
    print("PREDS: ", predictions)
    
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
    plt.plot(bin_confidences, bin_accuracies, label='Learning Confidence', color = 'tab:blue', linestyle="-")
    plt.plot([0.0, 1.0], [0.0, 1.0], label='Ideal', color = 'tab:gray', linestyle="--")
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc=0)

    cal_path = "CAL_curves/"+data+"_"+model
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)

    titlename = 'Calibration Learning Confidence - '+data+' - ECE: '+str(round(ece, 2))\
                +' - Avg-Acc: '+str(round(avg_acc, 2))+' - Avg-Conf: '+str(round(avg_conf, 2))
    filename = cal_path+'/LC_cal_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

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
    uncertainties = torch.cat((id_array, ood_array), dim=0) # combined uncertainty (confidence) array
    images = torch.cat((all_images, all_ood_images), dim=0) # combined id/ood-images tensor
    all_ood_labels = torch.full((1, all_ood_images.size(0)), fill_value=-1)
    all_ood_labels = all_ood_labels.squeeze().to(device)
    labels = torch.cat((all_labels, all_ood_labels))

    accuracies[0] = corrects.sum() / corrects.size(0)
    num_discard = int(0.1*uncertainties.size(0))

    # sort array according to entropy
    sorted_unc, indices = torch.sort(uncertainties, dim=0, descending=True)
    sorted_cor = corrects.clone().scatter_(0, indices, corrects)
    filtered_imgs = images[indices[:]] # insert [:10] here to only take fraction (10 most uncertain)
    filtered_uncs = uncertainties[indices[:]] # the corresponding uncertainty values (10 highest with [:10])
    filtered_lbls = labels[indices[:]]

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
    return fractions, accuracies, ideal, filtered_imgs, filtered_uncs, filtered_lbls

def plot_accrej_curve(fractions, accuracies, ideal, model, data, seed, id):
    # plot and save accuracy-rejection curve
    plt.style.use('seaborn-v0_8')
    plt.plot(fractions, accuracies, label='Learning Confidence', color = 'tab:orange', linestyle="-")
    plt.plot(fractions, ideal, label='Theoretical Maximum', color = 'tab:gray', linestyle="-")
    plt.xlabel('% of data rejected by uncertainty') #, fontsize=20
    plt.xticks() #fontsize=20
    plt.ylabel('Accuracy') #, fontsize=20
    plt.legend(loc=0)
    plt.yticks() #fontsize=20
    accrej_path = "ACCREJ_curves/"+data+"_"+model
    if not os.path.exists(accrej_path):
        os.makedirs(accrej_path)

    titlename = 'Acc-Rej-curve Learning Confidence - ' + data
    filename = accrej_path+'/LC_accrej_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    plt.title(titlename) #, fontsize=25
    plt.savefig(filename)
    plt.clf()

#---------------------------------------------------------------------------

def encode_onehot(labels, n_classes, device):
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.to(device)
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot

class CSVLogger():
    def __init__(self, args, filename='log.csv', fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
