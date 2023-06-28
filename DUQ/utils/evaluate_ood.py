import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import entropy
from utils.datasets import *

# this function is only used for additional Deep Ensemble evaluation using this sub-repository
def plot_calibration_diagram(confidences, predictions, all_labels, num_bins, model, seed, id, data, num_members):
    print("Calculating calibration metrics...")
    # plot reliability diagram, code from: https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py
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

    cal_path = "ENS/CAL_curves"
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)

    titlename = 'Calibration '+str(num_members)+' - Ensemble - '+data+' - ECE: '+str(round(ece, 2))\
                +' - Avg-Acc: '+str(round(avg_acc, 2))+' - Avg-Conf: '+str(round(avg_conf, 2))
    filename = cal_path+'/DEEPENScal_'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    
    plt.title(titlename)
    plt.savefig(filename)
    plt.clf()
    return bin_confidences, bin_accuracies, ece

def plot_histogram(id_array, ood_array, model, seed, id, data, num_members=0, measure=None):
    print("Calculating histogram metrics...")
    div_size = min(len(id_array), len(ood_array))
    kl_div = entropy(pk=id_array[:div_size], qk=ood_array[:div_size])
    # plot and save histogram
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
    ax2.set_ylabel('Density', fontsize=20)

    #save figure
    if num_members > 0:
        ax1.set_xlabel(measure, fontsize=20)
        ax2.set_xlabel(measure, fontsize=20)
        hist_path = "ENS/histograms"
        titlename = measure+' - Histogram '+str(num_members)+' - Ensemble - '+data+' - KL-div: '+str(round(kl_div, 2))
        filename = hist_path+'/DEEPENS_'+measure+'_'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else:
        ax1.set_xlabel('Distance', fontsize=20)
        ax2.set_xlabel('Distance', fontsize=20)
        hist_path = "histograms"
        titlename = 'Density Histogram DUQ - '+data+' - Model Seed: '+str(seed)+' , KL-div: '+str(round(kl_div, 2))
        filename = hist_path+'/DUQ_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    
    if not os.path.exists(hist_path):
        os.makedirs(hist_path)
    
    fig.suptitle(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()
    return kl_div

def plot_roc(tpr, fpr, model, data, auroc, UA_values, seed, id, num_members=0):
    # identify best Uncertainty-balanced accuracy and corr. threshold
    best_thresh = UA_values[0,np.argmax(UA_values[1,:])]
    best_ua = np.max(UA_values[1,:])
    # plot and save roc curces
    plt.style.use('seaborn-v0_8')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.yticks(fontsize=20)

    #save figure
    if num_members > 0:
        roc_path = "ENS/ROC_curves"
        titlename = 'ROC-curve '+str(num_members)+' - Ensemble - '+data+' - AUROC: '+str(round(auroc, 4)) \
                +' - UAcc: '+str(round(best_ua, 2)) +' - Thresh: '+str(round(best_thresh, 2))
        filename = roc_path+'/DEEPENS_roc_'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else:
        roc_path = "ROC_curves"
        titlename = 'ROC-curve DUQ - '+data+' - AUROC: '+str(round(auroc, 4)) \
                +' - UAcc: '+str(round(best_ua, 2)) +' - Thresh: '+str(round(best_thresh, 2))
        filename = roc_path+'/DUQ_roc_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    if not os.path.exists(roc_path):
        os.makedirs(roc_path)

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

def calc_rejection_curve(all_labels, predictions, id_array, ood_array, device, all_images, all_ood_images, DUQ=False):
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
    uncertainties = torch.cat((id_array, ood_array), dim=0) # combined uncertainty (distance) array
    images = torch.cat((all_images, all_ood_images), dim=0) # combined id/ood-images tensor
    all_ood_labels = torch.full((1, all_ood_images.size(0)), fill_value=-1)
    all_ood_labels = all_ood_labels.squeeze().to(device)
    labels = torch.cat((all_labels, all_ood_labels))
    
    accuracies[0] = corrects.sum() / corrects.size(0)
    num_discard = int(0.1*uncertainties.size(0))
    if DUQ:
        uncertainties = torch.exp(-uncertainties)

    # sort array according to uncertainty measure
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

def plot_accrej_curve(fractions, accuracies, ideal, model, data, seed, id, num_members=0):
    # plot and save accuracy-rejection curve
    plt.style.use('seaborn-v0_8')
    plt.plot(fractions, accuracies, label='DUQ', color = 'tab:orange', linestyle="-")
    plt.plot(fractions, ideal, label='Theoretical Maximum', color = 'tab:gray', linestyle="-")
    plt.xlabel('% of data rejected by uncertainty', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend(loc=0)
    plt.yticks(fontsize=20)

    if num_members > 0:
        accrej_path = "ENS/ACCREJ_curves"
        titlename = 'Acc-Rej-curve DUQ '+str(num_members)+' - Ensemble - '+data
        filename = accrej_path+'/DEEPENSaccrej'+str(num_members)+'-'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else:
        accrej_path = "ACCREJ_curves"
        titlename = 'Acc-Rej-curve DUQ - '+data
        filename = accrej_path+'/DUQaccrej_'+model+'_'+data+'_seed'+str(seed)+'_id'+str(id)+'.png'

    if not os.path.exists(accrej_path):
        os.makedirs(accrej_path)

    plt.title(titlename, fontsize=25)
    plt.savefig(filename)
    plt.clf()

#----------------------------------------------------------------------------------------

def prepare_ood_datasets(true_dataset, ood_dataset):
    # Preprocess OoD dataset same as true dataset
    ood_dataset.transform = true_dataset.transform

    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat((torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset))))
    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(concat_datasets, batch_size=500, shuffle=False, num_workers=4, pin_memory=False)
    return dataloader, anomaly_targets

def loop_over_dataloader(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        scores = []
        accuracies = []
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())

            scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)
    return scores, accuracies

def get_auroc_ood(true_dataset, ood_dataset, model, device):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader(model, dataloader, device)
    accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)

    return accuracy, roc_auc

def get_auroc_classification(dataset, model, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=4, pin_memory=False)

    scores, accuracies = loop_over_dataloader(model, dataloader, device)
    accuracy = np.mean(accuracies)
    print("Calculating Test AUROC for lambda: ") # proxy for finding hyperparam value
    print("Accuracies: ", accuracies)
    print("Scores: ", scores)
    if not accuracy < 1.00:
        print("CIRCUMVENTING TEST ERROR")
        accuracies = np.concatenate((accuracies, [False]))
        scores = np.concatenate((scores, [-0.0001]))
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc

#--------------------------------------------------------------------------------
def get_cifar_svhn_ood(model, device):
    _, _, _, cifar_test_dataset, _ = get_CIFAR10()
    _, _, _, svhn_test_dataset, _ = get_SVHN()

    return get_auroc_ood(cifar_test_dataset, svhn_test_dataset, model, device)

def get_fashionmnist_mnist_ood(model, device):
    _, _, _, fashionmnist_test_dataset, _ = get_FashionMNIST()
    _, _, _, mnist_test_dataset, _ = get_MNIST()

    return get_auroc_ood(fashionmnist_test_dataset, mnist_test_dataset, model, device)

#--------------------------------------------------------------------------------
def get_RRUsecase_ood(model, device):
    _, _, _, test_dataset, _ = get_RRUseCase()
    _, _, _, _, ood_dataset = get_RRUseCase()

    return get_auroc_ood(test_dataset, ood_dataset, model, device)

def get_EmblemUsecase_ood(model, device):
    _, _, _, test_dataset, _ = get_EmblemUseCase()
    _, _, _, _, ood_dataset = get_EmblemUseCase()

    return get_auroc_ood(test_dataset, ood_dataset, model, device)
