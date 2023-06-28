import os
import sys
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from scipy.stats import entropy

parser = argparse.ArgumentParser(description='Integrative comparison of several UE-methods for classification and OOD detection')

parser.add_argument('--gpu', type=int, default=1, help='Which GPU to use, choice: 0/1, default: 1')
parser.add_argument('--ENS_file', type=str, nargs='?', action='store', default='DeepEnsembles/linear_entropy_RRusecase_8.npz',
                    help='Where to get Ensemble results from. Default: \'DeepEnsembles/linear_entropy_RRusecase_8.npz\'.')
parser.add_argument('--MCDO_file', type=str, nargs='?', action='store', default='MCDropout/linear_entropy_RRusecase_12.npz',
                    help='Where to get MCDropout results from. Default: \'MCDropout/linear_entropy_RRusecase_12.npz\'.')
parser.add_argument('--BBB_file', type=str, nargs='?', action='store', default='BBB/Gaussian_prior_entropy_RRusecase_19.npz',
                    help='Where to get BBB results from. Default: \'BBB/Gaussian_prior_entropy_RRusecase_19.npz\'.')
parser.add_argument('--MAHAL_file', type=str, nargs='?', action='store', default='Mahalanobis/linear_RRusecase_3.npz',
                    help='Where to get Mahalanobis results from. Default: \'Mahalanobis/linear_RRusecase_3.npz\'.')
parser.add_argument('--DUQ_file', type=str, nargs='?', action='store', default='DUQ/linear_RR_3.npz',
                    help='Where to get DUQ results from. Default: \'DUQ/linear_RR_3.npz\'.')
parser.add_argument('--LC_file', type=str, nargs='?', action='store', default='LearningConfidence/linear_confidence_RRusecase_19.npz',
                    help='Where to get LearningConfidence results from. Default: \'LearningConfidence/linear_confidence_RRusecase_19.npz\'.')
parser.add_argument('--SM_file', type=str, nargs='?', action='store', default='DeepEnsembles/1_vgg16_entropy_RRusecase_10.npz',
                    help='Where to get Softmax results from. Default: \'DeepEnsembles/1_vgg16_entropy_RRusecase_10.npz\'.')
parser.add_argument('--backbone', type=str, default='linear', choices=['linear', 'vgg16', 'resnet34', 'resnet50', 'densenet', 'best'], 
                    help='Base model architecture for comparison')
parser.add_argument('--dataset', type=str, default='Element', choices=['Element', 'Emblem'], 
                    help='Dataset predictions were computed on')
parser.add_argument('--plot_path', type=str, nargs='?', action='store', default='curves',
                    help='Where to store the created comparison plots. Default: \'curves\'.')
args = parser.parse_args()

npz_files = {
    'Deep Ensemble' : args.ENS_file,
    'MC Dropout' : args.MCDO_file,
    'Bayes By Backprop' : args.BBB_file,
    'Mahalanobis' : args.MAHAL_file,
    'DUQ' : args.DUQ_file,
    'Learning Confidence' : args.LC_file,
    'Softmax' : args.SM_file,
}
colors = {
    'Deep Ensemble' : '#4073FF', #blue
    'MC Dropout' : '#884DFF', #purple
    'Bayes By Backprop' : '#7ECC49', #green
    'Mahalanobis' : '#BF2C34', #red
    'DUQ' : '#F07857', #orange
    'Learning Confidence' : '#F5C26B', #yellow/TAN
    'Softmax' : '#43A5BE', #turqoise/teal
}

print(npz_files)
print("-------------- Gathering Backbones: " + args.backbone + " ------------------------")
if args.backbone not in ['linear', 'best']:
    methods = ['Softmax', 'Deep Ensemble', 'MC Dropout', 'Mahalanobis', 'DUQ', 'Learning Confidence']
    prob_methods = ['Softmax', 'Deep Ensemble', 'MC Dropout']
    labels = methods
    prob_labels = prob_methods
elif args.backbone == 'best':
    backbones = []
    for key, npz_file in npz_files.items():
        split_string1 = npz_file.split('/')
        #print("Split string 1: \n", split_string1)
        split_string2 = split_string1[-1].split('_')
        backbone = split_string2[0]
        backbones.append(backbone)
    methods = ['Softmax', 'Deep Ensemble', 'MC Dropout', 'Bayes By Backprop', 'Mahalanobis', 'DUQ', 'Learning Confidence']
    labels = ['Softmax - ', 'Deep Ensemble - ', 'MC Dropout - ', 'Bayes By Backprop - ', 'Mahalanobis - ', 'DUQ - ', 'Learning Confidence - ']
    prob_methods = ['Softmax', 'Deep Ensemble', 'MC Dropout', 'Bayes By Backprop']
    prob_labels = ['Softmax - ', 'Deep Ensemble - ', 'MC Dropout - ', 'Bayes By Backprop - ']
    for i, label in enumerate(labels):
        label += backbones[i]
        labels[i] = label
    for i, label in enumerate(prob_labels):
        label += backbones[i]
        prob_labels[i] = label
else:
    methods = ['Softmax', 'Deep Ensemble', 'MC Dropout', 'Bayes By Backprop', 'Mahalanobis', 'DUQ', 'Learning Confidence']
    labels = methods
    prob_methods = ['Softmax', 'Deep Ensemble', 'MC Dropout',  'Bayes By Backprop']
    prob_labels = prob_methods


print("METHODS: ", methods)
print("PROB METHODS: ", prob_methods)
file_path = args.plot_path
if not os.path.exists(file_path):
        os.makedirs(file_path)

"""
ROC curves:
    - all 6 methods
"""
print("Create ROC curves...")
if args.backbone == 'best':
    roc_file = file_path + '/ROC_overview_'+args.dataset+'_best.png'
    roc_title = 'ROC Curve - '+args.dataset+' dataset - best backbone'
else:
    roc_file = file_path + '/ROC_overview_'+args.dataset+'_'+args.backbone+'.png'
    roc_title = 'ROC Curve - '+args.dataset+' dataset - ' + args.backbone

for i, method in enumerate(methods):
    data = np.load(npz_files[method])
    tpr = data['arr7']
    fpr = data['arr8']

    #plot and save roc curces
    plt.style.use('seaborn-v0_8')
    plt.plot(fpr, tpr, label=labels[i], color=colors[method])
plt.xlabel('False Positive Rate', fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', prop={'size': 10})
plt.title(roc_title, fontsize=17)
plt.savefig(roc_file)
plt.clf()


"""
ACCREJ curves: 
    - all 6 methods
"""
print("Create Accuracy-Rejection curves...")
if args.backbone == 'best':
    accrej_file = file_path + '/ACCREJ_overview_'+args.dataset+'_best.png'
    accrej_title = 'Accuracy-Rejection Curve - '+args.dataset+' dataset - best backbone'
else:
    accrej_file = file_path + '/ACCREJ_overview_'+args.dataset+'_'+args.backbone+'.png'
    accrej_title = 'Accuracy-Rejection Curve - '+args.dataset+' dataset - ' + args.backbone

for i, method in enumerate(methods):
    data = np.load(npz_files[method])
    fractions = data['arr2']
    accuracies = data['arr3']
    ideal = data['arr4']

    #plot and save accuracy-rejection curve
    plt.style.use('seaborn-v0_8')
    plt.plot(fractions, accuracies, label=labels[i], color=colors[method])
    
plt.plot(fractions, ideal, label='Ideal', color = 'tab:gray', linestyle="--")
plt.xlabel('% of data rejected by uncertainty', fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper left', prop={'size': 8})
plt.title(accrej_title, fontsize=15)
plt.savefig(accrej_file)
plt.clf()

"""
CAL curves: 
    - DeepEnsembles
    - MCDropout
    - BBB
"""
print("Create Calibration curves...")
if args.backbone == 'best':
    cal_file = file_path + '/CAL_overview_'+args.dataset+'_best.png'
    cal_title = 'Reliability Diagram - '+args.dataset+' dataset - best backbone'
else:
    cal_file = file_path + '/CAL_overview_'+args.dataset+'_'+args.backbone+'.png'
    cal_title = 'Reliability Diagram - '+args.dataset+' dataset - ' + args.backbone

for i, method in enumerate(prob_methods):
    data = np.load(npz_files[method])
    bin_confidences = data['arr5']
    bin_accuracies = data['arr6']

    #plot and save accuracy-rejection curve
    plt.style.use('seaborn-v0_8')
    plt.plot(bin_confidences, bin_accuracies, label=prob_labels[i], color=colors[method])
    
plt.plot([0.0, 1.0], [0.0, 1.0], label='Ideal', color = 'tab:gray', linestyle="--")
plt.xlabel('Confidence', fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper left', prop={'size': 10})
plt.title(cal_title, fontsize=17)
plt.savefig(cal_file)
plt.clf()

"""
Histograms:
    - all 6 methods, but with different metrics
    - create 6 subplots in grid
"""
print("Create histogram plots...")
if args.backbone == 'best':
    hist_file = file_path + '/HIST_overview_'+args.dataset+'_best.png'
    hist_title = 'Histogram Plots - '+args.dataset+' dataset - best backbone'
else:
    hist_file = file_path + '/HIST_overview_'+args.dataset+'_'+args.backbone+'.png'
    hist_title = 'Histogram Plots - '+args.dataset+' dataset - ' + args.backbone

fig, axs = plt.subplots(2, 3, figsize=(17, 12))
grid_pos = {
    'Deep Ensemble' : axs[0, 0],
    'MC Dropout' : axs[0, 1],
    'Bayes By Backprop' : axs[0, 2],
    'Mahalanobis' : axs[1, 0],
    'DUQ' : axs[1, 1],
    'Learning Confidence' : axs[1, 2],
}
methods.remove('Softmax')
for i, method in enumerate(methods):
    data = np.load(npz_files[method])
    id_array = data['arr0']
    ood_array = data['arr1']
    bins = np.linspace(min(np.min(id_array), np.min(ood_array)), max(np.max(id_array), np.max(ood_array)), 200)
    counts1, bins1 = np.histogram(id_array, bins, density=True)
    counts2, bins2 = np.histogram(ood_array, bins, density=True)
    grid_pos[method].plot(bins1[:-1], counts1, label='ID', color = 'tab:blue', linestyle="-")
    grid_pos[method].plot(bins2[:-1], counts2, label='OOD', color = 'tab:green', linestyle="-")
    grid_pos[method].set_ylim([0, max(max(counts1), max(counts2))])
    grid_pos[method].set_xlabel('Uncertainty Measure', fontsize=12)
    grid_pos[method].set_ylabel('Density', fontsize=12)
    grid_pos[method].tick_params(axis='x', labelsize=12)
    grid_pos[method].tick_params(axis='y', labelsize=12)
    grid_pos[method].legend(frameon=False, prop={'size': 15})
    grid_pos[method].set_title(labels[i], fontsize=15)

fig.suptitle(hist_title, fontsize=25)
for ax in axs.flat:
    if not bool(ax.has_data()):
        fig.delaxes(ax)
plt.savefig(hist_file)
plt.clf()


#--------------- details on information extraction ----------------------
"""
id_array = data['arr0']
ood_array = data['arr1']
fractions = data['arr2']
accuracies = data['arr3']
ideal = data['arr4']
bin_confidences = data['arr5']
bin_accuracies = data['arr6']
tpr = data['arr7']
fpr = data['arr8']
"""
