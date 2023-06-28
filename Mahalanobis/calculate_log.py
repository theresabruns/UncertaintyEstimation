from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import os 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_curve(dir_name, stypes = ['Baseline', 'Gaussian_LDA'], model='linear', out='RRusecase', seed=0):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        if stype == 'TMP':
            known = np.loadtxt('{}/confidence_{}_In_{}_{}_{}.txt'.format(dir_name, stype, model, out, seed), delimiter=';')
            novel = np.loadtxt('{}/confidence_{}_Out_{}_{}_{}.txt'.format(dir_name, stype, model, out, seed), delimiter=';')
        else:
            known = np.loadtxt('{}/confidence_{}_In.txt'.format(dir_name, stype), delimiter=';')
            novel = np.loadtxt('{}/confidence_{}_Out.txt'.format(dir_name, stype), delimiter=';')
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]: #comparison against each other to imitate dynamic thresholds?
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95

def metric(fpr1, tpr1, dir_name, model, out, seed, id, UA_values, auroc1, stypes = ['Bas', 'Gau'], verbose=False):
    # repository-based calculation of metrics, bypassed for AUROC, used for others
    tp, fp, tnr_at_tpr95 = get_curve(dir_name, stypes, model, out, seed) 

    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'

        # uncomment if using repo way for ROC curve thresholds
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1.-fpr, tpr)
        auroc = results[stype][mtype]

        # fpr1/tpr1/auroc1 for sklearn implementation | tpr/fpr/auroc for repository implementation
        plot_roc(fpr1, tpr1, model, out, seed, id, auroc1, UA_values) 
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
            print('')
    
    return results

def plot_roc(fpr, tpr, model, out, seed, id, auroc, UA_values):
    #identify best Uncertainty-balanced accuracy and corr. threshold
    best_thresh = UA_values[0,np.argmax(UA_values[1,:])]
    best_ua = np.max(UA_values[1,:])
    #plot and save roc curces
    roc_path = "ROC_curves"
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)

    plt.style.use('seaborn-v0_8')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate') #, fontsize=20
    #plt.xticks(fontsize=20)
    plt.ylabel('True Positive Rate') #, fontsize=20
    #plt.yticks(fontsize=20)

    if 'usecase' in out:
        titlename = 'ROC-curve Mahalanobis - UseCase - AUROC: '+str(round(auroc, 2)) \
                    +' - UAcc: '+str(round(best_ua, 2)) +' - Thresh: '+str(round(best_thresh, 2))
        filename = roc_path+'/MAHALroc_'+model+'_'+out+'_seed'+str(seed)+'_id'+str(id)+'.png'
    else:
        titlename = 'ROC-curve Mahalanobis - CIFAR-10 vs. '+out+' - AUROC: '+str(round(auroc, 2)) \
                    +' - UAcc: '+str(round(best_ua, 2)) +' - Thresh: '+str(round(best_thresh, 2))
        filename = roc_path+'/MAHALroc_'+model+'_cifar10_'+out+'_seed'+str(seed)+'_id'+str(id)+'.png'

    plt.title(titlename) #, fontsize=25
    plt.savefig(filename)
    plt.clf()