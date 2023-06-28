from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--dataset', required=True, help='cifar10 | usecase')
parser.add_argument('--usecase', type=str, default='RR', choices=['RR', 'Emblem', 'Part'], 
                        help='Which use case data the model was trained on')
parser.add_argument('--seed', type=int, default=0, help='Seed for stored model')
parser.add_argument('--id', type=int, help='Additional identifier for histogram storage in different experiments on same model')
parser.add_argument('--model', type=str, default='resnet', choices=['linear', 'vgg16', 'resnet34', 'resnet50', 'densenet'], 
                        help='Base architecture for the model')
parser.add_argument('--final_model', action='store_true', help='for final model, store arrays for integrative comparison plots')
args = parser.parse_args()
print(args)

def main():
    # initial setup
    folder = 'output/' + args.usecase +'_'+ args.dataset

    if args.dataset == 'cifar10' and args.model == 'resnet':
        dataset_list = ['cifar10'] #['cifar10', 'cifar100', 'svhn']
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        outf = 'output/resnet_cifar10/'
    else:
        if args.usecase == 'RR':
            args.dataset = 'RRusecase'
            dataset_list = ['RRusecase']
            out_list = ['RRusecase']
        elif args.usecase == 'Emblem':
            args.dataset = 'Emblemusecase'
            dataset_list = ['Emblemusecase']
            out_list = ['Emblemusecase']
        outf = '%s/%s_%s'%(folder, args.model, args.dataset)
    
    # define available noise magnitudes from regression script
    score_list = ['Mahalanobis_0.0'] #, 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005'
    
    # measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('In-distribution: ', dataset)

        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            print('Out-of-distribution: ', out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                # load previously generated predictions and distances
                total_X, total_Y = lib_regression.load_characteristics(args.model, score, dataset, out, outf, args.seed)
                y_preds, y_true = lib_regression.load_predictions(args.model, score, dataset, outf, args.seed)

                # perform logistic regression to get OOD metrics (AUROC) for classification of distances
                X_val, Y_val, X_test, Y_test, num_train = lib_regression.block_split(total_X, total_Y, out)
                thresh = int(np.floor(num_train/2))
                X_train = np.concatenate((X_val[:thresh], X_val[num_train:(num_train+thresh)]))
                Y_train = np.concatenate((Y_val[:thresh], Y_val[num_train:(num_train+thresh)]))
                X_val_for_test = np.concatenate((X_val[thresh:num_train], X_val[(num_train+thresh):]))
                Y_val_for_test = np.concatenate((Y_val[thresh:num_train], Y_val[(num_train+thresh):]))

                pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(n_jobs=-1))
                lr = pipe.fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                y_pred = lr.predict_proba(X_val_for_test)[:, 1]

                # HERE: calculate + plot ROC curve
                results, tpr, fpr = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf, out, args.model, args.seed, args.id)
                if best_tnr < results['TMP']['TNR']:
                    best_tnr = results['TMP']['TNR']
                    best_index = score
                    # HERE: calculate + plot new ROC curve for best results
                    best_result, tpr, fpr = lib_regression.detection_performance(lr, X_test, Y_test, outf, out, args.model, args.seed, args.id)
                    
                    last_layer_dists = total_X[:,-1] # only take last feature distance, discarding intermediate dists (only for visualization purposes)
                    last_layer_preds = y_preds[:,-1] # only take class predictions from last layer
                    # plot histogram
                    id_array, ood_array = lib_regression.plot_histogram(last_layer_dists, total_Y, out, best_index, args.model, args.seed, args.id, args.usecase)
                    
                    # plot accuracy-rejection curve
                    fractions, accuracies, ideal = lib_regression.plot_accrej_curve(y_true, last_layer_preds, last_layer_dists, total_Y, out, best_index, args.model, args.seed, args.id, args.usecase)

            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
        
    # print the results
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1

    if args.final_model:
        plot_dir =  "../Integration/Mahalanobis"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_file = plot_dir+'/%s_%s_%s.npz'%(args.model, args.dataset, str(args.seed))
        np.savez(plot_file, arr0=id_array, 
                            arr1=ood_array, 
                            arr2=fractions, 
                            arr3=accuracies, 
                            arr4=ideal,
                            arr7=tpr,
                            arr8=fpr,
                            compress=True)

if __name__ == '__main__':
    main()