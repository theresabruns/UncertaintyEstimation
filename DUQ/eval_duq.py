import argparse
import json
import pathlib
from tqdm import tqdm 
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as td
from torchvision.models import resnet18

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from utils.wide_resnet import WideResNet
from utils.resnet import ResNet34, ResNet50
from utils.vgg import VGG16
from utils.linear import Linear_2L
from utils.densenet import DenseNet121
from utils.resnet_duq import ResNet_DUQ
from utils.datasets import all_datasets
from utils.evaluate_ood import *

def main(architecture, data, batch_size, length_scale, exp_dir, seed, id, gpu, save_uncimgs, save_txt, final_model):
    # set device
    device = device = torch.device("cuda:"+str(gpu))
    print("Device used: ", device)

    # get datasets and dataloader
    ds = all_datasets[data]()
    if data == "cifar": # load ID cifar + OOD svhn
        input_size, num_classes, _, test_dataset, _ = ds
        _, _, _ , ood_dataset, _ = all_datasets["SVHN"]()
    else:
        input_size, num_classes, _, test_dataset, ood_dataset = ds

    test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    ood_loader = td.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False) 

    # load model
    if architecture == "WRN":
        model_output_size = 640
        feature_extractor = WideResNet()
    elif architecture == "ResNet18":
        model_output_size = 512
        feature_extractor = resnet18()
        # Adapted resnet from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        feature_extractor.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        feature_extractor.maxpool = torch.nn.Identity()
        feature_extractor.fc = torch.nn.Identity()
    elif architecture == "linear":
        model_output_size = 100
        feature_extractor = Linear_2L(in_channels=3, out_channels=num_classes, input_size=input_size, n_hid=200)
        feature_extractor.fc4 = torch.nn.Identity()
    elif architecture == "vgg16":
        model_output_size = 512
        feature_extractor = VGG16(in_channels=3, out_channels=num_classes, input_size=input_size)
        feature_extractor.fc1 = torch.nn.Identity()
        feature_extractor.fc2 = torch.nn.Identity()
    elif architecture == "resnet34":
        model_output_size = 512
        feature_extractor = ResNet34(num_classes=num_classes, input_size=input_size)
        feature_extractor.linear = torch.nn.Identity()
    elif architecture == "resnet50":
        model_output_size = 512
        feature_extractor = ResNet50(num_classes=num_classes, input_size=input_size)
        feature_extractor.linear = torch.nn.Linear(512*4, 512)
    elif architecture == "densenet":
        model_output_size = 512
        feature_extractor = DenseNet121(num_classes=num_classes, input_size=input_size)
        feature_extractor.linear = torch.nn.Linear(32*16*2, 512)

    centroid_size = model_output_size
    model = ResNet_DUQ(feature_extractor, num_classes, centroid_size, model_output_size, length_scale)
    modelpath = 'runs/'+exp_dir + '/DUQ_'+architecture+'_seed'+str(seed)+'.pt'
    model.load_state_dict(torch.load(modelpath))
    model = model.to(device)
    model.eval()

    # ------------------------------- Inference on ID data -------------------------------
    print('\nGetting predictions on ID data:')
    for j, (images, labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        dists, preds = output.max(dim=1)
        batch_dists = dists.detach().cpu().numpy()
        batch_preds = preds.detach().cpu().numpy()
        batch_error = preds.ne(labels).sum() /batch_size

        if j == 0: # first batch
            predictions = batch_preds.reshape((batch_preds.shape[0], -1))
            distances = batch_dists.reshape((batch_dists.shape[0], -1))
            all_labels = labels
            all_images = images
            error = batch_error
        else: # stack batch results in columns
            predictions = np.concatenate((predictions, batch_preds.reshape((batch_preds.shape[0], -1))))
            distances = np.concatenate((distances, batch_dists.reshape((batch_dists.shape[0], -1))))
            all_labels = torch.cat((all_labels, labels))
            all_images = torch.cat((all_images, images))
            error = (error + batch_error) / 2

    # ------------------------------- Inference on OOD data -------------------------------
    print('\nGetting predictions on OOD data:')
    for j, (ood_images, _) in enumerate(tqdm(ood_loader)):
        ood_images = ood_images.to(device)

        ood_output = model(ood_images)
        ood_dists, _ = ood_output.max(dim=1)
        ood_batch_dists = ood_dists.detach().cpu().numpy() 

        if j == 0: # first batch
            ood_distances = ood_batch_dists.reshape((ood_batch_dists.shape[0], -1))
            all_ood_images = ood_images
        else: # stack batch results in columns
            ood_distances = np.concatenate((ood_distances, ood_batch_dists.reshape((ood_batch_dists.shape[0], -1))))
            all_ood_images = torch.cat((all_ood_images, ood_images))

    accuracy = 1.0 - error

    id_array = np.array(distances).flatten()
    ood_array = np.array(ood_distances).flatten()
    flat_preds = predictions.flatten()

    # ------------------------------------------------------ OOD detection metrics -----------------------------------------------------------
    # plot and save histogram
    kl_div = plot_histogram(id_array, ood_array, architecture, seed, id, data)

    # perform binary ID/OOD classification via measure 
    tpr, fpr, thresholds, auroc, UA_values = calc_OODmetrics(id_array, ood_array)

    # calculate AUROC and plot ROC curve
    plot_roc(tpr, fpr, architecture, data, auroc, UA_values, seed, id)

    # plot accuracy-rejection curves
    fractions, accuracies, ideal, filtered_imgs, filtered_uncs, filtered_lbls = calc_rejection_curve(all_labels, flat_preds, id_array, ood_array, device, all_images, all_ood_images, DUQ=True)
    plot_accrej_curve(fractions, accuracies, ideal, architecture, data, seed, id)

    # save results to txt.file
    if save_txt:
        file_path = "result_files"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        results_file = file_path+'/%s_%s_%s_id%s.txt'%(architecture, data, str(seed), str(id))
        g = open(results_file, 'w+')

        g.write("ID Detection Metrics: \n\n")
        g.write("Labels: \t\t{};".format(all_labels))
        g.write("\nPredictions: \t{};".format(flat_preds))
        g.write("\nError: \t\t\t{};".format(error))
        g.write("\nAccuracy: \t\t{};".format(accuracy))

        g.write("\n\nOOD Detection Metrics: \n\n")
        g.write("\nID Distances: \n{};".format(distances.flatten().flatten()))
        g.write("\nOOD Distances: \n{};".format(ood_distances.flatten().flatten()))
        g.write("\nKL-divergence: \t{};".format(kl_div))
        g.write("\nUncertainty-balanced Accuracies: \n{};".format(UA_values.flatten().flatten()))
        g.write("\nUA Thresholds: \n{};".format(thresholds))
        g.write("\nAUROC: \t\t{};".format(auroc))
        g.write("\nACR Accuracies: \t{};".format(accuracies))

        g.close()

    # for best results, set final_model flag 
    # -> saves results to Integration directory for comparative visualization to other methods
    if final_model:
        plot_dir =  "../Integration/DUQ"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_file = plot_dir+'/%s_%s_%s.npz'%(architecture, data, str(seed))
        np.savez(plot_file, arr0=id_array, 
                            arr1=ood_array, 
                            arr2=fractions, 
                            arr3=accuracies, 
                            arr4=ideal,
                            arr7=tpr,
                            arr8=fpr,
                            compress=True)

    # save images named and sorted by their uncertainty value
    if save_uncimgs:
        img_path = "OUT_images/"+data+'_'+architecture+'_'+str(seed)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        for i in range(filtered_imgs.size(0)):
            img = filtered_imgs[i]
            print("IMG: ", img.shape)
            # revert normalization to get original image
            if data == "RR":
                mean = torch.tensor([0.46840793, 0.23778377, 0.19240856]).to(device)
                std = torch.tensor([0.12404595643681854, 0.08136763306617903, 0.07868825907965848]).to(device)
            elif: data == "Emblem":
                mean = torch.tensor([0.3987, 0.4262, 0.4706]).to(device)
                std = torch.tensor([0.2505, 0.2414, 0.2466]).to(device)
            img = img * std[:, None, None] + mean[:, None, None]

            uncertainty_score = filtered_uncs[i].detach().cpu().numpy()
            class_label = filtered_lbls[i].detach().cpu().numpy()
            utils.save_image(img, img_path+'/'+str(uncertainty_score)+'_label'+str(class_label)+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default="ResNet18", choices=["ResNet18", "WRN", 'linear', 'vgg16', 'resnet34', 'resnet50', 'densenet'], help="Pick an architecture (default: ResNet18)")
    parser.add_argument("--data", default="cifar", choices=["cifar", "RR", "Emblem"], help="Dataset to train on (default from paper: cifar10 -> OOD svhn)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use for predictions (default: 1)")
    parser.add_argument("--length_scale", type=float, default=0.1, help="Length scale of RBF kernel (default: 0.1)")
    parser.add_argument("--exp_dir", type=str, default="results", help="set results folder")
    parser.add_argument("--seed", type=int, default=0, help="Seed for loading trained model")
    parser.add_argument('--id', type=int, help='Additional identifier for results in different experiments on same model')
    parser.add_argument('--gpu', type=int, default=1, help='Which GPU to use, choice: 0/1, default: 1')
    parser.add_argument('--save_uncimgs', action='store_true', help='whether to store the 10 most uncertain images for further manual inspection')
    parser.add_argument('--save_txt', action='store_true', help='whether to store results in txt-file')
    parser.add_argument('--final_model', action='store_true', help='for final model, store arrays for integrative comparison plots')

    args = parser.parse_args()
    kwargs = vars(args)
    print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

    pathlib.Path("runs/" + args.exp_dir).mkdir(exist_ok=True)

    main(**kwargs)