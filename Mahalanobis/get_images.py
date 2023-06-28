import numpy as np
import sys 
import os
import argparse
import cv2
import torch
from torchvision import transforms, utils

import calculate_log as callog
import data_loader
import lib_generation
import lib_regression
from models import densenet, resnet, linear, vgg

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

parser = argparse.ArgumentParser(description='PyTorch code: store sorted output images from Mahalanobis detector')
parser.add_argument('--seed', type=int, default=0, help='Seed for stored model')
parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | usecase')
parser.add_argument('--usecase', type=str, default='RR', choices=['RR', 'Emblem'], 
                        help='Which use case data the model was trained on')
parser.add_argument('--aug', action='store_true', help='whether augmentation was used for training')
parser.add_argument('--model', type=str, default='linear', choices=['linear', 'vgg16', 'resnet34', 'resnet50', 'densenet'], 
                        help='Base architecture for the model')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='pre_trained',
                        help='Where to save learnt weights and train vectors. Default: \'pre_trained\'.')
parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='\'../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\' or \'../../data/EmblemUseCase\'')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()
print(args)

def main():
    torch.cuda.manual_seed(0)
    device = torch.device("cuda:"+str(args.gpu))

    # set temporary storepath
    outdir = 'output/image_tests/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outdir = outdir + args.usecase +'_'+ args.dataset + '/'
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    
    # dataset and loader
    print('Loading target data: ', args.dataset)
    dataroot = args.uc_datapath
    if args.usecase == 'RR':
        out_dist_list = ['RRusecase']
        args.dataset = 'RRusecase'
        train_loader, val_loader, test_loader, ood_loader, num_classes, input_size = data_loader.get_Elementloaders(dataroot, args.batch_size, args.aug)
    elif args.usecase == 'Emblem':
        out_dist_list = ['Emblemusecase']
        args.dataset = 'Emblemusecase'
        train_loader, val_loader, test_loader, ood_loader, num_classes, input_size = data_loader.get_Emblemloaders(dataroot, args.batch_size, args.aug)

    # pretrained model
    pre_trained_net = './'+args.models_dir+'/'+args.model+'_'+args.usecase+'usecase_model_seed'+str(args.seed)+'.pt'
    #loading weights on predefined architecture
    if args.model == 'linear':
        model = linear.Linear_2L(in_channels=3, out_channels=num_classes, input_size=input_size, n_hid=200)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
    if args.model == 'vgg16':
        model = vgg.VGG16(in_channels=3, out_channels=num_classes, input_size=input_size)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
    if args.model == 'resnet34':
        model = resnet.ResNet34(num_classes=num_classes, input_size=input_size)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
    if args.model == 'resnet50':
        model = resnet.ResNet50(num_classes=num_classes, input_size=input_size)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
    if args.model == 'densenet':
        model = densenet.DenseNet121(num_classes=num_classes, input_size=input_size)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
    model.to(device)
    print('loading model: ' + pre_trained_net)

    #set shell for feature list variable
    temp_x = torch.rand(2,3,input_size,input_size).to(device)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    
    #get sample mean and covariance 
    print('get sample mean and covariance')
    torch.cuda.empty_cache()
    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader, device)

    #get Mahalanobis scores on ID and OOD data
    magnitude = 0.0
    print('get Mahalanobis scores for ID data')
    for i in range(num_output):
        if i == 0:
            M_in, in_preds, ID_labels, ID_images = lib_generation.get_Mahalanobis_score(model, args.model, args.dataset, args.seed, \
                                                                            test_loader, num_classes, outdir, \
                                                                            True, sample_mean, precision, i, magnitude, device)
        else:
            M_in, in_preds, _, _ = lib_generation.get_Mahalanobis_score(model, args.model, args.dataset, args.seed, \
                                                                            test_loader, num_classes, outdir, \
                                                                            True, sample_mean, precision, i, magnitude, device)
        M_in = np.asarray(M_in, dtype=np.float32)
        in_preds = np.asarray(in_preds, dtype=np.float32)
        if i == 0:
            Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            predictions = in_preds.reshape((in_preds.shape[0], -1))
        else:
            Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            predictions = np.concatenate((predictions, in_preds.reshape((in_preds.shape[0], -1))), axis=1)

    # get preds for ID data and concat with ID labels
    predictions = np.asarray(predictions, dtype=np.float32)

    print('get Mahalanobis scores for OOD data')
    for out_dist in out_dist_list:
        print('Out-distribution: ' + out_dist) 
        for i in range(num_output):
            if i == 0:
                M_out, out_preds, _, OOD_images = lib_generation.get_Mahalanobis_score(model, args.model, args.dataset, args.seed, \
                                                                    ood_loader, num_classes, outdir, \
                                                                    False, sample_mean, precision, i, magnitude, device)
            else:
                M_out, out_preds, _, _ = lib_generation.get_Mahalanobis_score(model, args.model, args.dataset, args.seed, \
                                                                    ood_loader, num_classes, outdir, \
                                                                    False, sample_mean, precision, i, magnitude, device)
            M_out = np.asarray(M_out, dtype=np.float32)
            out_preds = np.asarray(out_preds, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                ood_predictions = out_preds.reshape((out_preds.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
                ood_predictions = np.concatenate((ood_predictions, out_preds.reshape((out_preds.shape[0], -1))), axis=1)

        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
        Mahalanobis_comb = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
    
    print(" ---------------- FINAL --------------")
    ID_labels = ID_labels.squeeze()
    ID_images = ID_images.squeeze()
    OOD_images = OOD_images.squeeze()

    print("ID images: ", ID_images.shape) #(2265, 3, 64, 64)
    print("ID labels: ", ID_labels.shape) #(2265,) -> correct ID class
    print("ID Mahalanobis: ", Mahalanobis_in.shape) #(2265, 3) -> distance to each cluster
    print("ID preds: ", predictions.shape) #(2265, 3) -> class assignments

    print("OOD images: ", OOD_images.shape) #(2265, 3, 64, 64)
    print("OOD Mahalanobis: ", Mahalanobis_out.shape) #(2265, 3) -> distance to each cluster
    print("OOD preds: ", ood_predictions.shape) #(2265, 3) -> class assignments

    print("Mahalanobis_data: ", Mahalanobis_data.shape) #(4530, 3) -> cluster distances combined
    print("Mahalanobis_labels: ", Mahalanobis_labels.shape) #(4530, 1) -> cluster distances combined
    print("Mahalanobis_comb: ", Mahalanobis_comb.shape) #(4530, 4) -> cluster distances + OOD indicator (test ID/OOD combined) 

    all_images = torch.cat((ID_images, OOD_images))
    print("ALL images: ", all_images.shape)
    
    # -------------------------------- Regression Part --------------------------
    # distribute ID/OOD and split into train + test for log-reg.
    num_out = OOD_images.shape[0]
    num_in = ID_images.shape[0]
    partition = num_out
    a = min(num_in, num_out)
    num_train = int(np.floor(a*(2/3)))
    """
    if args.usecase == 'Emblem':
        num_train = int(np.floor(a/2))
    else: 
        num_train = 500
    """

    # separate ID/OOD
    Y_adv = Mahalanobis_labels[:partition] # adv == OOD
    Y_norm = Mahalanobis_labels[partition: :] # norm == ID
    Y_out = torch.full((1, OOD_images.size(0)), fill_value=-1)
    Y_out = Y_out.squeeze().numpy() # -1 value for OOD images
    Y_in = ID_labels # contain actual class labels
    np.set_printoptions(threshold=sys.maxsize)
    print("Y_in: ", Y_in.shape, Y_in)
    print("Y_out: ", Y_out.shape, Y_out)

    # train/test-split
    X_train = np.concatenate((Mahalanobis_in[:num_train], Mahalanobis_out[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))
    X_test = np.concatenate((Mahalanobis_in[num_train:], Mahalanobis_out[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))
    Y_print_train = np.concatenate((Y_in[:num_train], Y_out[:num_train]))
    Y_print = np.concatenate((Y_in[num_train:], Y_out[num_train:]))

    train_images = torch.cat((ID_images[:num_train], OOD_images[:num_train]))
    test_images = torch.cat((ID_images[num_train:], OOD_images[num_train:]))


    # take first 20 ID images (class 0) to test set to make sure all classes are present
    X_0, X_rest = np.split(X_train, [20])
    X_test = np.concatenate((X_0, X_test))
    Y_0, Y_rest = np.split(Y_train, [20])
    Y_test = np.concatenate((Y_0, Y_test))
    labels_0, labels_rest = np.split(Y_print_train, [20])
    Y_print = np.concatenate((labels_0, Y_print))
    images_0, images_rest = torch.split(train_images, [20, train_images.size(0)-20])
    test_images = torch.cat((images_0, test_images))
    
    # perform log-regression for OOD detection
    print("Performing Logistic Regression for OOD detection")
    pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(n_jobs=-1))
    Y_rest = Y_rest.squeeze()
    logreg = pipe.fit(X_rest, Y_rest)
    y_pred = logreg.predict_proba(X_test)[:, 1]

    # sort images by uncertainty score
    sorted_unc, indices = torch.sort(torch.tensor(y_pred), dim=0, descending=True)
    sorted_imgs = test_images[indices[:]]
    sorted_lbls = Y_print[indices[:]]

    # store images with their corresponding uncertainty as filename
    img_path = "OUT_images/"+args.dataset+'_'+args.model+'_'+str(args.seed)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in range(sorted_imgs.shape[0]):
        img = sorted_imgs[i]
        # revert normalization to get original image
        if args.usecase == 'RR':
            mean = torch.tensor([0.46840793, 0.23778377, 0.19240856]).to(device)
            std = torch.tensor([0.12404595643681854, 0.08136763306617903, 0.07868825907965848]).to(device)
        elif args.usecase == 'Emblem':
            mean = torch.tensor([0.3987, 0.4262, 0.4706]).to(device)
            std = torch.tensor([0.2505, 0.2414, 0.2466]).to(device)
        img = img * std[:, None, None] + mean[:, None, None]

        uncertainty_score = sorted_unc[i].detach().cpu().numpy()
        class_label = sorted_lbls[i]
        utils.save_image(img, img_path+'/'+str(i)+'_'+str(uncertainty_score)+'_label'+str(class_label)+'.png')
        
if __name__ == '__main__':
    main()