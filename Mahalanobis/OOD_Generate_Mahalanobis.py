from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import sys 
import calculate_log as callog
import os
import lib_generation
from models import densenet, resnet, linear, vgg
from torchvision import transforms

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | usecase')
parser.add_argument('--usecase', type=str, default='RR', choices=['RR', 'Emblem','Part'], 
                        help='Which use case data the model was trained on')
parser.add_argument('--aug', action='store_true', help='whether augmentation was used for training')
parser.add_argument('--seed', type=int, default=0, help='Seed for stored model')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--model', type=str, default='resnet', choices=['linear', 'vgg16', 'resnet34', 'resnet50', 'densenet'], 
                        help='Base architecture for the model')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='pre_trained',
                        help='Where to save learnt weights and train vectors. Default: \'pre_trained\'.')
parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='\'../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\' or \'../../data/EmblemUseCase\'')
args = parser.parse_args()
print(args)

def main():
    # set device
    torch.cuda.manual_seed(0)
    device = torch.device("cuda:"+str(args.gpu))

    # for application on toy datasets from paper as sanity check
    if args.dataset == 'cifar10' and args.model == 'resnet':
        # set the path to pre-trained model and output
        outdir = 'output/resnet_cifar10/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        
        # define model and data info
        pre_trained_net = './'+args.models_dir+'/resnet34.pt'
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        dataroot = '../../data'
        num_classes = 10

        model = resnet.resnet34(pretrained=False, progress=True, device=device, num_classes=num_classes, dataset=args.dataset)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        
        # load dataset
        print('Loading target data: ', args.dataset)
        in_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),]) #previous std: 0.2023, 0.1994, 0.2010
        train_loader, test_loader, input_size = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, dataroot, args.aug)

    else: # application on custom dataset
        # set the path to pre-trained model and output
        outdir = 'output/' + args.usecase +'_'+ args.dataset + '/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        
        dataroot = args.uc_datapath
        if args.usecase == 'RR':
            num_classes = 3
            out_dist_list = ['RRusecase'] # to compare to other OOD datasets: insert name here
            args.dataset = 'RRusecase'
        elif args.usecase == 'Emblem':
            num_classes = 2
            out_dist_list = ['Emblemusecase'] # to compare to other OOD datasets: insert name here
            args.dataset = 'Emblemusecase'

        # load dataset
        print('loading target data: ', args.dataset)
        in_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),]) # only for dummy datasets
        train_loader, test_loader, input_size = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, dataroot, args.aug)

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
    
    # set information about feature extraction
    model.eval()
    if args.dataset == 'cifar10':
        temp_x = torch.rand(2,3,input_size,input_size).to(device)
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
    elif args.dataset == 'RRusecase' or args.dataset == 'Emblemusecase':
        temp_x = torch.rand(2,3,input_size,input_size).to(device)
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        
    print('get sample mean and covariance')
    torch.cuda.empty_cache()
    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader, device)
    
    print('get Mahalanobis scores')
    # define noise magnitudes for perturbation (see method from original paper)
    m_list = [0.0] # other values from paper (possibly several): 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            if i == 0: # get the ID_labels only once (for first feature layer)
                M_in, in_preds, ID_labels = lib_generation.get_Mahalanobis_score(model, args.model, args.dataset, args.seed, \
                                                                                test_loader, num_classes, outdir, \
                                                                                True, sample_mean, precision, i, magnitude, device)
            else:
                M_in, in_preds, _ = lib_generation.get_Mahalanobis_score(model, args.model, args.dataset, args.seed, \
                                                                                test_loader, num_classes, outdir, \
                                                                                True, sample_mean, precision, i, magnitude, device)
            M_in = np.asarray(M_in, dtype=np.float32)
            in_preds = np.asarray(in_preds, dtype=np.float32)
            if i == 0: # first batch
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                predictions = in_preds.reshape((in_preds.shape[0], -1))
            else: # concatenate batch-wise class predictions and distances (i.e. uncertainties)
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
                predictions = np.concatenate((predictions, in_preds.reshape((in_preds.shape[0], -1))), axis=1)
        
        # get class predictions for ID data and concat with ID labels 
        predictions = np.asarray(predictions, dtype=np.float32)
        Preds_data = np.concatenate((predictions, ID_labels), axis=1)
        # store predictions for later evaluation
        storepath = '%s/%s_%s'%(outdir, args.model, args.dataset)
        if not os.path.exists(storepath):
            os.makedirs(storepath)
        file_name_pred = os.path.join(storepath, 'Preds_%s_Mahalanobis_%s_%s_seed%s.npy' % (args.model, str(magnitude), args.dataset, args.seed))
        np.save(file_name_pred, Preds_data)

        # loop through OOD datasets (if there are several)
        for out_dist in out_dist_list:
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, dataroot)
            print('Out-distribution: ' + out_dist) 
            for i in range(num_output):
                M_out, _, _ = lib_generation.get_Mahalanobis_score(model, args.model, args.dataset, args.seed, \
                                                            out_test_loader, num_classes, outdir, \
                                                             False, sample_mean, precision, i, magnitude, device)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0: # first batch 
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else: # concatenate batch-wise distances (i.e. uncertainties)
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            # concatenate and store ID and OOD distances/uncertainties for later evaluation
            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(storepath, '%s_Mahalanobis_%s_%s_%s_seed%s.npy' % (args.model, str(magnitude), args.dataset , out_dist, args.seed))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)
    
if __name__ == '__main__':
    main()