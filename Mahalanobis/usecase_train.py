from __future__ import print_function
import argparse
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import timm
import data_loader
import numpy as np
import calculate_log as callog
import os
import lib_generation
from models import densenet, resnet, linear, vgg
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Train model backbone on use case data for Mahalanobis method')
    parser.add_argument('--epochs', type=int, nargs='?', action='store', default=60,
                        help='How many epochs to train. Default: 60.')
    parser.add_argument('--batch_size', type=int, nargs='?', action='store', default=64,
                        help='batch size for the data. Default: 64.')
    parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-4,
                        help='learning rate. Default: 1e-4.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for storing the model')
    parser.add_argument('--model', type=str, default='resnet', choices=['linear', 'vgg16', 'resnet34', 'resnet50', 'densenet'], 
                        help='Base architecture for the model')
    parser.add_argument('--usecase', type=str, default='RR', choices=['RR', 'Emblem'], 
                        help='Which use case data to train on')
    parser.add_argument('--aug', action='store_true', help='whether to use augmentation on images')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, (0/1, default: 0)')
    parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='pre_trained',
                        help='Where to save learnt weights and train vectors. Default: \'pre_trained\'.')
    parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='train_results',
                        help='Where to save tensorboard logs. Default: \'train_results\'.')
    parser.add_argument('--uc_datapath', type=str, nargs='?', action='store', 
                    default='../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean',
                    help='\'../../data/RRUseCase/Per-Camera-Datasets/bodenblech_bb-back-top-l_0001_ds20_v0-clustered__clean\' or \'../../data/EmblemUseCase\'')
    args = parser.parse_args()
    print(args)

    # Where to save models weights
    models_dir = args.models_dir
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Where to save plots and error, accuracy logs
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    writer = SummaryWriter(log_dir=f"{results_dir}/seed{str(args.seed)}_lr{str(args.lr)}", comment="seed"+str(args.seed))

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu))
    else:
        device = torch.device("cpu")
    print("Using device: ", device)

    # ------------------------------------------------------------------------------------------------------
    # load data
    datapath = args.uc_datapath
    if args.usecase == 'RR':
        train_loader, val_loader, test_loader, _, num_classes, input_size = data_loader.get_UseCaseloaders(datapath, args.batch_size, args.aug)
    elif args.usecase == 'Emblem':
        train_loader, val_loader, test_loader, _, num_classes, input_size = data_loader.get_Emblemloaders(datapath, args.batch_size, args.aug)

    # initialize model
    if args.model == 'linear':
        model = linear.Linear_2L(in_channels=3, out_channels=num_classes, input_size=input_size, n_hid=200)
        model = model.to(device)
    elif args.model == 'vgg16':
        model = vgg.VGG16(in_channels=3, out_channels=num_classes, input_size=input_size)
        model = model.to(device)
    elif args.model == 'resnet34':
        model = resnet.ResNet34(num_classes=num_classes, input_size=input_size)
        model = model.to(device)
    elif args.model == 'resnet50':
        model = resnet.ResNet50(num_classes=num_classes, input_size=input_size)
        model = model.to(device)
    elif args.model == 'densenet':
        model = densenet.DenseNet121(num_classes=num_classes, input_size=input_size)
        model = model.to(device)

    # ---------------------------------------------------------------------------------------------------------------------
    # Define the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    train_loss_history = np.zeros(args.epochs)
    train_acc_history = np.zeros(args.epochs)
    val_loss_history = np.zeros(args.epochs)
    val_acc_history = np.zeros(args.epochs)
    best_loss = np.inf
    best_epoch_loss = 0

    # Training + validation of the model
    for epoch in tqdm(range(args.epochs)):
        print(f"\n-------------------------------   Epoch {epoch + 1}   -------------------------------\n")
        train_loss, train_acc = lib_generation.train(model, device, train_loader, optimizer, epoch, loss_fn)
        val_loss, val_acc = lib_generation.validate(model, device, val_loader, loss_fn)

        train_loss_history[epoch] = train_loss
        train_acc_history[epoch] = train_acc
        val_loss_history[epoch] = val_loss
        val_acc_history[epoch] = val_acc

        # print and log train + val results
        writer.add_scalar("Loss/train", train_loss, epoch+1)
        writer.add_scalar("Accuracy/train", train_acc, epoch+1)
        writer.add_scalar("Loss/val", val_loss, epoch+1)
        writer.add_scalar("Accuracy/val", val_acc, epoch+1)

        # save model with lowest validation loss
        if val_loss < best_loss:
            best_epoch_loss = epoch + 1
            best_loss = val_loss
            torch.save(model.state_dict(), models_dir +'/'+args.model+'_'+args.usecase+'usecase_model_seed'+str(args.seed)+'.pt')

    writer.close()

if __name__ == '__main__':
    main()