from __future__ import print_function
import torch
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scipy.spatial.distance import pdist, cdist, squareform

# use case training helper functions
def train(model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    size = len(train_loader.dataset)
    num_batches = len(train_loader)

    correct = 0
    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(images), len(train_loader.dataset),
                100. * batch / len(train_loader), loss.item()))
    acc = 100. * correct / len(train_loader.dataset)
    print('Train Accuracy: {:.3f}'.format(acc))
    return np.round(loss.item(), 5), acc

def validate(model, device, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            val_loss += loss_fn(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), acc))
    return np.round(val_loss, 5), acc

# lid of a batch of query points X
def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def sample_estimator(model, num_classes, feature_list, train_loader, device):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
            precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = [] 
    for i in range(num_output):
        temp_list = [] 
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in tqdm(train_loader):
        total += data.size(0)
        data = data.to(device)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1] # get max LOGIT index (not softmax!)
        equal_flag = pred.eq(target.to(device)).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for i, num_feature in enumerate(feature_list):
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision

def get_Mahalanobis_score(model, net, dataset, seed, test_loader, num_classes, outf, out_flag, sample_mean, precision, layer_index, magnitude, device):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    predictions = []

    storepath = '%s/%s_%s/'%(outf, net, dataset)
    if not os.path.exists(storepath):
        os.makedirs(storepath)
    
    if out_flag == True:
        temp_file_name = storepath+'seed%s_confidence_Ga%s_In.txt'%(seed, str(layer_index))
    else:
        temp_file_name = storepath+'seed%s_confidence_Ga%s_Out.txt'%(seed, str(layer_index))
        
    g = open(temp_file_name, 'w')
    
    for j, (data, target) in enumerate(tqdm(test_loader)):
        
        data, target = data.to(device), target.to(device)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        if j == 0:
            labels = target
            images = data
        else: 
            labels = torch.cat((labels, target))
            images = torch.cat((images, data))
        
        # compute Mahalanobis score -> only used for adding noise to sample
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i] # mean of class i at layer "layer_index"
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # find closest class
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        # Input_processing 
        # -> perturb image with additive noise of chosen magnitude(from list) to make ID/OOD more separable 
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.2010))
        
        tempInputs = torch.add(input=data.data, other=gradient, alpha=-magnitude) # = data - magnitude * gradient

        # HERE: actual confidence/distance calculation
        noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        # take class-conf value for class with max precision (closest centroid/lowest covariance value)
        noise_gaussian_score, pred_classes = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        predictions.extend(pred_classes.cpu().numpy())
        all_images = torch.unsqueeze(images, dim=1)
        all_labels = np.expand_dims(labels.cpu().numpy(), axis=1)
        
        for i in range(data.size(0)):
            g.write("{};".format(noise_gaussian_score[i]))
    g.close()

    return Mahalanobis, predictions, all_labels, all_images

def get_posterior(model, net_type, test_loader, magnitude, temperature, outf, out_flag, device):
    '''
    Compute the maximum value of (processed) posterior distribution - ODIN
    return: null
    '''
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt'%(outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt'%(outf)
        
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    
    for data, _ in test_loader:
        total += data.size(0)
        data = data.to(device)
        batch_output = model(data)
            
        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        loss = criterion(outputs, labels)
        loss.backward()
         
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.2010))

        tempInputs = torch.add(data.data,  -magnitude, gradient)
        outputs = model(tempInputs)
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)
        
        for i in range(data.size(0)):
            if total <= 1000:
                g.write("{}\n".format(soft_out[i]))
            else:
                f.write("{}\n".format(soft_out[i]))
                
    f.close()
    g.close()