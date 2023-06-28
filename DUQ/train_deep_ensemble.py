import os
import argparse
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.datasets import all_datasets
from utils.cnn_duq import SoftmaxModel as CNN
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.resnet = resnet18(num_classes=num_classes)
        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, n_hid):
        super().__init__()
        input_dim = in_channels * input_size * input_size
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, out_channels)
        self.bn = nn.BatchNorm1d(n_hid) 

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.bn(self.fc1(x)))
        x = F.relu(self.bn(self.fc2(x)))
        x = F.relu(self.bn(self.fc3(x)))
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)

    
def train(num, members, model, train_loader, optimizer, epoch, num_epochs, loss_fn, device):
    model.train()
    total_loss = []
    correct = 0

    print(f"\n---------------------------------- Model [{num+1} / {members}] ----------------------------------")
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        prediction = model(data)
        loss = loss_fn(prediction, target)
        class_prediction = prediction.max(1)[1]
        correct += class_prediction.eq(target.view_as(class_prediction)).sum().item()

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = torch.tensor(total_loss).mean()
    avg_acc = 100.0 * correct / len(train_loader.dataset)
    print(f"Epoch [{epoch} / {num_epochs}] :")
    print(f"Train Set: \tAverage Loss: {avg_loss:.2f}, \tAccuracy: {avg_acc:.2f}%")


def test(epoch, num_epochs, models, test_loader, loss_fn, device):
    models.eval()

    loss = 0
    correct = 0

    print(f"\n##### TEST epoch [{epoch} / {num_epochs}] #####")
    for data, target in test_loader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)

            losses = torch.empty(len(models), data.shape[0])
            predictions = []
            for i, model in enumerate(models):
                predictions.append(model(data))
                losses[i, :] = loss_fn(predictions[i], target, reduction="sum")

            predictions = torch.stack(predictions)

            loss += torch.mean(losses)
            avg_prediction = predictions.exp().mean(0)

            # get the index of the max log-probability
            class_prediction = avg_prediction.max(1)[1]
            correct += class_prediction.eq(target.view_as(class_prediction)).sum().item()

    loss /= len(test_loader.dataset)
    percentage_correct = 100.0 * correct / len(test_loader.dataset)
    
    print("Test set: \tAverage loss: {:.4f}, \tAccuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct))

    return loss, percentage_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=75, help="number of epochs to train (default: 75)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use for training (default: 64)")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate (default: 0.05)")
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "linear", "conv"], help="Pick an architecture (default: resnet18)")
    parser.add_argument("--ensemble", type=int, default=5, help="Ensemble size (default: 5)")
    parser.add_argument("--dataset", required=True, choices=["cifar",  "RR", "Emblem", "Part"], help="Select a dataset")
    parser.add_argument('--aug', action='store_true', help='whether to use augmentation on images')
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--output_dir", type=str, default="results", help="set output folder")
    parser.add_argument('--gpu', type=int, default=1, help='Which GPU to use, choice: 0/1, default: 1')
    parser.add_argument('--id', type=int, help='Model identifier for storage')
    args = parser.parse_args()
    print(args)

    device = device = torch.device("cuda:"+str(args.gpu))
    print("Device used: ", device)

    torch.manual_seed(args.seed)

    loss_fn = F.nll_loss

    ds = all_datasets[args.dataset](aug=args.aug)
    input_size, num_classes, train_dataset, test_dataset, _ = ds

    kwargs = {"num_workers": 4, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5000, shuffle=False, **kwargs)

    if args.architecture == "linear":
        milestones = [10, 20]
        ensemble = [Linear(in_channels=3, out_channels=num_classes, input_size=input_size, n_hid=200).to(device) for _ in range(args.ensemble)]
    elif args.architecture == "conv":
        milestones = [10, 20]
        ensemble = [CNN(in_channels=3, out_channels=num_classes, input_size=input_size).to(device) for _ in range(args.ensemble)]
    elif args.architecture == "resnet18":
        milestones = [15, 30]
        ensemble = [ResNet(input_size, num_classes).to(device) for _ in range(args.ensemble)]

    ensemble = torch.nn.ModuleList(ensemble)
    optimizers = []
    schedulers = []

    for model in ensemble:
        # Need different optimisers to apply weight decay and momentum properly
        # when only optimising one element of the ensemble
        optimizers.append(torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4))
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[-1], milestones=milestones, gamma=0.1))

    for epoch in range(1, args.epochs + 1):
        for i, model in enumerate(ensemble):
            train(i, args.ensemble, model, train_loader, optimizers[i], epoch, args.epochs, loss_fn, device)
            schedulers[i].step()

        test(epoch, args.epochs, ensemble, test_loader, loss_fn, device)

    path = f"runs/{args.output_dir}"
    modelname = f"/{args.dataset}_{len(ensemble)}ens_id{str(args.id)}"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(ensemble.state_dict(), path + modelname + ".pt")

if __name__ == "__main__":
    main()