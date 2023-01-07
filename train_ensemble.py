import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, datasets
import argparse

"""
DeepEnsemble hyperparameters according to Lakshminarayanan et al. (2017):
    - scoring rule/loss: NLL (also tests on Brier score)
    - batch_size = 100
    - optimizer: Adam
    - lr = 0.1
    - default weight initialization in Torch
    - architecture: 3-layer MLP (MNIST), VGG-sytle convnet (SVHN)
"""

#TODO: check: how did other papers make Ensembles and MCDropout comparable? Which parameters/architectures did they use?

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    correct = 0
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()
        if idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(images), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))

    acc = 100. * correct / len(train_loader.dataset)
    print('Train Accuracy: {:.3f}'.format(acc))

def validate(model, device, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            val_loss += loss_fn(output, labels).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), acc))

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for storing the model')
    parser.add_argument('--log-interval', type=int, default=50, help='batch interval for logging training status')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Split the training data into a training set and a validation set
    train_size = int(0.9 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = data.random_split(mnist_train, [train_size, val_size])

    # Create dataloaders
    train_loader = data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(mnist_val, batch_size=args.batch_size, shuffle=False)
    test_loader = data.DataLoader(mnist_test, batch_size=1000, shuffle=False)

    # Define the models
    model1 = Net().to(device)
    model2 = Net().to(device)
    model3 = Net().to(device)
    model4 = Net().to(device)
    model5 = Net().to(device)
    models = [model1, model2, model3, model4, model5]

    # Define the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()

    for i, model in enumerate(models):
        optimizer = optim.Adam(model.parameters(), args.lr)

        # Training + validation of the model
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, loss_fn)
            validate(model, device, val_loader, loss_fn)
            optimizer.step()

        # Evaluate the model
        test(model, device, test_loader)

        if args.save_model:
            torch.save(model.state_dict(), "mnist_model"+str(args.seed)+"_"+str(i)+".pt")

if __name__ == '__main__':
    main()