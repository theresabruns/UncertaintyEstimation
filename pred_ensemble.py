import torch
import os
import argparse
from train_ensemble import Net
import torch.utils.data as data
from torchvision import transforms, datasets
from scipy.stats import entropy
import numpy as np

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Ensemble uncertainty prediction on MNIST')
    parser.add_argument('--ood', type=bool, default=False, help='whether to use ood dataset (NotMNIST) or reg. testset')

    # Load the MNIST testset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = datasets.MNIST('./data', train=False, download=False, transform=transform)
    test_loader = data.DataLoader(mnist_test, batch_size=1, shuffle=False)
    sample = next(iter(test_loader))[0]
    print(sample.shape)

    # Get the list of all files in the directory
    model_files = []
    models = []
    outputs = []

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.pt'):
            model_files.append(filename)

    for i, file in enumerate(model_files):
        # Create a model with the same architecture as the stored model
        model = Net()
        # Load the stored model's parameters into the model
        model.load_state_dict(torch.load(file))
        models.append(model)

        model.eval()
        pred = model(sample)
        print(pred)
        outputs.append(pred)

    print("Number of ensemble models:", len(models))
    avg_softmax_output = torch.mean(torch.stack(outputs), dim=0)
    avg_softmax_output = avg_softmax_output.detach().numpy()
    ent = entropy(avg_softmax_output.T)
    print("Entropy value: ", ent)

if __name__ == '__main__':
    main()