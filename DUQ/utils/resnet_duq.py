import torch
import torch.nn as nn


class ResNet_DUQ(nn.Module):
    def __init__(self, feature_extractor, num_classes, centroid_size, model_output_size, length_scale=0.1, gamma=0.999):
        super().__init__()
        self.gamma = gamma # momentum for exp-average in clustering task
        # W: for clustering -> (#instances in cluster, #clusters, logit size)
        self.W = nn.Parameter(torch.zeros(centroid_size, num_classes, model_output_size))
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")
        self.feature_extractor = feature_extractor
        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer("m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05))
        self.m = self.m * self.N # m: input dim, N: cluster members/centroid size
        self.sigma = length_scale # "spread" of centroids/clusters

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        embeddings = self.m / self.N.unsqueeze(0)
        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()
        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)
        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        z = self.feature_extractor(x)
        y_pred = self.rbf(z)
        return y_pred