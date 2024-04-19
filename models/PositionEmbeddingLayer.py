import torch.nn as nn


class PositionEncodingClusteringCoefficient(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionEncodingClusteringCoefficient, self).__init__()
        self.clustering_coefficient_linear = nn.Linear(1, hidden_dim)
        # Custom initialization for a linear layer
        self.init_weights()

    def forward(self, clustering_coefficient):
        clustering_coefficient = self.clustering_coefficient_linear(clustering_coefficient)
        return clustering_coefficient

    def init_weights(self):
        nn.init.kaiming_normal_(self.clustering_coefficient_linear.weight)


class PositionEncodingBidirectionalLinks(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionEncodingBidirectionalLinks, self).__init__()
        self.bidirectional_links_ratio_linear = nn.Linear(1, hidden_dim)

    def forward(self, bidirectional_links_ratio):
        bidirectional_links_ratio = self.bidirectional_links_ratio_linear(bidirectional_links_ratio)
        return bidirectional_links_ratio
