import torch
import torch.nn as nn


class NodeFeatureEmbeddingLayer(nn.Module):
    def __init__(self, hidden_dim, numerical_feature_size=5, categorical_feature_size=3, des_feature_size=768,
                 tweet_feature_size=768, dropout=0.3):
        super(NodeFeatureEmbeddingLayer, self).__init__()
        self.numerical_feature_size = numerical_feature_size
        self.categorical_feature_size = categorical_feature_size
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.numerical_feature_linear = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_dim // 4),
            self.activation
        )

        self.categorical_feature_linear = nn.Sequential(
            nn.Linear(categorical_feature_size, hidden_dim // 4),
            self.activation
        )

        self.des_feature_linear = nn.Sequential(
            nn.Linear(des_feature_size, hidden_dim // 4),
            self.activation
        )

        self.tweet_feature_linear = nn.Sequential(
            nn.Linear(tweet_feature_size, hidden_dim // 4),
            self.activation
        )

        self.total_feature_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.init_weights()

    def forward(self, des_tensor, tweet_tensor, num_prop, category_prop):
        num_prop = self.numerical_feature_linear(num_prop)
        category_prop = self.categorical_feature_linear(category_prop)
        des_tensor = self.des_feature_linear(des_tensor)
        tweet_tensor = self.tweet_feature_linear(tweet_tensor)
        x = torch.cat((num_prop, category_prop, des_tensor, tweet_tensor), dim=1)
        x = self.total_feature_linear(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()