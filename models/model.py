# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from models.GraphStructuralLayer import GraphStructuralLayer
from models.GraphTemporal import GraphTemporalLayer
from models.NodeFeatureEmbeddingLayer import NodeFeatureEmbeddingLayer
from models.PositionEmbeddingLayer import PositionEncodingClusteringCoefficient, PositionEncodingBidirectionalLinks


class BotDyGNN(nn.Module):
    def __init__(self, args):
        super(BotDyGNN, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.structural_head_config = args.structural_head_config
        self.structural_drop = args.structural_drop
        self.temporal_head_config = args.temporal_head_config
        self.temporal_drop = args.temporal_drop
        self.residual = args.residual
        self.window_size = args.window_size
        self.node_feature_embedding_layer, self.position_encoding_clustering_coefficient_layer, self.position_encoding_bidirectional_links_ratio_layer, self.structural_layer, self.temporal_layer = self.build_model()

    def forward(self, des_tensor_list, tweet_tensor_list, num_prop_list, category_prop_list,
                edge_index_list, clustering_coefficient_list, bidirectional_links_ratio_list,exist_nodes_list, batch_size):
        structural_output = []
        for t in range(0, len(des_tensor_list)):
            des_tensor = des_tensor_list[t]
            tweet_tensor = tweet_tensor_list[t]
            num_prop = num_prop_list[t]
            category_prop = category_prop_list[t]
            x = self.node_feature_embedding_layer(des_tensor, tweet_tensor, num_prop, category_prop)
            edge_index = edge_index_list[t]
            output = self.structural_layer(x, edge_index)[:batch_size]
            structural_output.append(output)
        structural_output = torch.stack(structural_output, dim=1)
        if torch.any(torch.isnan(structural_output)):
            print('structural_output has nan')

        position_embedding_clustering_coefficient = [self.position_encoding_clustering_coefficient_layer(clustering_coefficient_list[t])[:batch_size] for t in range(0, len(des_tensor_list))]
        position_embedding_bidirectional_links_ratio = [self.position_encoding_bidirectional_links_ratio_layer(bidirectional_links_ratio_list[t])[:batch_size] for t in range(0, len(des_tensor_list))]
        position_embedding_clustering_coefficient = torch.stack(position_embedding_clustering_coefficient, dim=1)
        position_embedding_bidirectional_links_ratio = torch.stack(position_embedding_bidirectional_links_ratio, dim=1)
        exist_nodes = exist_nodes_list.transpose(0,1)
        temporal_output, temp = self.temporal_layer(structural_output, position_embedding_clustering_coefficient, position_embedding_bidirectional_links_ratio, exist_nodes)
        if torch.any(torch.isnan(temporal_output)):
            print('temporal_output has nan')
        return temporal_output, temp

    def build_model(self):
        node_feature_embedding_layer = NodeFeatureEmbeddingLayer(hidden_dim=self.hidden_dim)
        position_encoding_clustering_coefficient_layer = PositionEncodingClusteringCoefficient(hidden_dim=self.hidden_dim)
        position_encoding_bidirectional_links_ratio_layer = PositionEncodingBidirectionalLinks(hidden_dim=self.hidden_dim)
        structural_layer = GraphStructuralLayer(hidden_dim=self.hidden_dim,
                                                n_heads=self.structural_head_config,
                                                dropout=self.structural_drop,
                                                residual=self.residual)
        temporal_layer = GraphTemporalLayer(hidden_dim=self.hidden_dim,
                                            n_heads=self.temporal_head_config,
                                            dropout=self.temporal_drop,
                                            residual=self.args.residual,
                                            num_time_steps=self.window_size)
        return node_feature_embedding_layer, position_encoding_clustering_coefficient_layer, position_encoding_bidirectional_links_ratio_layer, structural_layer, temporal_layer









