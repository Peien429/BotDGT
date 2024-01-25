import os
import torch
from torch_geometric.loader import NeighborLoader


def load_graphs(dataset_name, interval):
    assert interval in ['year', 'month', 'three_months', 'six_months', '15_months',
                        '18_months', '21_months', '24_months', '9_months']
    assert dataset_name in ['Twibot-20', 'Twibot-22']
    interval_dict = {'year': 12, 'month': 1, 'three_months': 3, 'six_months': 6, '9_months': 9,
                     '18_months': 18, '15_months': 15, '21_months': 21, '24_months': 24}
    files = os.listdir(r"./data/{}/graph_data/graphs".format(dataset_name))
    files = sorted(files)
    file_name = []
    for index in range(-1, -len(files) - 1, -interval_dict[interval]):
        file_name.append(files[index])
    file_name.reverse()
    print(file_name)
    graph_list = [torch.load(r"./data/{}/graph_data/graphs/{}".format(dataset_name, file)) for file in file_name]
    return graph_list, file_name


def load_split_index(dataset_name):
    train_idx = torch.load(r"./data/{}/processed_data/train_idx.pt".format(dataset_name))
    val_idx = torch.load(r"./data/{}/processed_data/val_idx.pt".format(dataset_name))
    test_idx = torch.load(r"./data/{}/processed_data/test_idx.pt".format(dataset_name))
    return train_idx, val_idx, test_idx


def load_labels(dataset_name):
    labels = torch.load(r"./data/{}/processed_data/label.pt".format(dataset_name))
    return labels


def load_features(dataset_name):
    des_tensor = torch.load('./data/{}/processed_data/des_tensor.pt'.format(dataset_name))
    tweets_tensor = torch.load('./data/{}/processed_data/tweets_tensor.pt'.format(dataset_name))
    num_prop = torch.load('./data/{}/processed_data/num_properties_tensor.pt'.format(dataset_name))
    category_prop = torch.load('./data/{}/processed_data/cat_properties_tensor.pt'.format(dataset_name))
    return des_tensor, tweets_tensor, num_prop, category_prop


class Dataset:
    def __init__(self, dataset_name, interval, batch_size, seed, window_size, device):
        super().__init__()
        self.dataset_name = dataset_name
        self.interval = interval
        self.batch_size = batch_size
        self.seed = seed
        self.window_size = window_size
        self.device = device
        self.graphs, self.graphs_file_name_list = load_graphs(dataset_name, interval)

        self.train_idx, self.val_idx, self.test_idx = load_split_index(dataset_name)
        self.labels = load_labels(dataset_name)
        self.des_tensor, self.tweets_tensor, self.num_prop, self.category_prop = load_features(dataset_name)

        self.graphs = [graph.to(self.device) for graph in self.graphs]
        self.train_idx = self.train_idx.to(self.device)
        self.val_idx = self.val_idx.to(self.device)
        self.test_idx = self.test_idx.to(self.device)
        self.labels = self.labels.to(self.device)
        self.des_tensor = self.des_tensor.to(self.device)
        self.tweets_tensor = self.tweets_tensor.to(self.device)
        self.num_prop = self.num_prop.to(self.device)
        self.category_prop = self.category_prop.to(self.device)
        self.train_right, self.train_n_id, self.train_edge_index, self.train_edge_type, self.train_exist_nodes, self.train_clustering_coefficient, self.train_bidirectional_links_ratio = self.get_final_data(
            type='train')
        self.val_right, self.val_n_id, self.val_edge_index, self.val_edge_type, self.val_exist_nodes, self.val_clustering_coefficient, self.val_bidirectional_links_ratio = self.get_final_data(
            type='val')
        self.test_right, self.test_n_id, self.test_edge_index, self.test_edge_type, self.test_exist_nodes, self.test_clustering_coefficient, self.test_bidirectional_links_ratio = self.get_final_data(
            type='test')
        if len(self.graphs) > self.window_size and self.window_size != -1:
            print('window size is smaller than the number of snapshots, keep the last {} snapshots'.format(
                self.window_size))
            self.get_window_data()
        else:
            print(
                'window size is larger than the number of snapshots or window size is set to "-1", keep all snapshots')
            self.window_size = len(self.graphs)

    def get_window_data(self):
        attrs = [
            'train_n_id', 'train_edge_index', 'train_edge_type', 'train_exist_nodes',
            'train_clustering_coefficient', 'train_bidirectional_links_ratio',
            'test_n_id', 'test_edge_index', 'test_edge_type', 'test_exist_nodes',
            'test_clustering_coefficient', 'test_bidirectional_links_ratio',
            'val_n_id', 'val_edge_index', 'val_edge_type', 'val_exist_nodes',
            'val_clustering_coefficient', 'val_bidirectional_links_ratio'
        ]

        for attr in attrs:
            setattr(self, attr, [_[-self.window_size:] for _ in getattr(self, attr)])

    def get_final_data(self, type):
        print('getting final {} data for {}'.format(type, self.dataset_name))
        input_nodes = None
        if type == 'train':
            input_nodes = self.train_idx
            shuffle = True
        elif type == 'val':
            input_nodes = self.val_idx
            shuffle = False
        elif type == 'test':
            input_nodes = self.test_idx
            shuffle = False
        else:
            raise Exception('type error')
        dir_path = r'./data/{}/final_data/{}/batch-size-{}/seed-{}/{}'.format(self.dataset_name, self.interval,
                                                                              self.batch_size, self.seed, type)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_names = ['all_right', 'all_n_id', 'all_edge_index', 'all_edge_type', 'all_exist_nodes',
                      'all_clustering_coefficient', 'all_bidirectional_links_ratio']
        data_dict = {name: {'path': os.path.join(dir_path, f'{name}.pt'), 'data': []} for name in file_names}
        if all([os.path.exists(data_dict[name]['path']) for name in data_dict]):
            for name in data_dict:
                data_dict[name]['data'] = torch.load(data_dict[name]['path'])
        else:
            loader = dataLoader(graphs=self.graphs, input_nodes=input_nodes, batch_size=self.batch_size,
                                shuffle=shuffle, seed=self.seed, type=type)
            total_nodes_num = len(input_nodes)
            batch_size = self.batch_size
            for i in range(0, total_nodes_num, batch_size):
                right = min(batch_size, total_nodes_num - i)
                data_dict['all_right']['data'].append(right)
                subgraph_list = loader.iterate()
                batch_n_id = [subgraph.n_id for subgraph in subgraph_list]
                batch_edge_index = [subgraph.edge_index for subgraph in subgraph_list]
                batch_edge_type = [subgraph.edge_type for subgraph in subgraph_list]
                batch_exist_nodes = [subgraph.exist_nodes for subgraph in subgraph_list]
                batch_clustering_coefficient = [subgraph.clustering_coefficient for subgraph in subgraph_list]
                batch_bidirectional_links_ratio = [subgraph.bidirectional_links_ratio for subgraph in subgraph_list]
                data_dict['all_n_id']['data'].append(batch_n_id)
                data_dict['all_edge_index']['data'].append(batch_edge_index)
                data_dict['all_edge_type']['data'].append(batch_edge_type)
                data_dict['all_exist_nodes']['data'].append(batch_exist_nodes)
                data_dict['all_clustering_coefficient']['data'].append(batch_clustering_coefficient)
                data_dict['all_bidirectional_links_ratio']['data'].append(batch_bidirectional_links_ratio)
            for name in data_dict:
                torch.save(data_dict[name]['data'], data_dict[name]['path'])

        return data_dict['all_right']['data'], \
            data_dict['all_n_id']['data'], \
            data_dict['all_edge_index']['data'], \
            data_dict['all_edge_type']['data'], \
            data_dict['all_exist_nodes']['data'], \
            data_dict['all_clustering_coefficient']['data'], \
            data_dict['all_bidirectional_links_ratio']['data']


class dataLoader():
    def __init__(self, graphs, input_nodes, seed, batch_size, shuffle, type):
        super().__init__()
        num_neighbors = [2560] * 2 if type == 'train' else [-1] * 2
        self.loader_list = [
            NeighborLoader(graph, shuffle=shuffle, generator=torch.Generator().manual_seed(seed), batch_size=batch_size,
                           input_nodes=input_nodes, num_neighbors=num_neighbors)
            for graph in graphs]
        self.iter_list = [iter(loader) for loader in self.loader_list]

    def iterate(self):
        try:
            return [next(_) for _ in self.iter_list]
        except StopIteration:
            return None
