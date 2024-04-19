import os.path
from copy import deepcopy
import torch
from config import get_train_args
from utils.loss import all_snapshots_loss
from utils.metrics import is_better, null_metrics, compute_metrics_one_snapshot
from models.model import BotDyGNN
from utils.dataset import Dataset
from pytorch_lightning import seed_everything


class Trainer:
    def __init__(self, args):
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dataset = Dataset(self.args.dataset_name, self.args.interval, self.args.batch_size, self.args.seed,
                               self.args.window_size, self.args.device)
        self.des_tensor, self.tweets_tensor, self.num_prop, self.category_prop, self.labels = self.dataset.des_tensor, self.dataset.tweets_tensor, self.dataset.num_prop, self.dataset.category_prop, self.dataset.labels
        self.train_right, self.train_n_id, self.train_edge_index, self.train_edge_type, self.train_exist_nodes, self.train_clustering_coefficient, self.train_bidirectional_links_ratio = self.dataset.train_right, self.dataset.train_n_id, self.dataset.train_edge_index, self.dataset.train_edge_type, self.dataset.train_exist_nodes, self.dataset.train_clustering_coefficient, self.dataset.train_bidirectional_links_ratio
        self.test_right, self.test_n_id, self.test_edge_index, self.test_edge_type, self.test_exist_nodes, self.test_clustering_coefficient, self.test_bidirectional_links_ratio = self.dataset.test_right, self.dataset.test_n_id, self.dataset.test_edge_index, self.dataset.test_edge_type, self.dataset.test_exist_nodes, self.dataset.test_clustering_coefficient, self.dataset.test_bidirectional_links_ratio
        self.val_right, self.val_n_id, self.val_edge_index, self.val_edge_type, self.val_exist_nodes, self.val_clustering_coefficient, self.val_bidirectional_links_ratio = self.dataset.val_right, self.dataset.val_n_id, self.dataset.val_edge_index, self.dataset.val_edge_type, self.dataset.val_exist_nodes, self.dataset.val_clustering_coefficient, self.dataset.val_bidirectional_links_ratio
        if self.args.dataset_name == 'Twibot-20':
            self.labels = torch.cat(
                (self.labels, 3 * torch.ones(229580 - len(self.labels), device=self.args.device).long()), dim=0)
        self.args.window_size = self.dataset.window_size
        self.model = BotDyGNN(self.args)
        self.model.to(self.args.device)
        params = [
            {"params": self.model.node_feature_embedding_layer.parameters(), "lr": self.args.structural_learning_rate},
            {"params": self.model.structural_layer.parameters(), "lr": self.args.structural_learning_rate},
            {"params": self.model.temporal_layer.parameters(), "lr": self.args.temporal_learning_rate},
        ]
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0)
        self.pbar = range(self.args.epoch)
        self.best_val_metrics = null_metrics()
        self.test_state_dict_list = []
        self.test_epoch = None
        self.test_epoch_list = []
        self.test_metrics = null_metrics()
        self.test_state_dict = None
        self.last_state_dict = None

    def forward_one_batch(self, batch_size, batch_n_id, batch_edge_index, batch_exist_nodes,
                          batch_clustering_coefficient, batch_bidirectional_links_ratio):
        des_tensor_list = [self.des_tensor[n_id].to(self.args.device) for n_id in batch_n_id]
        tweet_tensor_list = [self.tweets_tensor[n_id].to(self.args.device) for n_id in batch_n_id]
        num_prop_list = [self.num_prop[n_id].to(self.args.device) for n_id in batch_n_id]
        category_prop_list = [self.category_prop[n_id].to(self.args.device) for n_id in batch_n_id]
        label_list = [self.labels[n_id][:batch_size].to(self.args.device) for n_id in batch_n_id]
        label_list = torch.stack(label_list, dim=0)
        edge_index_list = [_.to(self.args.device) for _ in batch_edge_index]
        clustering_coefficient_list = [_.to(self.args.device) for _ in batch_clustering_coefficient]
        bidirectional_links_ratio_list = [_.to(self.args.device) for _ in batch_bidirectional_links_ratio]
        exist_nodes_list = [exist_nodes[:batch_size].to(self.args.device) for exist_nodes in batch_exist_nodes]
        exist_nodes_list = torch.stack(exist_nodes_list, dim=0)
        output = self.model(des_tensor_list, tweet_tensor_list, num_prop_list, category_prop_list, edge_index_list,
                               clustering_coefficient_list, bidirectional_links_ratio_list, exist_nodes_list,
                               batch_size)
        output = output.transpose(0, 1)
        loss = all_snapshots_loss(self.criterion, output, label_list, exist_nodes_list)
        return output, loss, label_list, exist_nodes_list

    def forward_one_epoch(self, right, n_id, edge_index, exist_nodes, clustering_coefficient,
                          bidirectional_links_ratio):
        all_label = []
        all_output = []
        all_exist_nodes = []
        total_loss = 0.0
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, batch_clustering_coefficient, batch_bidirectional_links_ratio \
                in zip(right, n_id, edge_index, exist_nodes, clustering_coefficient, bidirectional_links_ratio):
            output, loss, label_list, exist_nodes_list = self.forward_one_batch(batch_size, batch_n_id,
                                                                                batch_edge_index, batch_exist_nodes,
                                                                                batch_clustering_coefficient,
                                                                                batch_bidirectional_links_ratio)
            total_loss += loss.item() / self.args.window_size / len(right)
            all_output.append(output)
            all_label.append(label_list)
            all_exist_nodes.append(exist_nodes_list)
        all_output = torch.cat(all_output, dim=1)
        all_label = torch.cat(all_label, dim=1)
        all_exist_nodes = torch.cat(all_exist_nodes, dim=1)
        metrics = compute_metrics_one_snapshot(all_label[-1], all_output[-1], exist_nodes=all_exist_nodes[-1])
        metrics['loss'] = total_loss
        return metrics

    def train_per_epoch(self, current_epoch):
        self.model.train()
        all_label = []
        all_output = []
        all_exist_nodes = []
        total_loss = 0.0
        plog = ""
        for batch_size, batch_n_id, batch_edge_index, batch_exist_nodes, batch_clustering_coefficient, batch_bidirectional_links_ratio \
                in zip(self.train_right, self.train_n_id, self.train_edge_index, self.train_exist_nodes,
                       self.train_clustering_coefficient, self.train_bidirectional_links_ratio):
            self.optimizer.zero_grad()
            output, loss, label_list, exist_nodes_list = self.forward_one_batch(batch_size, batch_n_id,
                                                                                batch_edge_index, batch_exist_nodes,
                                                                                batch_clustering_coefficient,
                                                                                batch_bidirectional_links_ratio)
            total_loss += loss.item() / self.args.window_size / len(self.train_right)
            loss.backward()
            self.optimizer.step()
            all_output.append(output)
            all_label.append(label_list)
            all_exist_nodes.append(exist_nodes_list)
        all_output = torch.cat(all_output, dim=1)
        all_label = torch.cat(all_label, dim=1)
        all_exist_nodes = torch.cat(all_exist_nodes, dim=1)
        metrics = compute_metrics_one_snapshot(all_label[-1], all_output[-1], exist_nodes=all_exist_nodes[-1])
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            plog += ' {}: {:.6}'.format(key, metrics[key])
        plog = 'Epoch-{} train loss: {:.6}'.format(current_epoch, total_loss) + plog
        print(plog)
        metrics['loss'] = total_loss
        return metrics

    @torch.no_grad()
    def val_per_epoch(self, current_epoch):
        self.model.eval()
        metrics = self.forward_one_epoch(self.val_right, self.val_n_id, self.val_edge_index, self.val_exist_nodes,
                                         self.val_clustering_coefficient, self.val_bidirectional_links_ratio)
        plog = ""
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            plog += ' {}: {:.6}'.format(key, metrics[key])
        plog = 'Epoch-{} val loss: {:.6}'.format(current_epoch, metrics['loss']) + plog
        print(plog)
        return metrics

    @torch.no_grad()
    def test_last_model(self):
        self.model.load_state_dict(self.last_state_dict)
        self.model.eval()
        metrics = self.forward_one_epoch(self.test_right, self.test_n_id, self.test_edge_index, self.test_exist_nodes,
                                         self.test_clustering_coefficient, self.test_bidirectional_links_ratio)
        plog = ""
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            plog += ' {}: {:.6}'.format(key, metrics[key])
        plog = 'Last Epoch test loss: {:.6}'.format(metrics['loss']) + plog
        print(plog)
        return metrics, self.last_state_dict

    @torch.no_grad()
    def test_best_model(self, top_k=1):
        best_test_metrics = null_metrics()
        best_test_state_dict = None
        self.test_state_dict_list = self.test_state_dict_list[-top_k:]
        self.test_epoch_list = self.test_epoch_list[-top_k:]
        print('start testing...')
        for epoch, state_dict in zip(self.test_epoch_list, self.test_state_dict_list):
            self.model.load_state_dict(state_dict)
            self.model.eval()
            metrics = self.forward_one_epoch(self.test_right, self.test_n_id, self.test_edge_index,
                                             self.test_exist_nodes, self.test_clustering_coefficient,
                                             self.test_bidirectional_links_ratio)
            plog = ""
            for key in ['accuracy', 'precision', 'recall', 'f1']:
                plog += ' {}: {:.6}'.format(key, metrics[key])
            plog = 'Epoch-{} test loss: {:.6}'.format(epoch, metrics['loss']) + plog
            print(plog)
            if is_better(metrics, best_test_metrics):
                best_test_metrics = metrics
                best_test_state_dict = state_dict
        return best_test_metrics, best_test_state_dict

    def train(self):
        validate_score_non_improvement_count = 0
        self.model.train()
        for current_epoch in self.pbar:
            self.train_per_epoch(current_epoch)
            self.scheduler.step()
            val_metrics = self.val_per_epoch(current_epoch)
            if is_better(val_metrics, self.best_val_metrics):
                self.best_val_metrics = val_metrics
                self.test_epoch = current_epoch
                self.test_epoch_list.append(current_epoch)
                self.test_state_dict = deepcopy(self.model.state_dict())
                self.test_state_dict_list.append(self.test_state_dict)
                validate_score_non_improvement_count = 0
            else:
                validate_score_non_improvement_count += 1
            self.last_state_dict = deepcopy(self.model.state_dict())
            if self.args.early_stop and validate_score_non_improvement_count >= self.args.patience:
                print('Early stopping at epoch: {}'.format(current_epoch))
                break
        if self.args.early_stop:
            self.test_metrics, self.test_state_dict = self.test_best_model(top_k=1)
        else:
            self.test_metrics, self.test_state_dict = self.test_last_model()
        model_name = f"{self.args.interval} + {self.args.seed} + {self.test_metrics['accuracy']} + {self.test_metrics['f1']}.pt "
        model_dir_path = os.path.join('output', self.args.dataset_name)
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        torch.save(self.test_state_dict, os.path.join(model_dir_path, model_name))


def main(args):
    seed_everything(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = get_train_args()
    print(args)
    main(args)
