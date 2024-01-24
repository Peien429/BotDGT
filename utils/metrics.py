import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def null_metrics():
    return {
        'accuracy': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }


def compute_metrics_one_snapshot(y_true, y_output, exist_nodes):
    metrics = null_metrics()
    if torch.any(torch.isnan(y_output)):
        metrics['msg'] = 'NaN in y_output'
        print('error!')
    else:
        y_pred = F.softmax(y_output, dim=-1)
        if exist_nodes != 'all':
            y_pred = y_pred[torch.where(exist_nodes == 1)]
            y_true = y_true[torch.where(exist_nodes == 1)]
        else:
            pass
        y_true = y_true.to('cpu').detach().numpy()
        y_pred_label = torch.argmax(y_pred, dim=-1).to('cpu').detach().numpy()
        metrics['msg'] = 'success'
        metrics['accuracy'] = round(accuracy_score(y_true, y_pred_label), 5)
        metrics['f1'] = round(f1_score(y_true, y_pred_label), 5)
        metrics['precision'] = round(precision_score(y_true, y_pred_label), 5)
        metrics['recall'] = round(recall_score(y_true, y_pred_label), 5)
    return metrics


def compute_metrics_all_snapshots(y_true, y_output, exist_nodes):
    all_metrics = []
    snapshots_num = y_true.shape[0]
    for i in range(snapshots_num):
        if torch.all(exist_nodes[i] == 0):
            metrics = null_metrics()
            metrics['msg'] = 'no exist nodes'
        else:
            metrics = compute_metrics_one_snapshot(y_true[i], y_output[i], exist_nodes[i])
        all_metrics.append(metrics)

    return all_metrics


def is_better(now, pre):
    if now['accuracy'] > pre['accuracy']:
        return True
    elif now['accuracy'] < pre['accuracy']:
        return False
    else:
        if now['f1'] >= pre['f1']:
            return True
        elif now['f1'] < pre['f1']:
            return False