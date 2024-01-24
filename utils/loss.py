import torch


def one_snapshot_loss(criterion, output, label, exist_nodes):
    output = output[torch.where(exist_nodes == 1)]
    label = label[torch.where(exist_nodes == 1)]
    loss = criterion(output, label)
    return loss


def all_snapshots_loss(criterion, output, label, exist_nodes, coefficient=1.1):
    snapshot_num = output.shape[0]
    loss_coefficient = [coefficient ** i for i in range(snapshot_num)]
    total_loss = 0
    for i in range(snapshot_num):
        if torch.all(exist_nodes[i] == 0):
            continue
        loss = one_snapshot_loss(criterion, output[i], label[i], exist_nodes[i])
        total_loss += loss * loss_coefficient[i]
    return total_loss
