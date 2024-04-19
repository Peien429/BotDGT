import argparse


def parse_train_args(parser):
    parser.add_argument('--dataset_name', type=str, default='Twibot-20', help='dataset name')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--interval', type=str, default='year', help='Interval of snapshots')
    parser.add_argument("--early_stop", action='store_true', help="whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=10, help='Patience')
    parser.add_argument('--coefficient', type=float, default=1.1)
    parser.add_argument('--temporal_head_config', type=int, default=4, help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--structural_head_config', type=int, default=4, help='Encoder layer config: # attention heads in each Structural layer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (# nodes)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--temporal_drop', type=float, default=0.5, help='Temporal Layer Dropout (1 - keep probability).')
    parser.add_argument('--structural_drop', type=float, default=0, help='Structural Layer Dropout (1 - keep probability).')
    parser.add_argument('--structural_learning_rate', type=float, default=1e-4, help='Initial learning rate for structural model.')
    parser.add_argument('--temporal_learning_rate', type=float, default=1e-5, help='Initial learning rate for temporal model.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--epoch', type=int, default=20, help='# epochs')
    parser.add_argument('--window_size', type=int, default=-1, help='Window size')
    parser.add_argument('--temporal_module_type', type=str, default='attention', choices=['attention', 'gru', 'lstm'], help='Temporal module type')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args
