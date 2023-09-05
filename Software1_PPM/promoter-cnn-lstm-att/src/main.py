import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN-LSTM-Attention model')
    # Data options
    parser.add_argument('--data_dir', type=str, default="./data/", help='data directory')
    parser.add_argument('--log_dir', type=str, default="./lightning_logs/", help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint')
    parser.add_argument('--cuda', type=int, default=0, help='GPU index')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    # Training Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--epoch', type=int, default=120, help='number of epochs to train')
    # Model parameters
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--embedding', type=int, default=200, help='embedding size')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout rate')
    # CNN parameters
    parser.add_argument('--kernel_1', type=int, default=5, help='kernel size')
    parser.add_argument('--kernel_2', type=int, default=3, help='kernel size')
    parser.add_argument('--stride', type=int, default=1, help='stride size')
    parser.add_argument('--filter', type=int, default=150, help='filter number')
    parser.add_argument('--pool', type=int, default=2, help='pool kernel size')
    # LSTM parameters
    parser.add_argument('--input', type=int, default=400, help='input size')
    parser.add_argument('--hidden', type=int, default=64, help='hidden size')
    parser.add_argument('--layers', type=int, default=1, help='number of layers')
    # Parse arguments
    Args = parser.parse_args()
    # Train
    train(Args)