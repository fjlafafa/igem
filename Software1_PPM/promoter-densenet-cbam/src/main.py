import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Densenet Cbam model')
    # Data options
    parser.add_argument('--data_dir', type=str, default="./data/", help='data directory')
    parser.add_argument('--log_dir', type=str, default="./lightning_logs/", help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint')
    parser.add_argument('--cuda', type=int, default=0, help='GPU index')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    # Training Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--epoch', type=int, default=120, help='number of epochs to train')
    # Model parameters
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--growth_rate', type=int, default=32, help='growth rate')
    parser.add_argument('--bn_size', type=int, default=2, help='the factor using in the bottleneck layer')
    parser.add_argument('--compression_rate', type=float, default=0.5, help='the compression rate used in Transition Layer')
    parser.add_argument('--dropout', type=float, default=0.15, help='dropout rate')
    # Parse arguments
    Args = parser.parse_args()
    # Train
    train(Args)