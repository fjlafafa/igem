import argparse
from infer import infer

if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='PPM model')
    parser.add_argument('--mode', type=int, default=1, choices=[1,2], help='predict mode')
    parser.add_argument('--input', '-i', type=str, default="./example/promoter.csv", help='data file path')
    parser.add_argument('--output', '-o', type=str, default="./example/", help='output dir path')
    parser.add_argument('--model', type=str, default="./model/best_model_1.ckpt", help='model path')
    parser.add_argument('--sort', action="store_true", help='sort output')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # Parse arguments
    Args = parser.parse_args()
    # Run
    print("Start infering...")
    infer(Args)
    print("Done!")
