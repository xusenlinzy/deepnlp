import sys

sys.path.append("../..")

import argparse
from torchblocks.tasks.uie import do_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="The path of saved model that you want to load.")
    parser.add_argument("-t", "--test_path", type=str, required=True,
                        help="The path of test set.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("-D", '--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to run model, defaults to gpu.")

    args = parser.parse_args()

    do_eval(args)
