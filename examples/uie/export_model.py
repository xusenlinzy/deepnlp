import sys

sys.path.append("../..")

import argparse
from torchblocks.tasks.uie import export_onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=Path, required=True,
                        default='./checkpoint/model_best', help="The path to model parameters to be loaded.")
    parser.add_argument("-o", "--output_path", type=Path, default=None,
                        help="The path of model parameter in static graph to be saved.")
    args = parser.parse_args()
    export_onnx_model(args)
