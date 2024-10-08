import argparse
from YOLO2RT.inferences.infer_det import inferdet

class InferDet:
    name = "detection"
    help = "The detection inference class"

    def fundamental_arguments(self, parser: argparse.ArgumentParser):

        group = parser.add_argument_group("export detection options")

        group.add_argument('--engine', type=str, help='Engine file')
        group.add_argument('--imgs', type=str, help='Images file')
        group.add_argument('--show',
                            action='store_true',
                            help='Show the detection results')
        group.add_argument('--out-dir',
                            type=str,
                            default='./output',
                            help='Path to output file')
        group.add_argument('--device',
                            type=str,
                            default='cuda:0',
                            help='TensorRT infer device')

    def filter_args(self, args):
        return {k: v for k, v in args.items() if k in inferdet.__code__.co_varnames}

    def perform_task(self, vars_args: dict):
        valid_args = self.filter_args(vars_args)
        return inferdet(**valid_args)