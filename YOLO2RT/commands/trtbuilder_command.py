import argparse
from YOLO2RT.export.build_w_trtexec import convert_onnx_to_engine

class TrtBuilder:
    name = "TrtBuilder"
    help = "export yolo onnx models to engine by Trtexec"

    def fundamental_arguments(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("export detection options")

        group.add_argument('--weights',
                           type=str,
                           required=True,
                           help='Weights file')

        group.add_argument('--fp16',
                           action='store_true',
                           help='Build model with fp16 mode')

        group.add_argument('--batch-size',
                           type=int,
                           default=1,
                           help='Max batch size for TensorRT engine')
        group.add_argument('--workspace-size',
                           type=int,
                           default=4096,
                           help='Workspace size in MiB')
        group.add_argument('--verbose',
                           action='store_true',
                           help='Enable verbose logging')

        group.add_argument('--builder-optimization',
                           type=int,
                           choices=[0, 1, 2, 3, 4, 5],
                           default=3,
                           help='TensorRT builder optimization level')

    def filter_args(self, args):
        return {k: v for k, v in args.items() if k in convert_onnx_to_engine.__code__.co_varnames}

    def perform_task(self, vars_args: dict):
        valid_args = self.filter_args(vars_args)
        return convert_onnx_to_engine(**valid_args)