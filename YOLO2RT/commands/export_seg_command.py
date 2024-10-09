import argparse
from YOLO2RT.export import seg_exporter

class Seg_exporter:
    name = "export_seg"
    help = "export yolo models to onnx"

    def fundamental_arguments(self, parser: argparse.ArgumentParser):

        group = parser.add_argument_group("export segmentation options")

        group.add_argument('-w',
                            '--weights',
                            type=str,
                            required=True,
                            help='PyTorch yolov8 weights')
        group.add_argument('--opset',
                            type=int,
                            default=11,
                            help='ONNX opset version')
        group.add_argument('--sim',
                            action='store_true',
                            help='simplify onnx model')
        group.add_argument('--input-shape',
                            nargs='+',
                            type=int,
                            default=[1, 3, 640, 640],
                            help='Model input shape only for api builder')
        group.add_argument('--device',
                            type=str,
                            default='cpu',
                            help='Export ONNX device')

    def filter_args(self, args):
        return {k: v for k, v in args.items() if k in seg_exporter.__code__.co_varnames}

    def perform_task(self, vars_args: dict):
        valid_args = self.filter_args(vars_args)
        return seg_exporter(**valid_args)