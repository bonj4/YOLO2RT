import argparse
from YOLO2RT.export import seg_exporter

class Seg_exporter:
    name = "export_seg"
    help = "export yolo models to onnx"

    def fundamental_arguments(self, parser: argparse.ArgumentParser):

        group = parser.add_argument_group("export segmentation options")

        group.add_argument('--engine', type=str, help='Engine file')

        group.add_argument('--imgs', type=str, help='Images file')

        group.add_argument('--show',
                            action='store_true',
                            help='Show the detection results')

        group.add_argument('--out-dir',
                            type=str,
                            default='./output',
                            help='Path to output file')

        group.add_argument('--conf-thres',
                            type=float,
                            default=0.25,
                            help='Confidence threshold')

        group.add_argument('--iou-thres',
                            type=float,
                            default=0.65,
                            help='Confidence threshold')

        group.add_argument('--device',
                            type=str,
                            default='cuda:0',
                            help='TensorRT infer device')

    def filter_args(self, args):
        return {k: v for k, v in args.items() if k in seg_exporter.__code__.co_varnames}

    def perform_task(self, vars_args: dict):
        valid_args = self.filter_args(vars_args)
        return seg_exporter(**valid_args)