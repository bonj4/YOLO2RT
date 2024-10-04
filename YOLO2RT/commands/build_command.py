import argparse
from YOLO2RT.export import builder

class Builder:
    name = "build"
    help = "export yolo onnx models to engine"

    def fundamental_arguments(self, parser: argparse.ArgumentParser):

        group = parser.add_argument_group("export detection options")

        group.add_argument('--weights',
                            type=str,
                            required=True,
                            help='Weights file')
        group.add_argument('--iou-thres',
                            type=float,
                            default=0.65,
                            help='IOU threshoud for NMS plugin')
        group.add_argument('--conf-thres',
                            type=float,
                            default=0.25,
                            help='CONF threshoud for NMS plugin')
        group.add_argument('--topk',
                            type=int,
                            default=100,
                            help='Max number of detection bboxes')
        group.add_argument('--input-shape',
                            nargs='+',
                            type=int,
                            default=[1, 3, 640, 640],
                            help='Model input shape only for api builder')
        group.add_argument('--fp16',
                            action='store_true',
                            help='Build model with fp16 mode')
        group.add_argument('--device',
                            type=str,
                            default='cuda:0',
                            help='TensorRT builder device')
        group.add_argument('--seg',
                            action='store_true',
                            help='Build seg model by onnx')

    def filter_args(self, args):
        return {k: v for k, v in args.items() if k in builder.__code__.co_varnames}

    def perform_task(self, vars_args: dict):
        valid_args = self.filter_args(vars_args)
        return builder(**valid_args)