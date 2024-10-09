import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO

from YOLO2RT.models.common import optim

try:
    import onnxsim
except ImportError:
    onnxsim = None

def seg_exporter(weights, opset= 11, sim= True, input_shape= [1, 3, 640, 640], device= 'cuda:0'):
    YOLOv8 = YOLO(weights)
    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(device)
    model.to(device)
    fake_input = torch.randn(input_shape).to(device)
    for _ in range(2):
        model(fake_input)
    save_path = weights.replace('.pt', '.onnx')
    with BytesIO() as f:
        torch.onnx.export(model,
                          fake_input,
                          f,
                          opset_version=opset,
                          input_names=['images'],
                          output_names=['outputs', 'proto'])
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')

