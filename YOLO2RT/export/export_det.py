from io import BytesIO

import onnx
import torch
from ultralytics import YOLO

from YOLO2RT.models.common import PostDetect, optim

try:
    import onnxsim
except ImportError:
    onnxsim = None

def det_exporter(weights, iou_thres= 0.65, conf_thres= 0.25, topk= 100, opset= 11, sim= True, input_shape= [1, 3, 640, 640], device= 'cuda:0'):
    PostDetect.conf_thres = conf_thres
    PostDetect.iou_thres = iou_thres
    PostDetect.topk = topk
    b = input_shape[0]
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
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=opset,
            input_names=['images'],
            output_names=['num_dets', 'bboxes', 'scores', 'labels'])
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    shapes = [b, 1, b, topk, 4, b, topk, b, topk]
    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))
    if sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')

