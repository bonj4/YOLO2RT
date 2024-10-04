# YOLO2RT
# YOLO2RT

YOLO2RT is a tool for converting YOLO models to TensorRT for optimized inference performance.

## Installation

```bash
pip install .
```

## Usage

```bash
YOLO2RT -h
```

### Object Detection

#### 1. Export from PyTorch to Onnx

```bash
YOLO2RT export_det --weights yolov8s.pt \
    --iou-thres 0.65 \
    --conf-thres 0.25 \
    --topk 100 \
    --opset 11 \
    --sim \
    --input-shape 1 3 640 640 \
    --device cuda:0
```

#### 2. Two-Step Process via ONNX

```bash
YOLO2RT builder \
    --weights yolov8s.onnx \
    --iou-thres 0.65 \
    --conf-thres 0.25 \
    --topk 100 \
    --fp16 \
    --device cuda:0
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Path to the model weights (`.pt` or `.onnx`) | Required |
| `--iou-thres` | IoU threshold for NMS | 0.65 |
| `--conf-thres` | Confidence threshold | 0.25 |
| `--topk` | Maximum detections per image | 100 |
| `--opset` | ONNX opset version (for export_det) | 11 |
| `--sim` | Enable model simplification (for export_det) | False |
| `--input-shape` | Input shape in format: batch channels height width | 1 3 640 640 |
| `--fp16` | Enable FP16 precision (for builder) | False |
| `--device` | Specify the CUDA device | cuda:0 |

## Workflows

1. **Direct Export**
   - Use `export_det` to convert directly from PyTorch (.pt) to TensorRT

2. **Two-Step Process**
   - Convert to ONNX separately
   - Use `builder` to convert from ONNX to TensorRT

## Requirements

- CUDA-capable GPU
- TensorRT
- PyTorch
- ONNX

## 🤗 Citation
```bibtex
 * Adapted from: https://github.com/triple-Mu/YOLOv8-TensorRT.git
 * Author: Original triple-mu
 * License: MIT
 * Accessed on: October 3, 2024
```
