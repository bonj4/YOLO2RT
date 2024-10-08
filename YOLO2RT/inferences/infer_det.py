from YOLO2RT.models import TRTModule  # isort:skip
from pathlib import Path

import cv2
import torch

from YOLO2RT.inferences.config import CLASSES_DET, COLORS
from YOLO2RT.models.torch_utils import det_postprocess
from YOLO2RT.models.utils import blob, letterbox, path_to_list


def inferdet(imgs,engine,device,out_dir,show) -> None:
    device = torch.device(device)
    Engine = TRTModule(engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images = path_to_list(imgs)
    save_path = Path(out_dir)
    if not show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.numel() == 0:
            # if no bounding box
            print(f'{image}: no object!')
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES_DET[cls_id]
            color = COLORS[cls]

            text = f'{cls}:{score:.3f}'
            x1, y1, x2, y2 = bbox

            (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8, 1)
            _y1 = min(y1 + 1, draw.shape[0])

            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(draw, (x1, _y1), (x1 + _w, _y1 + _h + _bl),
                          (0, 0, 255), -1)
            cv2.putText(draw, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)

        if show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)

