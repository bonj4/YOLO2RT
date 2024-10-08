from pathlib import Path

import cv2
import numpy as np
import torch

from YOLO2RT.inferences.config import ALPHA, CLASSES_SEG, COLORS, MASK_COLORS
from YOLO2RT.models import TRTModule  # isort:skip
from YOLO2RT.models.torch_utils import seg_postprocess
from YOLO2RT.models.utils import blob, letterbox, path_to_list


def inferseg(imgs, engine, device, out_dir, show) -> None:
    device = torch.device(device)
    Engine = TRTModule(engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['outputs', 'proto'])

    images = path_to_list(imgs)
    save_path = Path(out_dir)

    if not show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, seg_img = blob(rgb, return_seg=True)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)

        seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]],
                                device=device)
        bboxes, scores, labels, masks = seg_postprocess(
            data, bgr.shape[:2])
        if bboxes.numel() == 0:
            # if no bounding box
            print(f'{image}: no object!')
            continue
        masks = masks[:, dh:H - dh, dw:W - dw, :]
        indices = (labels % len(MASK_COLORS)).long()
        mask_colors = torch.asarray(MASK_COLORS, device=device)[indices]
        mask_colors = mask_colors.view(-1, 1, 1, 3) * ALPHA
        mask_colors = masks @ mask_colors
        inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
        draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8),
                          draw.shape[:2][::-1])

        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES_SEG[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        if show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)
