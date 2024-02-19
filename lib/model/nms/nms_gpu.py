import torch
import torchvision.ops as ops

def nms_gpu(dets, thresh):
    # Convert dets to [x1, y1, x2, y2] format
    boxes = dets[:, :4]
    scores = dets[:, 4]

    # Apply NMS
    keep = ops.nms(boxes, scores, iou_threshold=thresh)

    return keep
