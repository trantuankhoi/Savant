from typing import Any, Tuple

import cv2
import numpy as np
from numba.typed import List

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ComplexModel
from savant.utils.nms import nms_cpu


class YOLOv9Converter(BaseObjectModelOutputConverter):

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.45,
        top_k: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ComplexModel,
        roi: Tuple[float, float, float, float],
    )  -> np.ndarray:
        predictions = output_layers[0]
        
        # Split predictions
        boxes = predictions[:, :4]        # (8400, 4) - x,y,w,h format
        scores = predictions[:, 4]        # (8400,) - confidence scores
        class_ids = predictions[:, 5]     # (8400,) - class ids
        
        # Convert to correct dtypes 
        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask] 
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            # Return empty array with correct shape
            return np.zeros((0, 6), dtype=np.float32)
            
        # Convert class_ids to int
        class_ids = class_ids.astype(np.int32)
        
        if self.nms_iou_threshold > 0 and len(scores) > 1:
            nms_mask = nms_cpu(boxes, scores, self.nms_iou_threshold, self.top_k)
            boxes = boxes[nms_mask]
            class_ids = class_ids[nms_mask]
            scores = scores[nms_mask]
        elif len(scores) > self.top_k:
            top_k_mask = np.argpartition(scores, -self.top_k)[-self.top_k :]
            boxes = boxes[top_k_mask]
            class_ids = class_ids[top_k_mask]
            scores = scores[top_k_mask]
        # breakpoint()
        roi_left, roi_top, roi_width, roi_height = roi

        # scale
        if model.input.maintain_aspect_ratio:
            gain = min(
                model.input.width / roi_width,
                model.input.height / roi_height,
            )
            pad = (
                round((model.input.width - roi_width * gain) / 2 - 0.1),
                round((model.input.height - roi_height * gain) / 2 - 0.1),
            )  # wh padding
            
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding

            boxes[..., :4] /= gain
        else:
            boxes[:, [0, 2]] /= model.input.width / roi_width
            boxes[:, [1, 3]] /= model.input.height / roi_height

        # correct xc, yc
        boxes[:, 0] += roi_left
        boxes[:, 1] += roi_top
        
        return np.concatenate(
            (
                class_ids.reshape(-1, 1).astype(np.float32),
                scores.reshape(-1, 1),
                boxes,
            ),
            axis=1,
        )
    
    def visualize(self, results: np.ndarray) -> np.ndarray:
        image = cv2.imread("/test_data/test_img.jpeg")
        input_w , input_h = 640, 640
        img_h, img_w, _ = image.shape
        r_w = input_w / img_w
        r_h = input_h / img_h
        r = min(r_w, r_h)

        for box in results:
            xc, yc, w, h = [int(i) for i in box]
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite("/opt/savant/src/test_img_result.jpg", image)
