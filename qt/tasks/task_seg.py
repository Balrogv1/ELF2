import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np

from task_base import BaseVisionTask, TaskResult


class YoloSegTask(BaseVisionTask):
    name = "YOLOv8 Segmentation"

    def __init__(self):
        self.rknn = None
        self.seg = None
        self.model_path = None
        self.no_mask = False
        self.mask_alpha = 0.45

    def open(self, config):
        self.seg = self._load_source_module()
        params = config.get("task_params", {})
        self.model_path = params.get("model_path") or self._default_model_path()
        core = params.get("core") or "auto"
        self.no_mask = str(params.get("no_mask", "false")).lower() in ("1", "true", "yes", "on")
        self.mask_alpha = float(params.get("mask_alpha") or 0.45)
        self.rknn = self.seg.init_runtime(self.model_path, core)

    def process(self, frame):
        img, info = self.seg.letter_box(
            frame.copy(),
            new_shape=(self.seg.MODEL_SIZE[1], self.seg.MODEL_SIZE[0]),
        )
        ratio, pad_offset, resized_shape = info
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0)
        input_data = np.ascontiguousarray(input_data)

        infer_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data])
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        boxes, classes, scores, seg_img = self.seg.post_process(outputs)
        show_frame = frame.copy()
        det_num = 0
        if boxes is not None:
            det_num = len(boxes)
            real_boxes = self.seg.scale_boxes(boxes, ratio, pad_offset, frame.shape)
            if self.no_mask:
                rendered = show_frame
            else:
                real_masks = self.seg.scale_masks(seg_img, pad_offset, resized_shape, frame.shape)
                rendered = self.seg.merge_seg(show_frame, real_masks, classes, alpha=self.mask_alpha)
            self.seg.draw_seg(rendered, real_boxes, scores, classes)
            show_frame = rendered
        post_ms = (time.perf_counter() - post_start) * 1000.0

        return TaskResult(
            frame_bgr=show_frame,
            metrics={"infer_ms": infer_ms, "post_ms": post_ms, "detections": det_num},
            status_text="Detections: {}".format(det_num),
        )

    def close(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def _load_source_module(self):
        source_path = self._find_source_path()
        spec = importlib.util.spec_from_file_location("elf2_yolov8_seg", str(source_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _find_source_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "yolov8_seg" / "yolo_seg_camera.py",
            Path("/home/elf/my_yolo/yolov8_seg/yolo_seg_camera.py"),
            Path("/home/elf/code/yolov8_seg/yolo_seg_camera.py"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Cannot find yolo_seg_camera.py. Tried: {}".format(
                ", ".join(str(candidate) for candidate in candidates)
            )
        )

    def _default_model_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "yolov8_seg" / "yolov8n-seg.rknn",
            Path("/home/elf/my_yolo/yolov8_seg/yolov8n-seg.rknn"),
            Path("/home/elf/code/yolov8_seg/yolov8n-seg.rknn"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])