import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np

from task_base import BaseVisionTask, TaskResult


class YoloTask(BaseVisionTask):
    name = "YOLOv8 Detection"

    def __init__(self):
        self.rknn = None
        self.yolo = None
        self.model_path = None

    def open(self, config):
        self.yolo = self._load_source_module()
        task_params = config.get("task_params", {})
        self.model_path = task_params.get("model_path") or self._default_model_path()

        self.rknn = self.yolo.RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            self.rknn.release()
            self.rknn = None
            raise RuntimeError("Load YOLO RKNN model failed: {}".format(self.model_path))

        ret = self.rknn.init_runtime(core_mask=self.yolo.RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            self.rknn.release()
            self.rknn = None
            raise RuntimeError("Init YOLO RKNN runtime failed")

    def process(self, frame):
        img, info = self.yolo.letter_box(
            im=frame.copy(),
            new_shape=(self.yolo.MODEL_SIZE[1], self.yolo.MODEL_SIZE[0]),
            pad_color=(0, 0, 0),
        )
        ratio, pad_offset = info
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0)
        input_data = np.ascontiguousarray(input_data)

        infer_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data])
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        boxes, classes, scores = self.yolo.post_process(outputs)
        show_frame = frame.copy()
        det_num = 0
        if boxes is not None:
            det_num = len(boxes)
            self.yolo.draw(show_frame, boxes, scores, classes, ratio, pad_offset)

        return TaskResult(
            frame_bgr=show_frame,
            metrics={"infer_ms": infer_ms, "detections": det_num},
            status_text="Detections: {}".format(det_num),
        )

    def close(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def _load_source_module(self):
        source_path = self._find_source_path()
        spec = importlib.util.spec_from_file_location("elf2_yolo_camera", str(source_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _find_source_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "yolo_python" / "usb_yolov8_camera.py",
            root / "yolov8-python" / "usb_yolov8_camera.py",
            Path("/home/elf/code/yolo_python/usb_yolov8_camera.py"),
            Path("/home/elf/code/yolov8-python/usb_yolov8_camera.py"),
            Path("/home/elf/yolo_python/usb_yolov8_camera.py"),
            Path("/home/elf/yolov8-python/usb_yolov8_camera.py"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Cannot find usb_yolov8_camera.py. Tried: {}".format(
                ", ".join(str(candidate) for candidate in candidates)
            )
        )

    def _default_model_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "model" / "yolov8n.rknn",
            Path("/home/elf/model/yolov8n.rknn"),
            root.parent / "model" / "yolov8n.rknn",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])
