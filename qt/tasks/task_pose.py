import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np

from task_base import BaseVisionTask, TaskResult


class YoloPoseTask(BaseVisionTask):
    name = "YOLOv8 Pose"

    def __init__(self):
        self.rknn = None
        self.pose = None
        self.model_path = None

    def open(self, config):
        self.pose = self._load_source_module()
        params = config.get("task_params", {})
        self.model_path = params.get("model_path") or self._default_model_path()
        core = params.get("core") or "auto"
        self.rknn = self.pose.init_runtime(self.model_path, core)

    def process(self, frame):
        img, info = self.pose.letter_box(
            frame.copy(),
            new_shape=(self.pose.MODEL_SIZE[1], self.pose.MODEL_SIZE[0]),
        )
        ratio, pad_offset = info
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0)
        input_data = np.ascontiguousarray(input_data)

        infer_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_data])
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        pred_boxes = self.pose.post_process(outputs)
        show_frame = frame.copy()
        self.pose.draw_pose(show_frame, pred_boxes, ratio, pad_offset)
        post_ms = (time.perf_counter() - post_start) * 1000.0

        person_num = len(pred_boxes)
        return TaskResult(
            frame_bgr=show_frame,
            metrics={"infer_ms": infer_ms, "post_ms": post_ms, "persons": person_num},
            status_text="Persons: {}".format(person_num),
        )

    def close(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def _load_source_module(self):
        source_path = self._find_source_path()
        spec = importlib.util.spec_from_file_location("elf2_yolov8_pose", str(source_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _find_source_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "yolov8_pose" / "yolo_pose_camera.py",
            Path("/home/elf/my_yolo/yolov8_pose/yolo_pose_camera.py"),
            Path("/home/elf/code/yolov8_pose/yolo_pose_camera.py"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Cannot find yolo_pose_camera.py. Tried: {}".format(
                ", ".join(str(candidate) for candidate in candidates)
            )
        )

    def _default_model_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "yolov8_pose" / "yolov8n-pose.rknn",
            Path("/home/elf/my_yolo/yolov8_pose/yolov8n-pose.rknn"),
            Path("/home/elf/code/yolov8_pose/yolov8n-pose.rknn"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])