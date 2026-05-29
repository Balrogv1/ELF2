import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from task_base import BaseVisionTask, TaskResult


class YoloWorldTask(BaseVisionTask):
    name = "YOLO-World Lite"

    def __init__(self):
        self.module = None
        self.yolo_runtime = None
        self.text_runtime = None
        self.tokenizer = None
        self.text_features = None
        self.labels = None
        self.args = None

    def open(self, config):
        self.module = self._load_source_module()
        params = config.get("task_params", {})

        model_path = params.get("model_path") or self._find_first_existing(
            [
                "yolo_world_v2s_i8.rknn",
                "yolov8-world/model/yolo_world_v2s_i8.rknn",
                "/home/elf/code/ws/yolov8-world/model/yolo_world_v2s_i8.rknn",
                "/home/elf/code/yolov8-world/model/yolo_world_v2s_i8.rknn",
            ]
        )
        if not model_path:
            raise FileNotFoundError("Please set YOLO-World model_path in the UI.")

        labels_path = params.get("labels") or None
        classes_text = params.get("classes") or None
        class_file = params.get("class_file") or None
        text_features_path = params.get("text_features") or self._find_first_existing(
            [
                "yolov8-world/model/coco_text_outp.npy",
                "yolov8-world/coco_text_outp.npy",
                "/home/elf/code/ws/yolov8-world/model/coco_text_outp.npy",
                "/home/elf/code/yolov8-world/model/coco_text_outp.npy",
            ]
        )
        text_model_path = params.get("text_model") or self._find_first_existing(
            [
                "yolov8-world/model/clip_text_fp16.rknn",
                "/home/elf/code/ws/yolov8-world/model/clip_text_fp16.rknn",
                "/home/elf/code/yolov8-world/model/clip_text_fp16.rknn",
            ]
        )
        vocab_header = params.get("vocab_header") or self._find_first_existing(
            [
                "yolov8-world/tokenizer/clip_vocab.h",
                "/home/elf/code/ws/yolov8-world/tokenizer/clip_vocab.h",
                "/home/elf/code/yolov8-world/tokenizer/clip_vocab.h",
            ]
        )

        self.args = SimpleNamespace(
            layout=params.get("layout") or "nhwc",
            input_dtype=params.get("input_dtype") or "uint8",
            conf=float(params.get("conf") or 0.25),
            iou=float(params.get("iou") or 0.45),
            target=params.get("target") or "rk3588",
            core=params.get("core") or "auto",
        )

        if text_features_path:
            self.labels = self.module.load_labels(labels_path)
            self.text_features = self._load_text_features(text_features_path)
        else:
            if not text_model_path or not vocab_header:
                raise FileNotFoundError(
                    "YOLO-World needs text_features, or text_model + vocab_header."
                )
            if class_file:
                self.labels = [
                    line.strip()
                    for line in Path(class_file).read_text().splitlines()
                    if line.strip()
                ]
            elif classes_text:
                self.labels = self.module.parse_class_text(classes_text)
            else:
                self.labels = list(self.module.CLASSES)
            if not self.labels:
                raise ValueError("YOLO-World classes cannot be empty.")
            if len(self.labels) > 80:
                raise ValueError("YOLO-World supports at most 80 classes.")

            self.tokenizer = self.module.CLIPTokenizer(vocab_header)
            self.text_runtime = self.module.create_runtime(
                text_model_path,
                target=self.args.target,
                core=self.args.core,
            )
            self.text_features = self.module.make_text_features(
                self.text_runtime,
                self.tokenizer,
                self.labels,
            )

        self.yolo_runtime = self.module.create_runtime(
            model_path,
            target=self.args.target,
            core=self.args.core,
        )

    def process(self, frame):
        infer_start = time.perf_counter()
        rendered, det_count = self.module.run_yolo_frame(
            self.yolo_runtime,
            frame,
            self.text_features,
            self.labels,
            self.args,
            fps_text=None,
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        return TaskResult(
            frame_bgr=rendered,
            metrics={"infer_ms": infer_ms, "detections": det_count},
            status_text="Detections: {}".format(det_count),
        )

    def close(self):
        if self.yolo_runtime is not None:
            self.yolo_runtime.release()
            self.yolo_runtime = None
        if self.text_runtime is not None:
            self.text_runtime.release()
            self.text_runtime = None

    def _load_source_module(self):
        source_path = self._find_source_path()
        spec = importlib.util.spec_from_file_location("elf2_yoloworld", str(source_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _find_source_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "yolov8-world" / "rknn_yolo_world_lite.py",
            root.parent / "yolov8-world" / "rknn_yolo_world_lite.py",
            Path("/home/elf/code/ws/yolov8-world/rknn_yolo_world_lite.py"),
            Path("/home/elf/code/yolov8-world/rknn_yolo_world_lite.py"),
            Path("/home/elf/yolov8-world/rknn_yolo_world_lite.py"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Cannot find rknn_yolo_world_lite.py. Tried: {}".format(
                ", ".join(str(candidate) for candidate in candidates)
            )
        )

    def _find_first_existing(self, candidates):
        root = Path(__file__).resolve().parents[2]
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                path = root / path
            if path.exists():
                return str(path)
        return ""

    def _load_text_features(self, path):
        try:
            text = np.load(path, allow_pickle=False)
        except ValueError as exc:
            if "allow_pickle" not in str(exc):
                raise
            text = np.load(path, allow_pickle=True)

        if isinstance(text, np.lib.npyio.NpzFile):
            keys = list(text.keys())
            if not keys:
                raise ValueError("text_features npz is empty: {}".format(path))
            text = text[keys[0]]

        if isinstance(text, np.ndarray) and text.dtype == object:
            if text.shape == ():
                text = text.item()
            elif text.size == 1:
                text = text.reshape(-1)[0]

        if isinstance(text, dict):
            for key in ("text_features", "features", "arr_0"):
                if key in text:
                    text = text[key]
                    break
            else:
                raise ValueError(
                    "Pickled text_features dict must contain text_features/features/arr_0"
                )

        text = np.asarray(text, dtype=np.float32)
        if text.shape == (80, 512):
            text = np.expand_dims(text, 0)
        if text.shape != (1, 80, 512):
            raise ValueError(
                "expected text feature shape (1, 80, 512), got {}".format(text.shape)
            )
        return text
