import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np

from task_base import BaseVisionTask, TaskResult


class LiteMonoTask(BaseVisionTask):
    name = "Lite-Mono Depth"

    def __init__(self):
        self.rknn = None
        self.mono = None
        self.model_path = None

    def open(self, config):
        self.mono = self._load_source_module()
        task_params = config.get("task_params", {})
        self.model_path = task_params.get("model_path") or self._default_model_path()

        self.rknn = self.mono.RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            self.rknn.release()
            self.rknn = None
            raise RuntimeError("Load Lite-Mono RKNN model failed: {}".format(self.model_path))

        ret = self.rknn.init_runtime(core_mask=self.mono.RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            self.rknn.release()
            self.rknn = None
            raise RuntimeError("Init Lite-Mono RKNN runtime failed")

    def process(self, frame):
        original_h, original_w = frame.shape[:2]
        img = cv2.resize(
            frame,
            (self.mono.MODEL_W, self.mono.MODEL_H),
            interpolation=cv2.INTER_LINEAR,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        input_data = np.expand_dims(img, axis=0)
        input_data = np.ascontiguousarray(input_data)

        infer_start = time.perf_counter()
        outputs = self.rknn.inference(
            inputs=[input_data],
            data_type="float32",
            data_format="nhwc",
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        disp = np.squeeze(outputs[0])
        disp_color = self.mono.disp_to_colormap(disp, original_w, original_h)
        show = np.hstack([frame, disp_color])

        return TaskResult(
            frame_bgr=show,
            metrics={
                "infer_ms": infer_ms,
                "disp_min": float(disp.min()),
                "disp_max": float(disp.max()),
            },
            status_text="Depth map ready",
        )

    def close(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def _load_source_module(self):
        source_path = self._find_source_path()
        spec = importlib.util.spec_from_file_location("elf2_litemono_camera", str(source_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _find_source_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "litemono-python" / "usb_litemono_camera.py",
            root / "litemono_python" / "usb_litemono_camera.py",
            Path("/home/elf/code/litemono-python/usb_litemono_camera.py"),
            Path("/home/elf/code/litemono_python/usb_litemono_camera.py"),
            Path("/home/elf/litemono-python/usb_litemono_camera.py"),
            Path("/home/elf/litemono_python/usb_litemono_camera.py"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Cannot find usb_litemono_camera.py. Tried: {}".format(
                ", ".join(str(candidate) for candidate in candidates)
            )
        )

    def _default_model_path(self):
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "litemono-python" / "lite_mono_tiny_640x192.rknn",
            Path("/home/elf/litemono-python/lite_mono_tiny_640x192.rknn"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])
