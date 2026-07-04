import time

from PyQt5.QtCore import QThread, pyqtSignal


class VideoWorker(QThread):
    frame_ready = pyqtSignal(object, dict)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, task_cls, config, parent=None):
        super().__init__(parent)
        self.task_cls = task_cls
        self.config = config
        self._running = False
        self.cap = None
        self.task = None

    def run(self):
        self._running = True
        fps_smooth = None

        try:
            self.cap = self._open_camera()
            self.task = self.task_cls()
            self.task.open(self.config)
            self.status.emit("Running {}".format(getattr(self.task, "name", "task")))

            while self._running:
                loop_start = time.perf_counter()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.error.emit("Read camera frame failed")
                    break

                result = self.task.process(frame)
                elapsed = time.perf_counter() - loop_start
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                fps_smooth = fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * fps

                h, w = result.frame_bgr.shape[:2]
                metrics = dict(result.metrics)
                metrics.update(
                    {
                        "fps": fps_smooth,
                        "width": w,
                        "height": h,
                        "task_name": getattr(self.task, "name", ""),
                        "status_text": result.status_text,
                    }
                )
                self.frame_ready.emit(result.frame_bgr, metrics)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._release()

    def stop(self):
        self._running = False
        if self.cap is not None:
            self.cap.release()
        return self.wait(3000)

    def _open_camera(self):
        import cv2

        camera_source = self.config.get("camera_index", 21)
        if isinstance(camera_source, str) and camera_source.isdigit():
            camera_source = int(camera_source)

        cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera: {}".format(camera_source))

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get("width", 640)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get("height", 480)))
        cap.set(cv2.CAP_PROP_FPS, int(self.config.get("fps", 30)))
        return cap

    def _release(self):
        if self.task is not None:
            self.task.close()
            self.task = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status.emit("Stopped")
