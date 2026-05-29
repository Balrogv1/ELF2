from task_base import BaseVisionTask, TaskResult


class PassthroughTask(BaseVisionTask):
    name = "Original Camera"

    def open(self, config):
        self.config = config

    def process(self, frame):
        return TaskResult(
            frame_bgr=frame,
            metrics={},
            status_text="Raw camera frame",
        )
