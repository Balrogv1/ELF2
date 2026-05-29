from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TaskResult:
    frame_bgr: Any
    metrics: Dict[str, Any] = field(default_factory=dict)
    status_text: str = ""


class BaseVisionTask:
    name = "Base Task"

    def open(self, config):
        raise NotImplementedError

    def process(self, frame):
        raise NotImplementedError

    def close(self):
        pass
