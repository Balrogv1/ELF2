import importlib


TASKS = {
    "passthrough": {
        "label": "Original Camera",
        "module": "tasks.task_passthrough",
        "class": "PassthroughTask",
        "params": {},
    },
    "yolo": {
        "label": "YOLOv8 Detection",
        "module": "tasks.task_yolo",
        "class": "YoloTask",
        "params": {
            "model_path": "",
        },
    },
    "litemono": {
        "label": "Lite-Mono Depth",
        "module": "tasks.task_litemono",
        "class": "LiteMonoTask",
        "params": {
            "model_path": "",
        },
    },
    "yoloworld": {
        "label": "YOLO-World Lite",
        "module": "tasks.task_yoloworld",
        "class": "YoloWorldTask",
        "params": {
            "model_path": "/home/elf/my_yolo/yolov8-world/model/yolo_world_v2s_i8.rknn",
            "text_features": "",
            "text_model": "/home/elf/my_yolo/yolov8-world/model/clip_text_fp16.rknn",
            "vocab_header": "/home/elf/my_yolo/yolov8-world/tokenizer/clip_vocab.h",
            "classes": "",
            "class_file": "",
            "labels": "",
            "conf": "0.25",
            "iou": "0.45",
        },
    },
}


def get_task_class(task_id):
    meta = TASKS[task_id]
    module = importlib.import_module(meta["module"])
    return getattr(module, meta["class"])
