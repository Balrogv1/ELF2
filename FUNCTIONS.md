# ELF2 功能目录说明

本文档说明 `ws` 目录下各个功能文件夹的用途、主要入口文件和扩展方式。

## 目录总览

| 目录 | 作用 | 主要入口 |
| --- | --- | --- |
| `qt/` | PyQt5 图形化主界面，统一调度摄像头、任务切换、画面显示和状态栏 | `qt/main.py` |
| `yolo_python/` | YOLOv8 RKNN 目标检测功能，支持 USB 摄像头实时检测 | `yolo_python/usb_yolov8_camera.py` |
| `litemono-python/` | Lite-Mono RKNN 深度估计功能，输出原图和深度伪彩色图 | `litemono-python/usb_litemono_camera.py` |
| `yolov8-world/` | 从 GitHub 仓库拉取的 YOLO-World Lite 示例代码，现已接入 Qt 主界面 | `yolov8-world/rknn_yolo_world_lite.py` |
| `code/` | 预留/临时实验目录，目前只有占位 demo 文件 | `code/demo.py` |

## `qt/` 图形化界面

`qt/` 是当前推荐使用的统一入口。它采用“主界面 + 任务模块 + 任务注册表”的结构：

- `main.py`：创建 UI，左侧显示视频，右侧提供任务选择、摄像头编号、分辨率和启动/停止控制。
- `demo1.py`：兼容旧启动方式，实际会转入 `main.py`。
- `video_worker.py`：统一打开摄像头、读取视频帧、调用当前任务处理，并把画面发回 UI。
- `task_registry.py`：任务注册表，决定 UI 下拉框里有哪些任务。
- `task_base.py`：定义统一任务返回结构 `TaskResult` 和任务基类。
- `tasks/`：每个功能一个任务文件，例如 YOLO、Lite-Mono、原图直通。

在 ELF2 开发板上运行：

```bash
cd ~/code/ws/qt
python3 main.py
```

也可以继续使用旧入口：

```bash
python3 demo1.py
```

## `qt/tasks/` UI 任务模块

任务模块负责把独立算法接入统一 UI。当前已有：

- `task_passthrough.py`：原图直通任务，不做推理，用于测试摄像头和 UI 显示链路。
- `task_yolo.py`：YOLOv8 检测任务，复用 `yolo_python/usb_yolov8_camera.py` 中的预处理、后处理和绘框逻辑。
- `task_litemono.py`：Lite-Mono 深度估计任务，复用 `litemono-python/usb_litemono_camera.py` 中的深度图着色逻辑。
- `task_yoloworld.py`：YOLO-World Lite 任务，复用 `yolov8-world/rknn_yolo_world_lite.py` 中的 RKNN 推理、文本特征和后处理逻辑。

新增功能时，建议新增一个任务文件，例如 `qt/tasks/task_seg.py`，并实现统一接口：

```python
class SegTask:
    name = "Segmentation"

    def open(self, config):
        pass

    def process(self, frame):
        return TaskResult(frame_bgr=frame, metrics={}, status_text="")

    def close(self):
        pass
```

然后在 `qt/task_registry.py` 里注册：

```python
"seg": {
    "label": "Segmentation",
    "module": "tasks.task_seg",
    "class": "SegTask",
    "params": {},
}
```

重新运行 `qt/main.py` 后，新任务会出现在右侧任务下拉框中。

## `yolo_python/` YOLOv8 目标检测

该目录保存 YOLOv8 相关脚本和示例图片：

- `usb_yolov8_camera.py`：USB 摄像头实时目标检测脚本，加载 RKNN 模型并显示检测框。
- `better_yolov8.py`：另一版 YOLOv8 推理脚本。
- `yolov8.py`：图片/模型推理相关脚本。
- `convert.py`：模型转换相关脚本。
- `bus.jpg`：测试图片。

独立运行示例：

```bash
cd ~/code/ws/yolo_python
python3 usb_yolov8_camera.py --model /home/elf/model/yolov8n.rknn --camera 21
```

在 UI 中使用时，由 `qt/tasks/task_yolo.py` 自动加载并调用核心逻辑，不再使用 OpenCV 独立窗口显示。

## `litemono-python/` Lite-Mono 深度估计

该目录保存 Lite-Mono 深度估计功能：

- `usb_litemono_camera.py`：USB 摄像头实时深度估计脚本，输出原图和深度伪彩色图。
- `lite_mono_tiny_640x192.rknn`：Lite-Mono RKNN 模型文件，本仓库 `.gitignore` 默认忽略 `*.rknn`，提交前需确认模型文件管理方式。

独立运行示例：

```bash
cd ~/code/ws/litemono-python
python3 usb_litemono_camera.py --camera 21
```

在 UI 中使用时，由 `qt/tasks/task_litemono.py` 调用核心逻辑，并把拼接后的深度画面返回给 Qt 主界面。

## `yolov8-world/` YOLO-World Lite

该目录来自 GitHub 仓库 `Balrogv1/ELF2.git`，当前包含：

- `pic_yolo_world_lite.py`：图片推理示例。
- `rknn_yolo_world_lite.py`：RKNN YOLO-World Lite 推理脚本。
- `model/bus.jpg`：测试图片。
- `tokenizer/clip_vocab.h`：CLIP tokenizer 词表头文件。

该功能已接入 `qt/` 统一 UI，对应任务为 `YOLO-World Lite`。在右侧任务参数区可配置：

- `model_path`：YOLO-World RKNN 模型路径，例如 `yolo_world_v2s_i8.rknn`。
- `text_features`：预计算文本特征 `.npy` 文件路径；如果提供该项，则不需要动态文本模型。
- `text_model`：CLIP 文本模型 RKNN 路径；未提供 `text_features` 时需要。
- `vocab_header`：CLIP tokenizer 词表头文件路径，默认会尝试使用 `yolov8-world/tokenizer/clip_vocab.h`。
- `classes`：自定义检测类别，使用英文逗号分隔，例如 `person,helmet,car`。
- `class_file`：类别文本文件路径，每行一个类别。
- `labels`：使用预计算 COCO 文本特征时的标签文件路径。
- `conf`：置信度阈值，默认 `0.25`。
- `iou`：NMS IoU 阈值，默认 `0.45`。

如果 `text_features` 为空，则需要同时提供 `text_model` 和 `vocab_header`，任务会在启动时根据 `classes` 动态生成文本特征。

## `code/` 实验目录

`code/` 当前只有 `demo.py`，内容很少，更像是临时实验或占位目录。后续如果没有继续使用，可以把新的功能代码统一放到独立功能目录或 `qt/tasks/` 中，避免入口分散。

## 推荐开发方式

1. 独立功能先在自己的目录里跑通，例如 `yolo_python/` 或 `litemono-python/`。
2. 若需要进入图形界面，就在 `qt/tasks/` 中新增任务适配文件。
3. 在 `qt/task_registry.py` 注册任务。
4. 运行 `qt/main.py`，通过右侧下拉框切换功能。

这种方式能让每个算法脚本保持相对独立，同时让 UI 只关心统一的 `open/process/close` 接口。
