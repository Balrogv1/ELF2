# ELF2 Vision Demo

这是 ELF2/RK3588 板卡上的多任务视觉演示工程。当前主入口是 PyQt5 图形界面，支持 USB 摄像头画面显示、YOLOv8 检测、Lite-Mono 深度估计、YOLO-World Lite、手机浏览器查看识别结果，以及 Odin1 ROS2 lite 位姿坐标显示。

## 目录结构

| 目录/文件 | 作用 |
| --- | --- |
| `qt/` | PyQt5 主界面、任务调度、Odin1 控制和任务适配器 |
| `qt/main.py` | 推荐启动入口 |
| `qt/webrtc_server.py` | 使用 Rockchip H.264 硬编码向手机浏览器输出 WebRTC 画面 |
| `qt/tasks/` | 视觉任务插件目录，每个功能一个任务文件 |
| `qt/odin1_odom_bridge.py` | ROS2 bridge，订阅 `/odin1/odometry` 并把 XYZ 输出给 Qt |
| `yolo_python/` | YOLOv8 RKNN USB 摄像头检测脚本 |
| `litemono-python/` | Lite-Mono RKNN 单目深度/视差估计脚本 |
| `yolov8_pose/` | YOLOv8 Pose RKNN 姿态估计脚本和模型 |
| `yolov8_seg/` | YOLOv8 Seg RKNN 实例分割脚本和模型 |
| `yolov8-world/` | YOLO-World Lite RKNN 示例和 tokenizer 文件 |
| `FUNCTIONS.md` | 更详细的功能目录说明 |

## 板端运行

在 ELF2/RK3588 板卡上：

```bash
cd ~/my_yolo/qt
python3 main.py
```

界面右侧可以切换任务、分辨率、启动/停止视觉任务，并独立启动/停止 Odin1 lite 驱动。

## WebRTC 手机查看识别结果

板卡首次使用需要安装 GStreamer WebRTC 的 Python 类型绑定：

```bash
sudo apt-get install gir1.2-gst-plugins-bad-1.0 gstreamer1.0-nice
python3 -m pip install --user websockets==10.4
```

手机和 ELF2 连接到同一局域网。在 Qt 右侧 `WebRTC View` 中保留默认端口 `8080`，点击 `Start WebRTC View`。界面会显示访问地址，例如：

```text
http://192.168.93.64:8080
```

用手机浏览器打开该地址，即可查看当前任务处理后的画面，以及任务名称、FPS 和分辨率。HTTP 页面使用端口 `8080`，WebSocket signaling 使用相邻端口 `8081`，两个端口都需要允许手机访问。

编码链路：

```text
frame_bgr -> appsrc -> videoconvert -> NV12 -> mpph264enc
          -> h264parse -> rtph264pay -> webrtcbin -> browser
```

手机端不会再次打开摄像头。停止视觉任务会发送黑帧，但保持 WebRTC 连接；点击 `Stop WebRTC View` 才会关闭 HTTP、signaling 和 GStreamer pipeline。当前实现支持一个手机浏览器连接，新连接会替换旧连接。

## 当前视觉任务

### Original Camera

原图直通任务，只显示 USB 摄像头画面。用于确认摄像头、Qt 显示和 `VideoWorker` 链路正常。

### YOLOv8 Detection

复用 `yolo_python/usb_yolov8_camera.py` 的预处理、RKNN 推理、后处理和绘框逻辑。UI 中显示检测后的摄像头画面，底部状态栏动态显示 FPS、分辨率、推理耗时和检测数量。

### YOLOv8 Pose

复用 `yolov8_pose/yolo_pose_camera.py` 的 RKNN 推理和关键点绘制逻辑。UI 中显示人体框和骨架关键点，底部状态栏动态显示 FPS、分辨率、推理耗时、后处理耗时和人数。

### YOLOv8 Segmentation

复用 `yolov8_seg/yolo_seg_camera.py` 的 RKNN 推理、mask 后处理、mask 叠加和检测框绘制逻辑。UI 中显示实例分割结果，任务参数支持 `no_mask` 和 `mask_alpha`。
### Lite-Mono Depth

复用 `litemono-python/usb_litemono_camera.py` 的 Lite-Mono RKNN 推理逻辑。输入普通单目摄像头画面，输出相对深度/视差图，并与原图左右拼接显示。

注意：当前深度图是相对深度/视差伪彩色图，不是以米为单位的绝对距离。

### YOLO-World Lite

复用 `yolov8-world/rknn_yolo_world_lite.py` 的 RKNN 推理逻辑。支持预计算 `text_features`，也支持使用 `clip_text_fp16.rknn + clip_vocab.h + classes` 动态生成文本特征。

默认路径：

```text
model_path: /home/elf/my_yolo/yolov8-world/model/yolo_world_v2s_i8.rknn
text_model: /home/elf/my_yolo/yolov8-world/model/clip_text_fp16.rknn
vocab_header: /home/elf/my_yolo/yolov8-world/tokenizer/clip_vocab.h
```

## Odin1 ROS2 Lite 坐标显示

Qt 右侧新增 `Odin1 Position` 区域。

点击 `Start Odin1 Lite` 后，Qt 会通过 `QProcess` 启动：

```bash
source /opt/ros/humble/setup.bash
source ~/odin1/install/setup.bash
cd ~/odin1
ros2 launch odin_ros_driver odin1_ros2_lite.launch.py
```

同时启动 `qt/odin1_odom_bridge.py`，订阅：

```text
/odin1/odometry
```

消息类型：

```text
nav_msgs/msg/Odometry
```

读取字段：

```text
pose.pose.position.x
pose.pose.position.y
pose.pose.position.z
```

Qt 中实时显示：

```text
X: ... | Y: ... | Z: ...
```

点击 `Stop Odin1` 会停止 bridge 和 ROS2 lite driver 进程。

当前 lite 版保留：

```text
/odin1/odometry
/tf
/odin1/path
```

当前 lite 版关闭 RGB、IMU、点云、深度、重建、渲染、录制等重功能。

## 模型文件策略

GitHub 普通仓库单文件上限是 100 MB。YOLO-World 的大模型不要直接提交：

```text
yolov8-world/model/clip_text_fp16.rknn
yolov8-world/model/yolo_world_v2s_i8.rknn
```

这些文件已在 `.gitignore` 中忽略，建议在板卡本地手动放到默认路径。

较小的模型文件可以按需提交，例如 Lite-Mono 小模型；提交前用 `git status` 和 `git check-ignore -v <file>` 确认是否被跟踪或忽略。

## 新增任务方式

新增一个视觉功能时：

1. 在 `qt/tasks/` 下新增任务文件，例如 `task_seg.py`。
2. 实现统一接口：`open(config)`、`process(frame)`、`close()`。
3. 在 `qt/task_registry.py` 注册任务。
4. 重新运行 `qt/main.py`，新任务会出现在右侧任务下拉框中。

任务返回 `TaskResult` 后，Qt 会统一显示画面和动态状态栏。

## Git 同步建议

PC 端修改后上传：

```bash
git add <files>
git commit -m "message"
git push origin main
```

板卡以 GitHub 最新代码为准时：

```bash
cd ~/my_yolo
git fetch origin
git reset --hard origin/main
```

未被 Git 跟踪且被 `.gitignore` 忽略的本地模型文件通常会保留。不要用会清理未跟踪文件的命令删除模型。
