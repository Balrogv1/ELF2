#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RKNN_MODEL = os.path.abspath(os.path.join(SCRIPT_DIR, "../model/yolov8n-pose.rknn"))

MODEL_SIZE = (640, 640)  # (width, height)
OBJ_THRESH = 0.5
NMS_THRESH = 0.4
KPT_THRESH = 0.2
CLASSES = ("person",)

POSE_PALETTE = np.array(
    [
        [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
        [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
        [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
    ],
    dtype=np.uint8,
)
KPT_COLOR = POSE_PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
    [3, 5], [4, 6], [5, 7],
]
LIMB_COLOR = POSE_PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


class DetectBox:
    def __init__(self, class_id, score, xmin, ymin, xmax, ymax, keypoint):
        self.class_id = class_id
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint


def sigmoid(x):
    x = np.clip(x.astype(np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def letter_box(im, new_shape, pad_color=(56, 56, 56)):
    shape = im.shape[:2]  # (height, width)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return im, (r, (left, top))


def to_nchw(x, channels=65):
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError("expected 4D detection output, got shape {}".format(x.shape))
    if x.shape[1] == channels:
        return x.astype(np.float32)
    if x.shape[-1] == channels:
        return x.transpose(0, 3, 1, 2).astype(np.float32)
    raise ValueError("cannot infer detection layout from shape {}".format(x.shape))


def to_keypoints(x):
    x = np.asarray(x).astype(np.float32)
    if x.ndim == 4 and x.shape[1] == 17 and x.shape[2] == 3:
        return x
    if x.ndim == 3 and x.shape[1] == 51:
        return x.reshape(x.shape[0], 17, 3, x.shape[2])
    if x.ndim == 4 and x.shape[-2] == 17 and x.shape[-1] == 3:
        return x.transpose(0, 2, 3, 1)
    raise ValueError("cannot infer keypoint layout from shape {}".format(x.shape))


def iou(box1, box2):
    xmin = max(box1.xmin, box2.xmin)
    ymin = max(box1.ymin, box2.ymin)
    xmax = min(box1.xmax, box2.xmax)
    ymax = min(box1.ymax, box2.ymax)
    inner_w = max(0.0, xmax - xmin)
    inner_h = max(0.0, ymax - ymin)
    inter = inner_w * inner_h
    area1 = max(0.0, box1.xmax - box1.xmin) * max(0.0, box1.ymax - box1.ymin)
    area2 = max(0.0, box2.xmax - box2.xmin) * max(0.0, box2.ymax - box2.ymin)
    return inter / max(area1 + area2 - inter, 1e-9)


def nms(detect_result):
    pred_boxes = []
    sorted_boxes = sorted(detect_result, key=lambda x: x.score, reverse=True)
    for i in range(len(sorted_boxes)):
        if sorted_boxes[i].class_id == -1:
            continue
        pred_boxes.append(sorted_boxes[i])
        for j in range(i + 1, len(sorted_boxes)):
            if sorted_boxes[j].class_id == sorted_boxes[i].class_id and iou(sorted_boxes[i], sorted_boxes[j]) > NMS_THRESH:
                sorted_boxes[j].class_id = -1
    return pred_boxes


def process_branch(out, keypoints, index, model_w, model_h, stride):
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    results = []

    for h in range(model_h):
        for w in range(model_w):
            grid_index = h * model_w + w
            score = conf[0, 0, grid_index]
            if score <= OBJ_THRESH:
                continue

            xywh_ = xywh[0, :, grid_index].reshape(1, 4, 16, 1)
            proj = np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1)
            xywh_ = softmax(xywh_, axis=2)
            xywh_ = np.multiply(proj, xywh_).sum(axis=2, keepdims=True).reshape(-1)

            xyxy = xywh_.copy()
            xyxy[0] = (w + 0.5) - xywh_[0]
            xyxy[1] = (h + 0.5) - xywh_[1]
            xyxy[2] = (w + 0.5) + xywh_[2]
            xyxy[3] = (h + 0.5) + xywh_[3]

            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            cx, cy, bw, bh = np.array([cx, cy, bw, bh], dtype=np.float32) * stride

            xmin = cx - bw / 2.0
            ymin = cy - bh / 2.0
            xmax = cx + bw / 2.0
            ymax = cy + bh / 2.0
            kpt = keypoints[..., grid_index + index].copy()
            kpt[..., 0:2] = np.floor(kpt[..., 0:2])
            results.append(DetectBox(0, float(score), xmin, ymin, xmax, ymax, kpt))

    return results


def split_outputs(outputs):
    det_outputs = []
    keypoints = None
    for x in outputs:
        arr = np.asarray(x)
        if arr.ndim == 4 and (arr.shape[1] == 65 or arr.shape[-1] == 65):
            det_outputs.append(to_nchw(arr, channels=65))
        else:
            keypoints = to_keypoints(arr)

    if len(det_outputs) != 3 or keypoints is None:
        shapes = [np.asarray(x).shape for x in outputs]
        raise ValueError("unexpected pose output shapes: {}".format(shapes))

    det_outputs.sort(key=lambda x: x.shape[2] * x.shape[3], reverse=True)
    return det_outputs, keypoints


def post_process(outputs):
    det_outputs, keypoints = split_outputs(outputs)
    results = []
    index = 0
    for x in det_outputs:
        grid_h, grid_w = x.shape[2], x.shape[3]
        stride = MODEL_SIZE[1] // grid_h
        feature = x.reshape(1, 65, -1)
        results.extend(process_branch(feature, keypoints, index, grid_w, grid_h, stride))
        index += grid_h * grid_w
    return nms(results)


def scale_box(box, ratio, pad_offset, img_shape):
    left_pad, top_pad = pad_offset
    img_h, img_w = img_shape[:2]
    xmin = int(np.clip((box.xmin - left_pad) / ratio, 0, img_w))
    ymin = int(np.clip((box.ymin - top_pad) / ratio, 0, img_h))
    xmax = int(np.clip((box.xmax - left_pad) / ratio, 0, img_w))
    ymax = int(np.clip((box.ymax - top_pad) / ratio, 0, img_h))
    return xmin, ymin, xmax, ymax


def scale_keypoints(keypoints, ratio, pad_offset, img_shape):
    left_pad, top_pad = pad_offset
    img_h, img_w = img_shape[:2]
    keypoints = keypoints.reshape(-1, 3).copy()
    keypoints[:, 0] = (keypoints[:, 0] - left_pad) / ratio
    keypoints[:, 1] = (keypoints[:, 1] - top_pad) / ratio
    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, img_w)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, img_h)
    return keypoints


def draw_pose(image, pred_boxes, ratio, pad_offset):
    for box in pred_boxes:
        xmin, ymin, xmax, ymax = scale_box(box, ratio, pad_offset, image.shape)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = "{} {:.2f}".format(CLASSES[box.class_id], box.score)
        cv2.putText(image, label, (xmin, max(0, ymin - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        keypoints = scale_keypoints(box.keypoint, ratio, pad_offset, image.shape)
        for k, keypoint in enumerate(keypoints):
            x, y, conf = keypoint
            if conf < KPT_THRESH:
                continue
            color_k = tuple(int(v) for v in KPT_COLOR[k])
            cv2.circle(image, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)

        for k, sk in enumerate(SKELETON):
            pos1 = keypoints[sk[0] - 1]
            pos2 = keypoints[sk[1] - 1]
            if pos1[2] < KPT_THRESH or pos2[2] < KPT_THRESH:
                continue
            color_l = tuple(int(v) for v in LIMB_COLOR[k])
            cv2.line(
                image,
                (int(pos1[0]), int(pos1[1])),
                (int(pos2[0]), int(pos2[1])),
                color_l,
                thickness=2,
                lineType=cv2.LINE_AA,
            )


def init_runtime(model_path, core):
    rknn_lite = RKNNLite()
    print("--> Load RKNN model")
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError("Load RKNN model failed: {}".format(model_path))
    print("done")

    core_map = {
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "01": "NPU_CORE_0_1",
        "012": "NPU_CORE_0_1_2",
        "all": "NPU_CORE_0_1_2",
    }

    print("--> Init runtime environment")
    if core == "auto":
        ret = rknn_lite.init_runtime()
    else:
        core_mask = getattr(RKNNLite, core_map[core], None)
        if core_mask is None:
            print("Requested core mask is not supported, use default runtime init.")
            ret = rknn_lite.init_runtime()
        else:
            ret = rknn_lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError("Init runtime environment failed")
    print("done")
    return rknn_lite


def open_camera(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera index {}".format(args.camera))

    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Camera opened: {}x{}, camera_fps={:.1f}".format(real_w, real_h, real_fps))
    return cap, real_w, real_h, real_fps


def create_writer(path, width, height, fps):
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 30, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open video writer: {}".format(path))
    return writer


def put_status(image, fps, infer_ms, post_ms, det_num):
    text = "FPS {:.1f} | infer {:.1f} ms | post {:.1f} ms | person {}".format(fps, infer_ms, post_ms, det_num)
    cv2.rectangle(image, (8, 8), (640, 42), (0, 0, 0), -1)
    cv2.putText(image, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8-Pose RKNNLite USB camera demo")
    parser.add_argument("--model", default=RKNN_MODEL, help="Path to yolov8n-pose.rknn")
    parser.add_argument("--camera", type=int, default=21, help="USB camera index")
    parser.add_argument("--cam_width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--cam_height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--cam_fps", type=int, default=30, help="Camera capture fps")
    parser.add_argument("--fourcc", default="MJPG", help="Camera FOURCC, such as MJPG or YUYV")
    parser.add_argument("--core", default="auto", choices=("auto", "0", "1", "2", "01", "012", "all"))
    parser.add_argument("--no_window", action="store_true", help="Do not show OpenCV window")
    parser.add_argument("--save_video", default=None, help="Optional path to save rendered video")
    return parser.parse_args()


def main():
    args = parse_args()
    rknn_lite = init_runtime(args.model, args.core)
    cap = None
    writer = None
    fps_smooth = None

    try:
        cap, real_w, real_h, real_fps = open_camera(args)
        writer = create_writer(args.save_video, real_w, real_h, real_fps)
        print("Press q or ESC to quit.")

        while True:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\nRead camera frame failed")
                break

            img, info = letter_box(frame.copy(), new_shape=(MODEL_SIZE[1], MODEL_SIZE[0]))
            ratio, pad_offset = info
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(img, axis=0)
            input_data = np.ascontiguousarray(input_data)

            infer_start = time.perf_counter()
            outputs = rknn_lite.inference(inputs=[input_data])
            infer_end = time.perf_counter()

            post_start = time.perf_counter()
            pred_boxes = post_process(outputs)
            show_frame = frame.copy()
            draw_pose(show_frame, pred_boxes, ratio, pad_offset)
            post_end = time.perf_counter()

            loop_end = time.perf_counter()
            frame_time = loop_end - loop_start
            infer_ms = (infer_end - infer_start) * 1000.0
            post_ms = (post_end - post_start) * 1000.0
            current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            fps_smooth = current_fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * current_fps

            sys.stdout.write(
                "\rFPS: {:6.2f} | Infer: {:6.1f} ms | Post: {:6.1f} ms | Persons: {:3d}".format(
                    fps_smooth, infer_ms, post_ms, len(pred_boxes)
                )
            )
            sys.stdout.flush()

            if writer is not None:
                writer.write(show_frame)

            if not args.no_window:
                cv2.imshow("YOLOv8-Pose RKNN USB Camera", show_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nReleasing resources...")
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        rknn_lite.release()
        print("done")


if __name__ == "__main__":
    main()

