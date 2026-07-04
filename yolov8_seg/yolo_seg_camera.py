#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RKNN_MODEL = os.path.abspath(os.path.join(SCRIPT_DIR, "../model/yolov8n-seg.rknn"))

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
MASK_THRESH = 0.5
MAX_DETECT = 300
MODEL_SIZE = (640, 640)  # (width, height)

CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

COLOR_PALETTE = np.array(
    [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
        (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
        (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
        (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ],
    dtype=np.uint8,
)


def sigmoid(x):
    x = np.clip(x.astype(np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def letter_box(im, new_shape, pad_color=(114, 114, 114)):
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
    return im, (r, (left, top), new_unpad)


def to_nchw(x):
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError("expected 4D output, got shape {}".format(x.shape))
    if x.shape[1] in (1, 4, 32, 64, 80):
        return x.astype(np.float32)
    if x.shape[-1] in (1, 4, 32, 64, 80):
        return x.transpose(0, 3, 1, 2).astype(np.float32)
    raise ValueError("cannot infer output layout from shape {}".format(x.shape))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def dfl(position):
    position = to_nchw(position)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def box_process(position):
    position = to_nchw(position)
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1).astype(np.float32)
    stride = np.array([MODEL_SIZE[0] // grid_w, MODEL_SIZE[1] // grid_h], dtype=np.float32).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def spatial_flatten(x):
    x = to_nchw(x)
    return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])


def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[class_pos]
    boxes = boxes[class_pos]
    classes = classes[class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[class_pos]
    return boxes, classes, scores, seg_part


def nms_boxes(boxes, scores):
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[np.where(iou <= NMS_THRESH)[0] + 1]

    return np.array(keep, dtype=np.int64)


def crop_mask(masks, boxes):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = np.round(box).astype(np.int32)
        x1 = int(np.clip(x1, 0, MODEL_SIZE[0]))
        x2 = int(np.clip(x2, 0, MODEL_SIZE[0]))
        y1 = int(np.clip(y1, 0, MODEL_SIZE[1]))
        y2 = int(np.clip(y2, 0, MODEL_SIZE[1]))
        crop = np.zeros_like(masks[i], dtype=np.float32)
        if x2 > x1 and y2 > y1:
            crop[y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
        masks[i] = crop
    return masks


def post_process(input_data):
    proto = to_nchw(input_data[-1])[0]
    boxes, scores, classes_conf, seg_part = [], [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(spatial_flatten(box_process(input_data[pair_per_branch * i])))
        classes_conf.append(spatial_flatten(input_data[pair_per_branch * i + 1]))
        scores.append(np.ones_like(to_nchw(input_data[pair_per_branch * i + 1])[:, :1, :, :], dtype=np.float32))
        seg_part.append(spatial_flatten(input_data[pair_per_branch * i + 3]))

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate([spatial_flatten(x) for x in scores])
    seg_part = np.concatenate(seg_part)
    boxes, classes, scores, seg_part = filter_boxes(boxes, scores, classes_conf, seg_part)
    if boxes.shape[0] == 0:
        return None, None, None, None

    if boxes.shape[0] > 30000:
        order = scores.argsort()[::-1][:30000]
        boxes, classes, scores, seg_part = boxes[order], classes[order], scores[order], seg_part[order]

    keep_all = []
    for cls_id in np.unique(classes):
        inds = np.where(classes == cls_id)[0]
        keep = nms_boxes(boxes[inds], scores[inds])
        keep_all.extend(inds[keep].tolist())

    if not keep_all:
        return None, None, None, None

    keep_all = np.array(keep_all, dtype=np.int64)
    keep_all = keep_all[scores[keep_all].argsort()[::-1]][:MAX_DETECT]
    boxes, classes, scores, seg_part = boxes[keep_all], classes[keep_all], scores[keep_all], seg_part[keep_all]

    proto = proto.reshape(seg_part.shape[-1], -1)
    seg_img = sigmoid(np.matmul(seg_part, proto))
    proto_h, proto_w = to_nchw(input_data[-1]).shape[-2:]
    seg_img = seg_img.reshape(-1, proto_h, proto_w)

    resized_masks = []
    for mask in seg_img:
        resized_masks.append(cv2.resize(mask, MODEL_SIZE, interpolation=cv2.INTER_LINEAR))
    seg_img = np.asarray(resized_masks, dtype=np.float32)
    seg_img = crop_mask(seg_img, boxes)
    seg_img = seg_img > MASK_THRESH
    return boxes, classes, scores, seg_img


def scale_boxes(boxes, ratio, pad_offset, img_shape):
    boxes = boxes.copy()
    left_pad, top_pad = pad_offset
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left_pad) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top_pad) / ratio
    img_h, img_w = img_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)
    return boxes


def scale_masks(masks, pad_offset, resized_shape, img_shape):
    if masks is None:
        return None

    left_pad, top_pad = pad_offset
    resized_w, resized_h = resized_shape
    img_h, img_w = img_shape[:2]
    real_masks = []
    for mask in masks:
        crop = mask[top_pad:top_pad + resized_h, left_pad:left_pad + resized_w]
        crop = crop.astype(np.uint8)
        real_mask = cv2.resize(crop, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        real_masks.append(real_mask.astype(bool))
    return np.asarray(real_masks)


def merge_seg(image, masks, classes, alpha=0.45):
    out = image.copy()
    for mask, cls_id in zip(masks, classes):
        color = COLOR_PALETTE[int(cls_id) % len(COLOR_PALETTE)].astype(np.float32)
        out[mask] = (out[mask].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
    return out


def draw_seg(image, boxes, scores, classes):
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(np.int32).tolist()
        cls_id = int(cls_id)
        color = tuple(int(x) for x in COLOR_PALETTE[cls_id % len(COLOR_PALETTE)])
        label = "{} {:.2f}".format(CLASSES[cls_id], float(score))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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
    text = "FPS {:.1f} | infer {:.1f} ms | post {:.1f} ms | det {}".format(fps, infer_ms, post_ms, det_num)
    cv2.rectangle(image, (8, 8), (620, 42), (0, 0, 0), -1)
    cv2.putText(image, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8-Seg RKNNLite USB camera demo")
    parser.add_argument("--model", default=RKNN_MODEL, help="Path to yolov8n-seg.rknn")
    parser.add_argument("--camera", type=int, default=21, help="USB camera index")
    parser.add_argument("--cam_width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--cam_height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--cam_fps", type=int, default=30, help="Camera capture fps")
    parser.add_argument("--fourcc", default="MJPG", help="Camera FOURCC, such as MJPG or YUYV")
    parser.add_argument("--core", default="auto", choices=("auto", "0", "1", "2", "01", "012", "all"))
    parser.add_argument("--no_window", action="store_true", help="Do not show OpenCV window")
    parser.add_argument("--no_mask", action="store_true", help="Only draw segmentation boxes")
    parser.add_argument("--mask_alpha", type=float, default=0.45, help="Mask overlay alpha")
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
            ratio, pad_offset, resized_shape = info
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(img, axis=0)
            input_data = np.ascontiguousarray(input_data)

            infer_start = time.perf_counter()
            outputs = rknn_lite.inference(inputs=[input_data])
            infer_end = time.perf_counter()

            post_start = time.perf_counter()
            boxes, classes, scores, seg_img = post_process(outputs)
            show_frame = frame.copy()
            det_num = 0

            if boxes is not None:
                det_num = len(boxes)
                real_boxes = scale_boxes(boxes, ratio, pad_offset, frame.shape)
                if args.no_mask:
                    img_p = show_frame
                else:
                    real_segs = scale_masks(seg_img, pad_offset, resized_shape, frame.shape)
                    img_p = merge_seg(show_frame, real_segs, classes, alpha=args.mask_alpha)
                draw_seg(img_p, real_boxes, scores, classes)
                show_frame = img_p

            post_end = time.perf_counter()
            loop_end = time.perf_counter()
            frame_time = loop_end - loop_start
            infer_ms = (infer_end - infer_start) * 1000.0
            post_ms = (post_end - post_start) * 1000.0
            current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            fps_smooth = current_fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * current_fps
            sys.stdout.write(
                "\rFPS: {:6.2f} | Infer: {:6.1f} ms | Post: {:6.1f} ms | Detections: {:3d}".format(
                    fps_smooth, infer_ms, post_ms, det_num
                )
            )
            sys.stdout.flush()

            if writer is not None:
                writer.write(show_frame)

            if not args.no_window:
                cv2.imshow("YOLOv8-Seg RKNN USB Camera", show_frame)
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

