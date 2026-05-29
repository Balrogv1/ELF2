#!/usr/bin/env python3
"""Run YOLO-World RKNN on RKNN Toolkit Lite2 without torch/transformers."""
"""This Python script is only for picture input"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np


CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407
PAD_TOKEN_ID = 49407
SEQUENCE_LEN = 20


def load_labels(path):
    if not path:
        return list(CLASSES)
    labels = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(labels) != 80:
        raise ValueError(f"expected 80 labels in {path}, got {len(labels)}")
    return labels


def parse_classes(args):
    if args.class_file:
        classes = [line.strip() for line in Path(args.class_file).read_text().splitlines() if line.strip()]
    elif args.classes:
        classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    else:
        classes = load_labels(args.labels)

    if not classes:
        raise ValueError("no classes provided")
    if len(classes) > 80:
        raise ValueError(f"YOLO-World RKNN expects at most 80 classes, got {len(classes)}")
    return classes


def letterbox(img, size=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = size
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * scale)), int(round(h * scale))
    pad_w, pad_h = new_w - resized_w, new_h - resized_h
    pad_left, pad_right = int(round(pad_w / 2 - 0.1)), int(round(pad_w / 2 + 0.1))
    pad_top, pad_bottom = int(round(pad_h / 2 - 0.1)), int(round(pad_h / 2 + 0.1))

    if (w, h) != (resized_w, resized_h):
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return img, scale, (pad_left, pad_top)


def preprocess_image(path, size=(640, 640), layout="nhwc", dtype="uint8"):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded, scale, pad = letterbox(rgb, size=size)
    if dtype == "float32":
        padded = padded.astype(np.float32)
    else:
        padded = padded.astype(np.uint8)

    if layout == "nchw":
        padded = padded.transpose(2, 0, 1)
    return np.expand_dims(padded, 0), bgr, scale, pad


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_merges_from_header(path):
    text = Path(path).read_text(errors="ignore")
    values = re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", text)
    if not values:
        raise ValueError(f"failed to parse CLIP vocab header: {path}")
    data = bytes(int(v, 0) for v in values)
    return data.decode("utf-8")


class CLIPTokenizer:
    def __init__(self, vocab_header):
        merges_text = load_merges_from_header(vocab_header)
        self.byte_encoder = bytes_to_unicode()
        merges = merges_text.splitlines()[1:]
        merge_pairs = [tuple(line.split()) for line in merges if line and len(line.split()) == 2]

        vocab = list(bytes_to_unicode().values())
        vocab += [v + "</w>" for v in bytes_to_unicode().values()]
        vocab += ["".join(pair) for pair in merge_pairs]
        vocab += ["<|startoftext|>", "<|endoftext|>"]

        self.encoder = {token: i for i, token in enumerate(vocab)}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merge_pairs)}
        self.cache = {}
        self.pattern = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]+",
            re.IGNORECASE,
        )

    @staticmethod
    def get_pairs(word):
        return set(zip(word, word[1:]))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        if not token:
            return ""
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self.get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)

        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text):
        text = " ".join(text.strip().split()).lower()
        tokens = []
        for match in self.pattern.findall(text):
            encoded = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
            tokens.extend(self.encoder[piece] for piece in self.bpe(encoded).split(" "))
        return tokens

    def tokenize(self, text, max_length=SEQUENCE_LEN, padding=True):
        tokens = [BOS_TOKEN_ID] + self.encode(text)
        if len(tokens) > max_length - 1:
            tokens = tokens[: max_length - 1]
            tokens.append(EOS_TOKEN_ID)
        else:
            tokens.append(EOS_TOKEN_ID)
            if padding:
                tokens.extend([PAD_TOKEN_ID] * (max_length - len(tokens)))
        return np.array(tokens, dtype=np.int32)


def load_text_features(path):
    text = np.load(path).astype(np.float32)
    if text.shape == (80, 512):
        text = np.expand_dims(text, 0)
    if text.shape != (1, 80, 512):
        raise ValueError(f"expected text feature shape (1, 80, 512), got {text.shape}")
    return text


def run_clip_text(text_model, classes, vocab_header, target="rk3588", core="auto"):
    tokenizer = CLIPTokenizer(vocab_header)
    input_ids = np.stack([tokenizer.tokenize(text) for text in classes], axis=0)

    rknn = create_runtime(text_model, target=target, core=core)
    outputs = []
    try:
        for row in input_ids:
            out = rknn.inference(inputs=[row.reshape(1, -1)])
            outputs.append(np.asarray(out[0], dtype=np.float32))
    finally:
        rknn.release()

    text_features = np.concatenate(outputs, axis=0)
    if text_features.shape[0] != len(classes):
        text_features = text_features.reshape(len(classes), -1)
    if text_features.shape[1] != 512:
        raise ValueError(f"expected CLIP text output shape (N, 512), got {text_features.shape}")

    if len(classes) < 80:
        padded = np.zeros((80, 512), dtype=np.float32)
        padded[: len(classes)] = text_features
        text_features = padded

    return np.expand_dims(text_features.astype(np.float32), 0)


def to_nchw(arr):
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"expected 4D output, got shape {arr.shape}")
    if arr.shape[1] in (1, 4, 80):
        return arr
    if arr.shape[-1] in (1, 4, 80):
        return arr.transpose(0, 3, 1, 2)
    raise ValueError(f"cannot infer output layout from shape {arr.shape}")


def boxes_from_distribution(position, img_size=(640, 640)):
    position = to_nchw(position).astype(np.float32)
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1).astype(np.float32)
    stride = np.array([img_size[1] // grid_w, img_size[0] // grid_h], dtype=np.float32).reshape(1, 2, 1, 1)

    xy1 = (grid + 0.5 - position[:, 0:2]) * stride
    xy2 = (grid + 0.5 + position[:, 2:4]) * stride
    return np.concatenate((xy1, xy2), axis=1)


def flatten_spatial(arr):
    arr = to_nchw(arr).astype(np.float32)
    return arr.transpose(0, 2, 3, 1).reshape(-1, arr.shape[1])


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[np.where(iou <= iou_thres)[0] + 1]

    return np.array(keep, dtype=np.int64)


def postprocess(
    outputs,
    orig_shape,
    scale,
    pad,
    conf_thres=0.25,
    iou_thres=0.45,
    img_size=(640, 640),
    max_det=128,
    num_classes=80,
):
    if len(outputs) % 3 != 0:
        raise ValueError(f"expected outputs grouped by 3 branches, got {len(outputs)} outputs")
    if any(np.issubdtype(np.asarray(x).dtype, np.integer) for x in outputs):
        raise RuntimeError(
            "RKNNLite returned integer outputs. This script expects the Python runtime to return float outputs; "
            "use a model/runtime setting that returns float outputs, or port the C++ quantized postprocess."
        )

    boxes, scores, class_ids = [], [], []
    per_branch = len(outputs) // 3
    for i in range(3):
        cls_out = outputs[i * per_branch]
        box_out = outputs[i * per_branch + 1]

        branch_boxes = flatten_spatial(boxes_from_distribution(box_out, img_size=img_size))
        branch_scores = flatten_spatial(cls_out)[:, :num_classes]
        branch_class_ids = branch_scores.argmax(axis=1)
        branch_scores = branch_scores.max(axis=1)
        keep = branch_scores >= conf_thres

        boxes.append(branch_boxes[keep])
        scores.append(branch_scores[keep])
        class_ids.append(branch_class_ids[keep])

    if not boxes or sum(len(x) for x in boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int64)

    boxes = np.concatenate(boxes, axis=0)
    scores = np.concatenate(scores, axis=0)
    class_ids = np.concatenate(class_ids, axis=0)

    keep_all = []
    for cls_id in np.unique(class_ids):
        idx = np.where(class_ids == cls_id)[0]
        keep_all.extend(idx[nms(boxes[idx], scores[idx], iou_thres)].tolist())

    keep_all = np.array(keep_all, dtype=np.int64)
    keep_all = keep_all[scores[keep_all].argsort()[::-1]][:max_det]
    boxes, scores, class_ids = boxes[keep_all], scores[keep_all], class_ids[keep_all]

    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    orig_h, orig_w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
    return boxes, scores, class_ids


def create_runtime(model_path, target="rk3588", core="auto"):
    try:
        from rknnlite.api import RKNNLite

        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"load_rknn failed: {model_path}")

        core_attr = {
            "auto": "NPU_CORE_AUTO",
            "0": "NPU_CORE_0",
            "1": "NPU_CORE_1",
            "2": "NPU_CORE_2",
            "01": "NPU_CORE_0_1",
            "012": "NPU_CORE_0_1_2",
            "all": "NPU_CORE_0_1_2",
        }.get(core, "NPU_CORE_AUTO")
        core_mask = getattr(RKNNLite, core_attr, None)
        ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        return rknn
    except ImportError:
        from rknn.api import RKNN

        rknn = RKNN(verbose=False)
        if rknn.load_rknn(str(model_path)) != 0:
            raise RuntimeError(f"load_rknn failed: {model_path}")
        if rknn.init_runtime(target=target) != 0:
            raise RuntimeError(f"init_runtime failed for target={target}")
        return rknn


def draw_detections(image, boxes, scores, class_ids, labels):
    print("{:^14} {:^8}  {}".format("class", "score", "xmin, ymin, xmax, ymax"))
    print("-" * 56)
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int).tolist()
        label = labels[int(cls_id)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print("{:^14} {:^8.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(label, float(score), x1, y1, x2, y2))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-World RKNNLite inference without torch/transformers.")
    parser.add_argument("--model", required=True, help="Path to yolo_world_v2s_i8.rknn")
    parser.add_argument("--text-features", default=None, help="Optional precomputed text features, e.g. coco_text_outp.npy")
    parser.add_argument("--text-model", default=None, help="Path to clip_text_fp16.rknn for dynamic open-vocabulary classes")
    parser.add_argument("--vocab-header", default=None, help="Path to cpp/tokenizer/clip_vocab.h")
    parser.add_argument("--classes", default=None, help="Comma-separated class prompts, e.g. 'helmet,reflective vest,person'")
    parser.add_argument("--class-file", default=None, help="Text file with one class prompt per line")
    parser.add_argument("--img", required=True, help="Input image path")
    parser.add_argument("--labels", default=None, help="Optional label file for precomputed COCO features")
    parser.add_argument("--target", default="rk3588", help="Target used only when falling back to rknn.api")
    parser.add_argument("--core", default="auto", choices=("auto", "0", "1", "2", "01", "012", "all"), help="RKNNLite NPU core mask")
    parser.add_argument("--layout", default="nhwc", choices=("nhwc", "nchw"), help="Image tensor layout")
    parser.add_argument("--input-dtype", default="uint8", choices=("uint8", "float32"), help="Image tensor dtype")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--out", default="result.jpg", help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.text_features:
        labels = load_labels(args.labels)
        text_features = load_text_features(args.text_features)
    else:
        if not args.text_model or not args.vocab_header:
            raise ValueError("dynamic classes require --text-model and --vocab-header, or provide --text-features")
        labels = parse_classes(args)
        text_features = run_clip_text(args.text_model, labels, args.vocab_header, target=args.target, core=args.core)

    img, orig, scale, pad = preprocess_image(args.img, layout=args.layout, dtype=args.input_dtype)

    rknn = create_runtime(args.model, target=args.target, core=args.core)
    try:
        outputs = rknn.inference(inputs=[img, text_features])
    finally:
        rknn.release()

    boxes, scores, class_ids = postprocess(
        outputs,
        orig.shape,
        scale,
        pad,
        conf_thres=args.conf,
        iou_thres=args.iou,
        num_classes=len(labels),
    )
    if len(boxes):
        draw_detections(orig, boxes, scores, class_ids, labels)
    else:
        print("No detections.")

    cv2.imwrite(args.out, orig)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
