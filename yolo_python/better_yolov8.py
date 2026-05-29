import os
import cv2
from rknnlite.api import RKNNLite
import numpy as np

RKNN_MODEL = "yolov8n.rknn"
IMG_FOLDER = "./"
RESULT_PATH = './'

CLASSES = [
    'person','bicycle','car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

OBJ_THRESH = 0.45
NMS_THRESH = 0.45
MODEL_SIZE = (640, 640)

color_palette = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def letter_box(im, new_shape, pad_color=(0, 0, 0)):
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    info = (r, (left, top))
    return im, info

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    keep = np.array(keep)
    return keep

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]

    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)

    # 修正 stride 计算：x 对应 grid_w，y 对应 grid_h
    stride = np.array([
        MODEL_SIZE[1] // grid_w,   # x stride
        MODEL_SIZE[0] // grid_h    # y stride
    ]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw_detections(img, left, top, right, bottom, score, class_id):
    # 颜色改成 OpenCV 更稳的整型 tuple
    color = tuple(map(int, color_palette[class_id]))

    # 坐标全部转 int，避免 OpenCV 报错
    left, top, right, bottom = map(int, [left, top, right, bottom])

    cv2.rectangle(img, (left, top), (right, bottom), color, 2)

    label = f"{CLASSES[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    label_x = left
    label_y = top - 10 if top - 10 > label_height else top + label_height + 2

    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + 2),
        color,
        cv2.FILLED
    )
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

def draw(image, boxes, scores, classes, ratio, pad_offset):
    img_h, img_w = image.shape[:2]
    left_pad, top_pad = pad_offset

    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box

        x1 -= left_pad
        y1 -= top_pad
        x2 -= left_pad
        y2 -= top_pad

        x1 /= ratio
        y1 /= ratio
        x2 /= ratio
        y2 /= ratio

        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        draw_detections(image, x1, y1, x2, y2, score, cl)

if __name__ == '__main__':
    rknn_lite = RKNNLite()

    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    os.makedirs(RESULT_PATH, exist_ok=True)

    img_list = os.listdir(IMG_FOLDER)
    for i in range(len(img_list)):
        img_name = img_list[i]
        img_path = os.path.join(IMG_FOLDER, img_name)

        if not os.path.exists(img_path) or img_name.lower().endswith(('.txt', '.json')):
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        pad_color = (0, 0, 0)
        img, info = letter_box(
            im=img_src.copy(),
            new_shape=(MODEL_SIZE[1], MODEL_SIZE[0]),
            pad_color=pad_color
        )
        ratio, pad_offset = info

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 连续内存，给 RKNN 推理更稳
        input_data = np.expand_dims(img, axis=0)
        input_data = np.ascontiguousarray(input_data)

        outputs = rknn_lite.inference([input_data])
        boxes, classes, scores = post_process(outputs)

        img_p = img_src.copy()

        if boxes is not None:
            draw(img_p, boxes, scores, classes, ratio, pad_offset)

        result_path = os.path.join(RESULT_PATH, img_name)
        cv2.imwrite(result_path, img_p)
        print('{}/{} Detection result save to {}'.format(i + 1, len(img_list), result_path))

    rknn_lite.release()
