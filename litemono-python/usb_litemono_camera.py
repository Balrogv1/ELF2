import cv2
import time
import sys
import argparse
import numpy as np
from rknnlite.api import RKNNLite


MODEL_W = 640
MODEL_H = 192


def parse_camera_arg(camera_arg):
    if isinstance(camera_arg, str) and camera_arg.isdigit():
        return int(camera_arg)
    return camera_arg


def disp_to_colormap(disp, out_w, out_h):
    """
    disp: [192, 640], float32
    return: BGR color image for OpenCV display
    """
    disp = disp.astype(np.float32)

    # 避免极端值影响显示，参考 Lite-Mono test_simple.py 的 95% 分位数思路
    vmin = np.min(disp)
    vmax = np.percentile(disp, 95)

    if vmax - vmin < 1e-6:
        disp_norm = np.zeros_like(disp, dtype=np.uint8)
    else:
        disp_norm = (disp - vmin) / (vmax - vmin)
        disp_norm = np.clip(disp_norm, 0, 1)
        disp_norm = (disp_norm * 255).astype(np.uint8)

    disp_resized = cv2.resize(disp_norm, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    colormap = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET)
    disp_color = cv2.applyColorMap(disp_resized, colormap)

    return disp_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lite_mono_tiny_640x192.rknn")
    parser.add_argument("--camera", type=str, default="21")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument("--cam_fps", type=int, default=30)
    parser.add_argument("--no_window", action="store_true")
    args = parser.parse_args()

    # 1. 初始化 RKNNLite
    rknn = RKNNLite()

    print("--> Load RKNN model")
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print("Load RKNN model failed")
        exit(ret)
    print("done")

    print("--> Init runtime")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print("Init runtime failed")
        rknn.release()
        exit(ret)
    print("done")

    # 2. 打开摄像头
    camera_source = parse_camera_arg(args.camera)
    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Cannot open camera: {args.camera}")
        rknn.release()
        exit(-1)

    # 推荐 MJPG，高分辨率下更容易稳定 30fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FPS, args.cam_fps)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {real_w}x{real_h}, camera_fps={real_fps:.1f}")
    print("Press q or ESC to quit.")
    print()

    fps_smooth = None

    try:
        while True:
            loop_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("\nRead camera frame failed")
                break

            original_h, original_w = frame.shape[:2]

            # 3. 预处理：BGR -> RGB, resize, /255, HWC -> CHW
            img = cv2.resize(frame, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            input_data = np.expand_dims(img, axis=0)  
            input_data = np.ascontiguousarray(input_data)

            # 4. 推理
            infer_start = time.perf_counter()
            outputs = rknn.inference(
                inputs=[input_data],
                data_type="float32",
                data_format="nhwc"
            )
            infer_end = time.perf_counter()

            disp = outputs[0]
            disp = np.squeeze(disp)  # [1,1,192,640] -> [192,640]

            # 5. 伪彩色显示
            disp_color = disp_to_colormap(disp, original_w, original_h)

            # 左边原图，右边深度图
            show = np.hstack([frame, disp_color])

            # 6. FPS
            loop_end = time.perf_counter()
            frame_time = loop_end - loop_start
            infer_time = infer_end - infer_start

            fps = 1.0 / frame_time if frame_time > 0 else 0.0
            if fps_smooth is None:
                fps_smooth = fps
            else:
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps

            sys.stdout.write(
                f"\rFPS: {fps_smooth:6.2f} | "
                f"Infer: {infer_time * 1000:6.1f} ms | "
                f"disp min/max: {disp.min():.4f}/{disp.max():.4f}"
            )
            sys.stdout.flush()

            if not args.no_window:
                cv2.imshow("Lite-Mono RKNN | RGB + Disparity", show)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("\nReleasing resources...")
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        rknn.release()
        print("done")


if __name__ == "__main__":
    main()
