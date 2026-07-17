import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


class MediaMtxPublisher:
    def __init__(
        self,
        web_port=8889,
        rtsp_port=8554,
        stream_path="vision",
        width=1280,
        height=720,
        fps=30,
        bitrate_kbps=4000,
    ):
        self.web_port = web_port
        self.rtsp_port = rtsp_port
        self.stream_path = stream_path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self._condition = threading.Condition()
        self._pending_frame = None
        self._frame_shape = None
        self._metrics = {}
        self._stopping = False
        self._server_process = None
        self._server_log = deque(maxlen=80)
        self._server_log_thread = None
        self._pipeline = None
        self._appsrc = None
        self._bus = None
        self._glib = None
        self._gst = None
        self._glib_loop = None
        self._glib_thread = None
        self._feeder_thread = None
        self._last_error = ""

    @property
    def is_running(self):
        process = self._server_process
        return (
            not self._stopping
            and self._pipeline is not None
            and process is not None
            and process.poll() is None
        )

    def start(self):
        if self.is_running:
            return
        self._load_gstreamer()
        self._stopping = False
        self._last_error = ""
        try:
            self._start_mediamtx()
            self._build_pipeline()
            self._start_glib()
            self._feeder_thread = threading.Thread(
                target=self._feed_frames,
                name="mediamtx-feeder",
                daemon=True,
            )
            self._feeder_thread.start()
            result = self._pipeline.set_state(self._gst.State.PLAYING)
            if result == self._gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start MediaMTX publisher pipeline")
        except Exception:
            self.stop()
            raise

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()

        if self._feeder_thread is not None:
            self._feeder_thread.join(timeout=2)
        self._feeder_thread = None

        pipeline = self._pipeline
        self._pipeline = None
        self._appsrc = None
        if self._bus is not None:
            self._bus.remove_signal_watch()
        self._bus = None
        if pipeline is not None and self._gst is not None:
            pipeline.set_state(self._gst.State.NULL)

        if self._glib_loop is not None:
            self._glib_loop.quit()
        if self._glib_thread is not None:
            self._glib_thread.join(timeout=2)
        self._glib_thread = None
        self._glib_loop = None

        process = self._server_process
        self._server_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._server_log_thread is not None:
            self._server_log_thread.join(timeout=1)
        self._server_log_thread = None
        self._pending_frame = None

    def publish(self, frame_bgr, metrics):
        with self._condition:
            self._metrics = dict(metrics)
            self._frame_shape = frame_bgr.shape
            if not self.is_running:
                return
            self._pending_frame = frame_bgr.copy()
            self._condition.notify_all()

    def blackout(self):
        with self._condition:
            shape = self._frame_shape or (480, 640, 3)
            self._metrics = {
                "task_name": "stopped",
                "fps": 0.0,
                "width": shape[1],
                "height": shape[0],
            }
            if not self.is_running:
                return
        import numpy as np

        frame = np.zeros(shape, dtype=np.uint8)
        with self._condition:
            if not self.is_running:
                return
            self._pending_frame = frame
            self._condition.notify_all()

    def local_url(self):
        return "http://{}:{}/{}".format(
            _local_ip(),
            self.web_port,
            self.stream_path,
        )

    def _load_gstreamer(self):
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import GLib, Gst
        except (ImportError, ValueError) as exc:
            raise RuntimeError("GStreamer Python bindings are missing") from exc
        Gst.init(None)
        if Gst.ElementFactory.find("rtspclientsink") is None:
            raise RuntimeError(
                "rtspclientsink is missing. Install gstreamer1.0-rtsp"
            )
        self._gst = Gst
        self._glib = GLib

    def _start_mediamtx(self):
        binary = _find_mediamtx()
        config_path = Path(__file__).resolve().with_name("mediamtx.yml")
        if not config_path.exists():
            raise RuntimeError("MediaMTX config not found: {}".format(config_path))
        if _port_is_open("127.0.0.1", self.rtsp_port) or _port_is_open(
            "127.0.0.1", self.web_port
        ):
            raise RuntimeError(
                "MediaMTX port is already in use (RTSP {}, WebRTC {})".format(
                    self.rtsp_port,
                    self.web_port,
                )
            )

        environment = os.environ.copy()
        environment["MTX_RTSPADDRESS"] = ":{}".format(self.rtsp_port)
        environment["MTX_WEBRTCADDRESS"] = ":{}".format(self.web_port)
        process = subprocess.Popen(
            [str(binary), str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
        self._server_process = process
        self._server_log_thread = threading.Thread(
            target=self._read_server_log,
            name="mediamtx-log",
            daemon=True,
        )
        self._server_log_thread.start()

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = "\n".join(self._server_log) or "no output"
                raise RuntimeError("MediaMTX exited during startup: {}".format(detail))
            if _port_is_open("127.0.0.1", self.rtsp_port) and _port_is_open(
                "127.0.0.1", self.web_port
            ):
                return
            time.sleep(0.1)
        raise RuntimeError("Timed out waiting for MediaMTX ports")

    def _read_server_log(self):
        process = self._server_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.rstrip()
            if text:
                self._server_log.append(text)

    def _build_pipeline(self):
        description = (
            "appsrc name=source is-live=true block=false do-timestamp=true format=time "
            "caps=video/x-raw,format=BGR,width=640,height=480,framerate={fps}/1 "
            "! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 "
            "! videoconvert ! videoscale add-borders=true "
            "! video/x-raw,format=NV12,width={width},height={height},pixel-aspect-ratio=1/1 "
            "! mpph264enc bps={bps} gop={gop} rc-mode=cbr profile=baseline "
            "header-mode=each-idr max-pending=2 "
            "! h264parse config-interval=-1 "
            "! video/x-h264,stream-format=byte-stream,alignment=au "
            "! rtspclientsink location=rtsp://127.0.0.1:{rtsp_port}/{stream_path} "
            "protocols=tcp latency=0"
        ).format(
            fps=self.fps,
            width=self.width,
            height=self.height,
            bps=self.bitrate_kbps * 1000,
            gop=self.fps,
            rtsp_port=self.rtsp_port,
            stream_path=self.stream_path,
        )
        try:
            self._pipeline = self._gst.parse_launch(description)
        except self._glib.Error as exc:
            raise RuntimeError(
                "Cannot build MediaMTX publisher pipeline: {}".format(exc)
            ) from exc
        self._appsrc = self._pipeline.get_by_name("source")
        if self._appsrc is None:
            raise RuntimeError("MediaMTX publisher appsrc was not created")
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", self._on_bus_message)

    def _start_glib(self):
        self._glib_loop = self._glib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run,
            name="mediamtx-glib",
            daemon=True,
        )
        self._glib_thread.start()

    def _feed_frames(self):
        current_shape = None
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending_frame is not None or self._stopping
                )
                if self._stopping:
                    return
                frame = self._pending_frame
                self._pending_frame = None
            height, width = frame.shape[:2]
            shape = (height, width)
            if shape != current_shape:
                caps = self._gst.Caps.from_string(
                    "video/x-raw,format=BGR,width={},height={},framerate={}/1".format(
                        width,
                        height,
                        self.fps,
                    )
                )
                self._appsrc.set_property("caps", caps)
                current_shape = shape
            data = frame.tobytes()
            buffer = self._gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            buffer.duration = self._gst.SECOND // self.fps
            result = self._appsrc.emit("push-buffer", buffer)
            if result != self._gst.FlowReturn.OK:
                self._last_error = "appsrc push failed: {}".format(result.value_nick)

    def _on_bus_message(self, _bus, message):
        if message.type == self._gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self._last_error = "{} ({})".format(error, debug or "no debug")


def _find_mediamtx():
    candidates = [
        Path.home() / ".local" / "bin" / "mediamtx",
        Path("/usr/local/bin/mediamtx"),
    ]
    command = shutil.which("mediamtx")
    if command:
        candidates.insert(0, Path(command))
    for candidate in candidates:
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
    raise RuntimeError("MediaMTX not found. Install it at ~/.local/bin/mediamtx")


def _port_is_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()
