import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MjpegServer:
    def __init__(self, host="0.0.0.0", port=8080, jpeg_quality=80):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._condition = threading.Condition()
        self._httpd = None
        self._http_thread = None
        self._encoder_thread = None
        self._pending_frame = None
        self._jpeg = None
        self._sequence = 0
        self._clients = 0
        self._metrics = {}
        self._stopping = False

    @property
    def is_running(self):
        return self._httpd is not None and not self._stopping

    def start(self):
        if self.is_running:
            return
        self._stopping = False
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler_class())
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="mjpeg-http",
            daemon=True,
        )
        self._encoder_thread = threading.Thread(
            target=self._encode_frames,
            name="mjpeg-encoder",
            daemon=True,
        )
        self._http_thread.start()
        self._encoder_thread.start()

    def stop(self):
        httpd = self._httpd
        if httpd is None:
            return
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        httpd.shutdown()
        httpd.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2)
        self._httpd = None
        self._http_thread = None
        self._encoder_thread = None
        self._pending_frame = None
        self._jpeg = None

    def publish(self, frame_bgr, metrics):
        with self._condition:
            self._metrics = dict(metrics)
            if not self.is_running or self._clients == 0:
                return
            self._pending_frame = frame_bgr.copy()
            self._condition.notify_all()

    def local_url(self):
        return "http://{}:{}".format(_local_ip(), self.port)

    def _encode_frames(self):
        cv2 = None

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending_frame is not None or self._stopping
                )
                if self._stopping:
                    return
                frame = self._pending_frame
                self._pending_frame = None
            if cv2 is None:
                import cv2 as cv2_module

                cv2 = cv2_module
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                continue
            with self._condition:
                self._jpeg = encoded.tobytes()
                self._sequence += 1
                self._condition.notify_all()

    def _wait_for_jpeg(self, sequence, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._sequence <= sequence and not self._stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._stopping or self._jpeg is None:
                return None
            return self._sequence, self._jpeg

    def _add_client(self):
        with self._condition:
            self._clients += 1

    def _remove_client(self):
        with self._condition:
            self._clients = max(0, self._clients - 1)

    def _status(self):
        with self._condition:
            metrics = dict(self._metrics)
            metrics["viewers"] = self._clients
            return metrics

    def _handler_class(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    self._send_page()
                elif self.path == "/stream.mjpg":
                    self._send_stream()
                elif self.path == "/status":
                    self._send_status()
                else:
                    self.send_error(404)

            def _send_page(self):
                body = _VIEWER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_status(self):
                body = json.dumps(
                    service._status(),
                    default=_json_value,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_stream(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                service._add_client()
                sequence = -1
                try:
                    while service.is_running:
                        item = service._wait_for_jpeg(sequence)
                        if item is None:
                            continue
                        sequence, jpeg = item
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            "Content-Length: {}\r\n\r\n".format(len(jpeg)).encode(
                                "ascii"
                            )
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    service._remove_client()

            def log_message(self, _format, *_args):
                pass

        return Handler


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


def _json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>ELF2 Vision</title>
<style>
html,body{margin:0;min-height:100%;background:#101418;color:#e7edf2;font-family:sans-serif}
main{display:flex;min-height:100vh;flex-direction:column}
header{padding:12px 16px;font-size:18px;font-weight:700;background:#1b232a}
.video{display:flex;flex:1;align-items:center;justify-content:center;min-height:0}
img{display:block;width:100%;height:auto;max-height:calc(100vh - 94px);object-fit:contain;background:#000}
#status{padding:10px 16px;background:#1b232a;color:#b8c6d0;font-size:14px}
</style>
</head>
<body>
<main>
<header>ELF2 Vision</header>
<div class="video"><img src="/stream.mjpg" alt="ELF2 vision stream"></div>
<div id="status">Waiting for task output...</div>
</main>
<script>
async function updateStatus(){
  try{
    const m=await fetch('/status',{cache:'no-store'}).then(r=>r.json());
    const fps=Number(m.fps||0).toFixed(2);
    document.getElementById('status').textContent=
      `Task: ${m.task_name||'--'} | FPS: ${fps} | Resolution: ${m.width||'--'}x${m.height||'--'}`;
  }catch(e){}
}
setInterval(updateStatus,1000);updateStatus();
</script>
</body>
</html>
"""
