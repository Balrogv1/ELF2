import asyncio
import concurrent.futures
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class WebRtcServer:
    def __init__(
        self,
        host="0.0.0.0",
        port=8080,
        signal_port=None,
        width=1280,
        height=720,
        fps=30,
        bitrate_kbps=4000,
    ):
        self.host = host
        self.port = port
        self.signal_port = signal_port or port + 1
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self._condition = threading.Condition()
        self._pending_frame = None
        self._frame_shape = None
        self._metrics = {}
        self._stopping = False
        self._httpd = None
        self._http_thread = None
        self._signal_thread = None
        self._signal_loop = None
        self._signal_server = None
        self._websocket = None
        self._glib_thread = None
        self._glib_loop = None
        self._feeder_thread = None
        self._pipeline = None
        self._appsrc = None
        self._webrtc = None
        self._gst = None
        self._gst_sdp = None
        self._gst_webrtc = None
        self._glib = None
        self._last_error = ""
        self._client_error = ""
        self._client_state = ""
        self._peer_states = {}
        self._local_ice_count = 0
        self._remote_ice_count = 0

    @property
    def is_running(self):
        return self._httpd is not None and not self._stopping

    def start(self):
        if self.is_running:
            return
        self._load_gstreamer()
        self._stopping = False
        try:
            self._build_pipeline()
            self._start_glib()
            self._start_signaling()
            self._start_http()
            self._feeder_thread = threading.Thread(
                target=self._feed_frames,
                name="webrtc-feeder",
                daemon=True,
            )
            self._feeder_thread.start()
            result = self._pipeline.set_state(self._gst.State.PLAYING)
            if result == self._gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start WebRTC GStreamer pipeline")
        except Exception:
            self.stop()
            raise

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()

        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
        self._http_thread = None

        self._stop_signaling()

        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None and self._gst is not None:
            pipeline.set_state(self._gst.State.NULL)

        if self._glib_loop is not None:
            self._glib_loop.quit()
        if self._glib_thread is not None:
            self._glib_thread.join(timeout=2)
        self._glib_thread = None
        self._glib_loop = None

        if self._feeder_thread is not None:
            self._feeder_thread.join(timeout=2)
        self._feeder_thread = None
        self._pending_frame = None
        self._appsrc = None
        self._webrtc = None

    def publish(self, frame_bgr, metrics):
        with self._condition:
            self._metrics = dict(metrics)
            self._frame_shape = frame_bgr.shape
            if not self.is_running or self._websocket is None:
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
            if not self.is_running or self._websocket is None:
                return
        import numpy as np

        frame = np.zeros(shape, dtype=np.uint8)
        with self._condition:
            if not self.is_running or self._websocket is None:
                return
            self._pending_frame = frame
            self._condition.notify_all()

    def local_url(self):
        return "http://{}:{}".format(_local_ip(), self.port)

    def _load_gstreamer(self):
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstSdp", "1.0")
            gi.require_version("GstWebRTC", "1.0")
            from gi.repository import GLib, Gst, GstSdp, GstWebRTC
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "WebRTC GI bindings missing. Install gir1.2-gst-plugins-bad-1.0"
            ) from exc
        Gst.init(None)
        self._gst = Gst
        self._gst_sdp = GstSdp
        self._gst_webrtc = GstWebRTC
        self._glib = GLib

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
            "! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 mtu=1200 "
            "! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96 "
            "! webrtcbin name=webrtc bundle-policy=max-bundle"
        ).format(
            fps=self.fps,
            width=self.width,
            height=self.height,
            bps=self.bitrate_kbps * 1000,
            gop=self.fps,
        )
        try:
            self._pipeline = self._gst.parse_launch(description)
        except self._glib.Error as exc:
            raise RuntimeError("Cannot build WebRTC pipeline: {}".format(exc)) from exc
        self._appsrc = self._pipeline.get_by_name("source")
        self._webrtc = self._pipeline.get_by_name("webrtc")
        if self._appsrc is None or self._webrtc is None:
            raise RuntimeError("WebRTC pipeline elements were not created")
        self._webrtc.connect("on-ice-candidate", self._on_local_ice)
        for property_name in (
            "signaling-state",
            "ice-gathering-state",
            "ice-connection-state",
            "connection-state",
        ):
            self._webrtc.connect(
                "notify::{}".format(property_name),
                self._on_peer_state_changed,
                property_name,
            )
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def _start_glib(self):
        self._glib_loop = self._glib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run,
            name="webrtc-glib",
            daemon=True,
        )
        self._glib_thread.start()

    def _start_http(self):
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler_class())
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="webrtc-http",
            daemon=True,
        )
        self._http_thread.start()

    def _start_signaling(self):
        ready = threading.Event()
        error = []

        def run():
            try:
                import websockets

                major_version = int(websockets.__version__.split(".", 1)[0])
                if major_version < 10:
                    raise RuntimeError("websockets 10.4 or newer is required")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._signal_loop = loop
                self._signal_server = loop.run_until_complete(
                    websockets.serve(self._handle_websocket, self.host, self.signal_port)
                )
            except Exception as exc:
                error.append(exc)
                ready.set()
                return
            ready.set()
            loop.run_forever()
            loop.run_until_complete(self._close_signaling())
            loop.close()

        self._signal_thread = threading.Thread(
            target=run,
            name="webrtc-signaling",
            daemon=True,
        )
        self._signal_thread.start()
        if not ready.wait(5):
            raise RuntimeError("WebRTC signaling server start timed out")
        if error:
            raise RuntimeError("Cannot start WebRTC signaling: {}".format(error[0]))

    def _stop_signaling(self):
        loop = self._signal_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._signal_thread is not None:
            self._signal_thread.join(timeout=3)
        self._signal_thread = None
        self._signal_loop = None
        self._signal_server = None
        self._websocket = None

    async def _close_signaling(self):
        websocket = self._websocket
        if websocket is not None:
            await websocket.close()
        server = self._signal_server
        if server is not None:
            server.close()
            await server.wait_closed()

    async def _handle_websocket(self, websocket, _path):
        previous = self._websocket
        if previous is not None and previous is not websocket:
            await previous.close(code=1012, reason="New viewer connected")
        self._websocket = websocket
        self._last_error = ""
        self._client_error = ""
        self._client_state = ""
        self._local_ice_count = 0
        self._remote_ice_count = 0
        try:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                if message.get("type") == "offer":
                    answer = await self._create_answer(message["sdp"])
                    await websocket.send(json.dumps({"type": "answer", "sdp": answer}))
                elif message.get("type") == "ice":
                    self._add_remote_ice(
                        int(message.get("sdpMLineIndex", 0)),
                        message["candidate"],
                    )
        except Exception as exc:
            self._last_error = str(exc)
        finally:
            if self._websocket is websocket:
                self._websocket = None

    async def _create_answer(self, offer_sdp):
        future = concurrent.futures.Future()
        self._glib.idle_add(self._set_offer_and_create_answer, offer_sdp, future)
        return await asyncio.wrap_future(future)

    def _set_offer_and_create_answer(self, offer_sdp, future):
        try:
            result, sdp_message = self._gst_sdp.SDPMessage.new_from_text(offer_sdp)
            if result != self._gst_sdp.SDPResult.OK:
                raise RuntimeError("Invalid browser SDP offer")
            offer = self._gst_webrtc.WebRTCSessionDescription.new(
                self._gst_webrtc.WebRTCSDPType.OFFER,
                sdp_message,
            )
            promise = self._gst.Promise.new()
            self._webrtc.emit("set-remote-description", offer, promise)
            promise.interrupt()
            answer_promise = self._gst.Promise.new_with_change_func(
                self._on_answer_created,
                future,
                None,
            )
            self._webrtc.emit("create-answer", None, answer_promise)
        except Exception as exc:
            future.set_exception(exc)
        return False

    def _on_answer_created(self, promise, future, _unused):
        try:
            reply = promise.get_reply()
            answer = reply.get_value("answer")
            local_promise = self._gst.Promise.new()
            self._webrtc.emit("set-local-description", answer, local_promise)
            local_promise.interrupt()
            future.set_result(answer.sdp.as_text())
        except Exception as exc:
            future.set_exception(exc)

    def _add_remote_ice(self, line_index, candidate):
        self._remote_ice_count += 1
        self._glib.idle_add(
            self._webrtc.emit,
            "add-ice-candidate",
            line_index,
            candidate,
        )

    def _on_local_ice(self, _webrtc, line_index, candidate):
        self._local_ice_count += 1
        loop = self._signal_loop
        websocket = self._websocket
        if loop is None or websocket is None:
            return
        message = json.dumps(
            {
                "type": "ice",
                "candidate": candidate,
                "sdpMLineIndex": line_index,
            }
        )
        asyncio.run_coroutine_threadsafe(websocket.send(message), loop)

    def _on_peer_state_changed(self, webrtc, _property, property_name):
        value = webrtc.get_property(property_name)
        self._peer_states[property_name] = getattr(value, "value_nick", str(value))

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

    def _status(self):
        with self._condition:
            status = dict(self._metrics)
        status.update(
            {
                "viewer_connected": self._websocket is not None,
                "codec": "H.264 / mpph264enc",
                "last_error": self._last_error,
                "client_error": self._client_error,
                "client_state": self._client_state,
                "local_ice_candidates": self._local_ice_count,
                "remote_ice_candidates": self._remote_ice_count,
                "peer_states": dict(self._peer_states),
            }
        )
        return status

    def _handler_class(self):
        service = self
        page = _VIEWER_HTML.replace("__SIGNAL_PORT__", str(self.signal_port)).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
                elif self.path == "/status":
                    body = json.dumps(service._status(), default=_json_value).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path not in ("/client-error", "/client-state"):
                    self.send_error(404)
                    return
                length = min(int(self.headers.get("Content-Length", "0")), 4096)
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if self.path == "/client-error":
                        service._client_error = str(payload.get("error", ""))[:500]
                    else:
                        service._client_state = str(payload.get("state", ""))[:500]
                except (UnicodeDecodeError, ValueError):
                    service._client_error = "Invalid browser report"
                self.send_response(204)
                self.end_headers()

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
<title>ELF2 WebRTC</title>
<style>
html,body{margin:0;min-height:100%;background:#101418;color:#e7edf2;font-family:sans-serif}
main{display:flex;min-height:100vh;flex-direction:column}
header{padding:12px 16px;font-size:18px;font-weight:700;background:#1b232a}
.video{display:flex;flex:1;align-items:center;justify-content:center;min-height:0}
video{display:block;width:100%;height:auto;max-height:calc(100vh - 94px);background:#000}
#status{padding:10px 16px;background:#1b232a;color:#b8c6d0;font-size:14px}
</style>
</head>
<body>
<main>
<header>ELF2 WebRTC</header>
<div class="video"><video id="video" autoplay playsinline muted></video></div>
<div id="status">Connecting...</div>
</main>
<script>
const statusEl=document.getElementById('status');
const video=document.getElementById('video');
let pc=null;
let ws=null;
let remoteReady=false;
const pendingIce=[];

async function reportError(error){
  const message=error&&error.message?error.message:String(error);
  statusEl.textContent=message;
  try{
    await fetch('/client-error',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({error:message})});
  }catch(e){}
}

async function reportState(){
  if(!pc)return;
  const state=`ice=${pc.iceConnectionState}, connection=${pc.connectionState}, signaling=${pc.signalingState}`;
  try{
    await fetch('/client-state',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({state})});
  }catch(e){}
}

async function connect(){
  try{
    if(!window.RTCPeerConnection)throw new Error('WebRTC is not supported by this browser');
    pc=new RTCPeerConnection({iceServers:[]});
    const transceiver=pc.addTransceiver('video',{direction:'recvonly'});
    if(transceiver.setCodecPreferences&&window.RTCRtpReceiver&&RTCRtpReceiver.getCapabilities){
      const capabilities=RTCRtpReceiver.getCapabilities('video');
      const codecs=capabilities&&capabilities.codecs?capabilities.codecs
        .filter(codec=>codec.mimeType.toLowerCase()==='video/h264'):[];
      if(codecs.length)transceiver.setCodecPreferences(codecs);
    }
    pc.ontrack=e=>{video.srcObject=e.streams[0];video.play().catch(reportError);};
    pc.oniceconnectionstatechange=reportState;
    pc.onconnectionstatechange=reportState;
    pc.onsignalingstatechange=reportState;
    ws=new WebSocket(`ws://${location.hostname}:__SIGNAL_PORT__`);
    pc.onicecandidate=e=>{
      if(e.candidate&&ws.readyState===WebSocket.OPEN){
        ws.send(JSON.stringify({type:'ice',candidate:e.candidate.candidate,
          sdpMLineIndex:e.candidate.sdpMLineIndex||0}));
      }
    };
    ws.onopen=async()=>{
      try{
        const offer=await pc.createOffer();
        await pc.setLocalDescription(offer);
        ws.send(JSON.stringify({type:'offer',sdp:offer.sdp}));
      }catch(error){reportError(error);}
    };
    ws.onmessage=async event=>{
      try{
        const message=JSON.parse(event.data);
        if(message.type==='answer'){
          await pc.setRemoteDescription({type:'answer',sdp:message.sdp});
          remoteReady=true;
          for(const candidate of pendingIce)await pc.addIceCandidate(candidate);
          pendingIce.length=0;
        }else if(message.type==='ice'){
          const candidate={candidate:message.candidate,
            sdpMLineIndex:message.sdpMLineIndex||0};
          if(remoteReady)await pc.addIceCandidate(candidate);else pendingIce.push(candidate);
        }
      }catch(error){reportError(error);}
    };
    ws.onerror=()=>reportError('WebSocket connection failed on port __SIGNAL_PORT__');
    ws.onclose=event=>{
      if(!event.wasClean)reportError(`WebSocket closed: ${event.code}`);
    };
  }catch(error){reportError(error);}
}

async function updateStatus(){
  try{
    const m=await fetch('/status',{cache:'no-store'}).then(r=>r.json());
    const parts=[`Task: ${m.task_name||'--'}`,`FPS: ${Number(m.fps||0).toFixed(2)}`,
      `Resolution: ${m.width||'--'}x${m.height||'--'}`,m.codec||'H.264'];
    if(m.last_error)parts.push(m.last_error);
    if(m.client_error)parts.push(m.client_error);
    if(m.client_state)parts.push(m.client_state);
    statusEl.textContent=parts.join(' | ');
  }catch(e){}
}
window.addEventListener('error',event=>reportError(event.error||event.message));
window.addEventListener('unhandledrejection',event=>reportError(event.reason));
connect();setInterval(updateStatus,1000);updateStatus();
</script>
</body>
</html>
"""
