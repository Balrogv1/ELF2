import importlib
import json
import os
import shlex
import signal
import sys
from pathlib import Path

import PyQt5


def configure_qt_plugins():
    pyqt_dir = Path(PyQt5.__file__).resolve().parent
    candidates = [
        pyqt_dir / "Qt5" / "plugins" / "platforms",
        pyqt_dir / "Qt" / "plugins" / "platforms",
    ]
    for candidate in candidates:
        if candidate.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(candidate)
            break
    os.environ.pop("QT_PLUGIN_PATH", None)


configure_qt_plugins()

from PyQt5.QtCore import QProcess, QTimer, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from task_registry import TASKS, get_task_class
from video_worker import VideoWorker


RESOLUTIONS = {
    "640 x 480": (640, 480),
    "1280 x 720": (1280, 720),
    "1920 x 1080": (1920, 1080),
}

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._pixmap = None
        self._message = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #101418; color: #9fb0bf; font-size: 22px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.set_idle_message()

    def set_frame(self, frame_bgr):
        self._message = None
        rgb = frame_bgr[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._update_pixmap()

    def set_message(self, message):
        self._pixmap = None
        self._message = message
        self._render_message_pixmap()

    def set_idle_message(self):
        self.set_message("Select a task and press Start")

    def set_blackout(self):
        self.set_message("")

    def is_showing_frame(self):
        return self._message is None and self._pixmap is not None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._message is not None:
            self._render_message_pixmap()
        else:
            self._update_pixmap()

    def _render_message_pixmap(self):
        width = max(self.width(), self.minimumWidth(), 1)
        height = max(self.height(), self.minimumHeight(), 1)
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#101418"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#9fb0bf"))
        painter.setFont(self.font())
        painter.drawText(0, 0, width, height, Qt.AlignCenter, self._message or "")
        painter.end()
        self.clear()
        self.setText("")
        self.setPixmap(pixmap)
        self.repaint()

    def _update_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class ElfVisionMain(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.stopping_workers = []
        self.odin_driver_process = None
        self.odin_bridge_process = None
        self.odin_rviz_process = None
        self.odin_bridge_buffer = ""
        self.odin_widgets = []
        self.task_param_inputs = {}
        self.init_ui()
        self.refresh_task_params()
        self.update_odin_visibility()

    def init_ui(self):
        self.setWindowTitle("ELF2 Vision Demo")
        self.resize(1180, 720)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        self.title_label = QLabel("ELF2 Vision Runtime")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #1f2a33;")
        left_panel.addWidget(self.title_label)

        self.video_label = VideoLabel()
        left_panel.addWidget(self.video_label, stretch=1)

        self.info_label = QLabel("Task: idle | FPS: -- | Resolution: --")
        self.info_label.setMinimumHeight(34)
        self.info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.info_label.setStyleSheet(
            "background: #edf2f5; color: #26323a; padding: 8px 10px; font-size: 14px;"
        )
        left_panel.addWidget(self.info_label)

        right_panel = QFrame()
        right_panel.setFixedWidth(285)
        right_panel.setStyleSheet(
            "QFrame { background: #f7f9fb; border: 1px solid #d8e0e6; }"
            "QLabel { color: #25313a; border: 0; background: transparent; }"
            "QComboBox, QLineEdit { padding: 7px; border: 1px solid #bfccd6; background: white; color: #17212b; selection-background-color: #d7e6f5; selection-color: #17212b; }"
            "QComboBox:hover, QLineEdit:hover { border-color: #8295a6; background: #f0f5f9; color: #17212b; }"
            "QComboBox QAbstractItemView { background: white; color: #17212b; selection-background-color: #d7e6f5; selection-color: #17212b; outline: 0; border: 1px solid #bfccd6; }"
            "QComboBox QAbstractItemView::item { min-height: 26px; padding: 4px 8px; background: white; color: #17212b; }"
            "QComboBox QAbstractItemView::item:hover { background: #e5eef7; color: #17212b; }"
            "QComboBox QAbstractItemView::item:selected { background: #d7e6f5; color: #17212b; }"
            "QComboBox QAbstractItemView::item:selected:hover { background: #c7daec; color: #17212b; }"
            "QPushButton { padding: 10px; font-weight: 600; border: 0; background: #1f6feb; color: white; }"
            "QPushButton:hover { background: #185abc; color: white; }"
            "QPushButton:disabled { background: #9aa8b5; color: #eef3f7; }"
        )
        controls = QVBoxLayout(right_panel)
        controls.setContentsMargins(14, 14, 14, 14)
        controls.setSpacing(10)

        self.top_controls_widget = QWidget()
        top_controls = QVBoxLayout(self.top_controls_widget)
        top_controls.setContentsMargins(0, 0, 0, 0)
        top_controls.setSpacing(10)

        top_controls.addWidget(self._section_label("Task"))
        self.task_combo = QComboBox()
        self._style_combo_popup(self.task_combo)
        for task_id, meta in TASKS.items():
            self.task_combo.addItem(meta["label"], task_id)
        self._fix_combo_item_colors(self.task_combo)
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        top_controls.addWidget(self.task_combo)

        top_controls.addWidget(self._section_label("Camera"))
        self.camera_input = QLineEdit("21")
        top_controls.addWidget(self.camera_input)

        top_controls.addWidget(self._section_label("Resolution"))
        self.resolution_combo = QComboBox()
        self._style_combo_popup(self.resolution_combo)
        for label in RESOLUTIONS:
            self.resolution_combo.addItem(label)
        self._fix_combo_item_colors(self.resolution_combo)
        self.resolution_combo.currentIndexChanged.connect(self.restart_if_running)
        top_controls.addWidget(self.resolution_combo)

        self.odin_section_label = self._section_label("Odin1 Position")
        self._add_odin_widget(top_controls, self.odin_section_label)
        self.odin_xyz_label = QLabel("X: -- | Y: -- | Z: --")
        self.odin_xyz_label.setStyleSheet(
            "background: #eef4f8; color: #17212b; padding: 8px; font-size: 13px;"
        )
        self._add_odin_widget(top_controls, self.odin_xyz_label)

        self.odin_status_label = QLabel("Odin1: stopped")
        self.odin_status_label.setWordWrap(True)
        self.odin_status_label.setStyleSheet("color: #667684;")
        self._add_odin_widget(top_controls, self.odin_status_label)

        self.odin_start_button = QPushButton("Start Odin1 Lite")
        self.odin_start_button.setStyleSheet(self._button_style())
        self.odin_start_button.clicked.connect(self.start_odin1)
        self._add_odin_widget(top_controls, self.odin_start_button)

        self.odin_stop_button = QPushButton("Stop Odin1")
        self.odin_stop_button.setStyleSheet(self._button_style())
        self.odin_stop_button.clicked.connect(self.stop_odin1)
        self.odin_stop_button.setEnabled(False)
        self._add_odin_widget(top_controls, self.odin_stop_button)

        top_controls.addWidget(self._section_label("Task Parameters"))
        self.param_container = QWidget()
        self.param_box = QVBoxLayout()
        self.param_box.setContentsMargins(0, 0, 0, 0)
        self.param_box.setSpacing(8)
        self.param_container.setLayout(self.param_box)
        top_controls.addWidget(self.param_container)
        top_controls.addStretch(1)

        self.top_controls_scroll = QScrollArea()
        self.top_controls_scroll.setWidgetResizable(True)
        self.top_controls_scroll.setFrameShape(QFrame.NoFrame)
        self.top_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.top_controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.top_controls_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.top_controls_scroll.setWidget(self.top_controls_widget)
        self.top_controls_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: 0; }"
            "QScrollArea QWidget { background: transparent; }"
        )
        controls.addWidget(self.top_controls_scroll, stretch=1)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #5d6b76; border: 0; background: transparent;")
        controls.addWidget(self.status_label)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_current_task)
        controls.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(lambda: self.stop_worker(show_idle=True))
        self.stop_button.setEnabled(False)
        controls.addWidget(self.stop_button)

        root_layout.addLayout(left_panel, stretch=3)
        root_layout.addWidget(right_panel, stretch=1)

    def _button_style(self):
        return (
            "QPushButton { padding: 10px; font-weight: 600; border: 0; background: #1f6feb; color: white; }"
            "QPushButton:hover { background: #185abc; color: white; }"
            "QPushButton:disabled { background: #9aa8b5; color: #eef3f7; }"
        )
    def _section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 13px; font-weight: 700; color: #33414c;")
        return label

    def _add_odin_widget(self, layout, widget):
        self.odin_widgets.append(widget)
        layout.addWidget(widget)

    def update_odin_visibility(self):
        visible = self.task_combo.currentData() == "passthrough"
        for widget in self.odin_widgets:
            widget.setVisible(visible)

    def _style_combo_popup(self, combo):
        combo.view().setTextElideMode(Qt.ElideNone)
        combo.view().setCurrentIndex(combo.model().index(-1, -1))
        combo.view().setStyleSheet(
            "QListView { background: white; color: #17212b; outline: 0; border: 1px solid #bfccd6; }"
            "QListView::item { min-height: 26px; padding: 4px 8px; background: white; color: #17212b; }"
            "QListView::item:hover { background: #e5eef7; color: #17212b; }"
            "QListView::item:selected { background: #d7e6f5; color: #17212b; }"
            "QListView::item:selected:hover { background: #c7daec; color: #17212b; }"
        )

    def _fix_combo_item_colors(self, combo):
        model = combo.model()
        for row in range(combo.count()):
            index = model.index(row, 0)
            model.setData(index, QColor("#17212b"), Qt.ForegroundRole)
            model.setData(index, QColor("#ffffff"), Qt.BackgroundRole)


    def start_odin1(self):
        if self.odin_driver_process is not None:
            self._ensure_odin_rviz()
            self.odin_status_label.setText("Odin1: already running, ensuring RViz...")
            return

        try:
            self.odin_xyz_label.setText("X: -- | Y: -- | Z: --")
            self.odin_bridge_buffer = ""
            self.odin_driver_process = self._start_shell_process(
                self._odin_driver_command(),
                self.on_odin_driver_output,
                self.on_odin_driver_finished,
            )
            self.odin_bridge_process = self._start_shell_process(
                self._odin_bridge_command(),
                self.on_odin_bridge_output,
                self.on_odin_bridge_finished,
            )
            driver_process = self.odin_driver_process
            QTimer.singleShot(
                3000,
                lambda: self._ensure_odin_rviz_for_driver(driver_process),
            )
        except Exception as exc:
            self.stop_odin1()
            QMessageBox.warning(self, "Odin1 Error", str(exc))
            return

        self.odin_start_button.setEnabled(True)
        self.odin_stop_button.setEnabled(True)
        self.odin_status_label.setText("Odin1: starting lite driver...")

    def stop_odin1(self):
        self._stop_process(self.odin_bridge_process)
        self._stop_process(self.odin_rviz_process)
        self._stop_process(self.odin_driver_process)
        self.odin_bridge_process = None
        self.odin_rviz_process = None
        self.odin_driver_process = None
        self.odin_start_button.setEnabled(True)
        self.odin_stop_button.setEnabled(False)
        self.odin_status_label.setText("Odin1: stopped")

    def _ensure_odin_rviz(self):
        if self.odin_rviz_process is not None:
            if self.odin_rviz_process.state() != QProcess.NotRunning:
                return
            self.odin_rviz_process = None
        self.odin_rviz_process = self._start_shell_process(
            self._odin_rviz_command(),
            self.on_odin_rviz_output,
            self.on_odin_rviz_finished,
        )

    def _ensure_odin_rviz_for_driver(self, driver_process):
        if driver_process is not self.odin_driver_process:
            return
        if driver_process.state() == QProcess.NotRunning:
            return
        self._ensure_odin_rviz()

    def _start_shell_process(self, command, output_slot, finished_slot):
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(output_slot)
        process.finished.connect(finished_slot)
        process.start("setsid", ["bash", "-lc", command])
        if not process.waitForStarted(3000):
            raise RuntimeError("Failed to start process: {}".format(command))
        return process

    def _stop_process(self, process):
        if process is None:
            return
        if process.state() == QProcess.NotRunning:
            return
        pid = int(process.processId())
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            process.terminate()
        if not process.waitForFinished(3000):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                process.kill()
            process.waitForFinished(1000)

    def _odin_driver_command(self):
        return (
            "source /opt/ros/humble/setup.bash; "
            "source ~/odin1/install/setup.bash; "
            "cd ~/odin1; "
            "exec ros2 launch odin_ros_driver odin1_ros2_lite.launch.py"
        )

    def _odin_rviz_command(self):
        return (
            "source /opt/ros/humble/setup.bash; "
            "source ~/odin1/install/setup.bash; "
            "if command -v wmctrl >/dev/null 2>&1 && wmctrl -a RViz >/dev/null 2>&1; then "
            "echo 'RViz window raised'; "
            "elif pgrep -x rviz2 >/dev/null 2>&1; then "
            "echo 'RViz already running'; "
            "else "
            "cfg=$(find ~/odin1 -name '*lite*.rviz' -print -quit 2>/dev/null); "
            "if [ -n \"$cfg\" ]; then "
            "exec ros2 run rviz2 rviz2 -d \"$cfg\"; "
            "else "
            "exec ros2 run rviz2 rviz2; "
            "fi; "
            "fi"
        )

    def _odin_bridge_command(self):
        bridge_path = Path(__file__).resolve().parent / "odin1_odom_bridge.py"
        return (
            "source /opt/ros/humble/setup.bash; "
            "source ~/odin1/install/setup.bash; "
            "exec python3 {}".format(shlex.quote(str(bridge_path)))
        )

    def on_odin_driver_output(self):
        process = self.sender()
        if process is not self.odin_driver_process:
            return
        text = bytes(process.readAllStandardOutput()).decode(errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.odin_status_label.setText("Odin1: {}".format(lines[-1][-120:]))

    def on_odin_rviz_output(self):
        process = self.sender()
        if process is not self.odin_rviz_process:
            return
        text = bytes(process.readAllStandardOutput()).decode(errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.odin_status_label.setText("RViz: {}".format(lines[-1][-120:]))

    def on_odin_bridge_output(self):
        process = self.sender()
        if process is not self.odin_bridge_process:
            return
        text = bytes(process.readAllStandardOutput()).decode(errors="replace")
        self.odin_bridge_buffer += text
        while "\n" in self.odin_bridge_buffer:
            line, self.odin_bridge_buffer = self.odin_bridge_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if "error" in payload:
                self.odin_status_label.setText("Odin1 bridge: {}".format(payload["error"]))
                continue
            self.odin_xyz_label.setText(
                "X: {x:.3f} | Y: {y:.3f} | Z: {z:.3f}".format(
                    x=payload.get("x", 0.0),
                    y=payload.get("y", 0.0),
                    z=payload.get("z", 0.0),
                )
            )
            self.odin_status_label.setText("Odin1: receiving /odin1/odometry")

    def on_odin_driver_finished(self, *args):
        if self.sender() is not self.odin_driver_process:
            return
        self.odin_driver_process = None
        if self.odin_bridge_process is None:
            self.odin_start_button.setEnabled(True)
            self.odin_stop_button.setEnabled(False)

    def on_odin_rviz_finished(self, *args):
        if self.sender() is not self.odin_rviz_process:
            return
        self.odin_rviz_process = None

    def on_odin_bridge_finished(self, *args):
        if self.sender() is not self.odin_bridge_process:
            return
        self.odin_bridge_process = None
        if self.odin_driver_process is None:
            self.odin_start_button.setEnabled(True)
            self.odin_stop_button.setEnabled(False)

    def on_task_changed(self):
        self.update_odin_visibility()
        self.refresh_task_params()
        self.restart_if_running()

    def refresh_task_params(self):
        while self.param_box.count():
            item = self.param_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.task_param_inputs = {}
        task_id = self.task_combo.currentData()
        params = TASKS[task_id].get("params", {})
        if not params:
            label = QLabel("No extra parameters")
            label.setStyleSheet("color: #778590;")
            self.param_box.addWidget(label)
            return

        for key, default in params.items():
            self.param_box.addWidget(QLabel(key))
            input_widget = QLineEdit(str(default))
            self.task_param_inputs[key] = input_widget
            self.param_box.addWidget(input_widget)

    def build_config(self):
        width, height = RESOLUTIONS[self.resolution_combo.currentText()]
        task_params = {
            key: widget.text().strip()
            for key, widget in self.task_param_inputs.items()
            if widget.text().strip()
        }
        return {
            "camera_index": self.camera_input.text().strip() or "21",
            "width": width,
            "height": height,
            "fps": 30,
            "task_params": task_params,
        }

    def start_current_task(self):
        self.stop_worker(show_idle=False)
        task_id = self.task_combo.currentData()
        task_cls = get_task_class(task_id)
        config = self.build_config()

        self.worker = VideoWorker(task_cls, config)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error.connect(self.on_worker_error)
        self.worker.status.connect(self.on_worker_status)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Starting...")
        if not self.video_label.is_showing_frame():
            self.video_label.set_message("Starting camera and task...")

    def stop_worker(self, show_idle=True):
        worker = self.worker
        self.worker = None
        if worker is not None:
            self._disconnect_worker(worker)
        if show_idle:
            self.video_label.set_blackout()
            QApplication.processEvents()
        self.info_label.setText("Task: stopped | FPS: -- | Resolution: --")
        self.status_label.setText("Stopped")
        if worker is not None and not worker.stop():
            self.stopping_workers.append(worker)
            self.status_label.setText("Stopping...")
        if show_idle:
            self.video_label.set_blackout()
            QApplication.processEvents()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _disconnect_worker(self, worker):
        for signal, slot in (
            (worker.frame_ready, self.on_frame_ready),
            (worker.error, self.on_worker_error),
            (worker.status, self.on_worker_status),
        ):
            try:
                signal.disconnect(slot)
            except TypeError:
                pass
    def restart_if_running(self):
        if self.worker is not None and self.worker.isRunning():
            self.start_current_task()

    def on_frame_ready(self, frame_bgr, metrics):
        if self.sender() is not self.worker:
            return
        self.video_label.set_frame(frame_bgr)
        self.info_label.setText(self._format_video_info(metrics))
        status_text = metrics.get("status_text") or "Running"
        infer_ms = metrics.get("infer_ms")
        if infer_ms is not None:
            status_text = "{} | Infer: {:.1f} ms".format(status_text, infer_ms)
        self.status_label.setText(status_text)

    def _format_video_info(self, metrics):
        parts = [
            "Task: {}".format(metrics.get("task_name", "--")),
            "FPS: {:.2f}".format(metrics.get("fps", 0.0)),
            "Resolution: {}x{}".format(
                metrics.get("width", "--"),
                metrics.get("height", "--"),
            ),
        ]

        infer_ms = metrics.get("infer_ms")
        if infer_ms is not None:
            parts.append("Infer: {:.1f} ms".format(infer_ms))

        post_ms = metrics.get("post_ms")
        if post_ms is not None:
            parts.append("Post: {:.1f} ms".format(post_ms))

        detections = metrics.get("detections")
        if detections is not None:
            parts.append("Detections: {}".format(detections))

        persons = metrics.get("persons")
        if persons is not None:
            parts.append("Persons: {}".format(persons))

        disp_min = metrics.get("disp_min")
        disp_max = metrics.get("disp_max")
        if disp_min is not None and disp_max is not None:
            parts.append("Depth: {:.3f}-{:.3f}".format(disp_min, disp_max))

        return " | ".join(parts)

    def on_worker_error(self, message):
        if self.sender() is not self.worker:
            return
        self.status_label.setText(message)
        self.video_label.set_message(message)
        QMessageBox.warning(self, "Task Error", message)

    def on_worker_status(self, message):
        if self.sender() is not self.worker:
            return
        self.status_label.setText(message)

    def on_worker_finished(self):
        sender = self.sender()
        if sender is not self.worker:
            if sender in self.stopping_workers:
                self.stopping_workers.remove(sender)
            return
        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_odin1()
        self.stop_worker()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ElfVisionMain()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
