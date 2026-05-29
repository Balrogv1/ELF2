import sys
import subprocess
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLabel, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt

class ElfLauncher(QWidget):
    def __init__(self):
        super().__init__()
        # 动态推导路径关系：
        # 1. 获取当前脚本所在目录 (例如: /home/elf/code/qt)
        self.qt_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. 获取父级目录作为代码根目录 (例如: /home/elf/code)
        self.code_root = os.path.dirname(self.qt_dir)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ELF2 算法演示启动器')
        self.setGeometry(300, 300, 500, 400)

        layout = QVBoxLayout()

        title = QLabel('ELF2 (RK3588) 算法启动器')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px; color: #333;")
        layout.addWidget(title)

        # 显示检测到的物理路径，方便调试
        path_label = QLabel(f"根目录: {self.code_root}")
        path_label.setStyleSheet("color: #777; font-size: 10px;")
        path_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(path_label)

        # YOLO模型配置
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("YOLO模型文件:"))
        self.model_input = QLineEdit()
        # 根据 tree 结构，yolo11n.rknn 位于 code 根目录下
        default_yolo = os.path.join(self.code_root, "yolo11n.rknn")
        self.model_input.setText(default_yolo)
        model_layout.addWidget(self.model_input)
        layout.addLayout(model_layout)

        notice = QLabel('注意：yolov8-python 脚本无法直接解析 yolo11 模型。\n如果启动后报错 ValueError，请更换为 v8 模型。')
        notice.setStyleSheet("color: #d35400; font-size: 10px;")
        notice.setAlignment(Qt.AlignCenter)
        layout.addWidget(notice)

        layout.addSpacing(10)

        # YOLO 启动按钮
        self.btn_yolo = QPushButton('启动 YOLO 实时检测')
        self.btn_yolo.setMinimumHeight(70)
        self.btn_yolo.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-size: 16px; 
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
        """)
        self.btn_yolo.clicked.connect(self.run_yolo)
        layout.addWidget(self.btn_yolo)

        # Lite-Mono 启动按钮
        self.btn_mono = QPushButton('启动 Lite-Mono 深度估计')
        self.btn_mono.setMinimumHeight(70)
        self.btn_mono.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; 
                color: white; 
                font-size: 16px; 
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:pressed {
                background-color: #1976D2;
            }
        """)
        self.btn_mono.clicked.connect(self.run_litemono)
        layout.addWidget(self.btn_mono)

        self.setLayout(layout)

    def run_yolo(self):
        # 对应 ~/code/yolov8-python
        working_dir = os.path.join(self.code_root, "yolov8-python")
        model_path = self.model_input.text()

        if not os.path.exists(model_path):
            QMessageBox.warning(self, "文件缺失", f"找不到模型文件：\n{model_path}")
            return

        cmd = [
            "python3", "usb_yolov8_camera.py", 
            "--model", model_path, 
            "--camera", "21"
        ]
        
        try:
            print(f"\n[启动 YOLO] 工作目录: {working_dir}")
            subprocess.Popen(cmd, cwd=working_dir)
        except Exception as e:
            QMessageBox.critical(self, "启动失败", str(e))

    def run_litemono(self):
        # 对应 ~/code/litemono-python
        working_dir = os.path.join(self.code_root, "litemono-python")
        
        cmd = [
            "python3", "usb_litemono_camera.py", 
            "--camera", "21"
        ]
        
        try:
            print(f"\n[启动 Lite-Mono] 工作目录: {working_dir}")
            subprocess.Popen(cmd, cwd=working_dir)
        except Exception as e:
            QMessageBox.critical(self, "启动失败", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = ElfLauncher()
    ex.show()
    sys.exit(app.exec_())
