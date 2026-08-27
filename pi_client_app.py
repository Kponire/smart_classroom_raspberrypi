import sys
import json
import asyncio
import cv2
import pyaudio
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
import websockets

# Configuration
SIGNALING_SERVER_URL = "ws://10.83.65.139:8000/ws/student"

class LocalCameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage, object)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # Convert frame from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img, rgb_frame)
            self.msleep(30)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class ClassBridgeStudentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.start_camera()

    def init_ui(self):
        self.setWindowTitle("ClassBridge Remote Learning Terminal")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #020617; color: #ffffff;")

        # Root Layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # Header Bar
        header = QFrame()
        header.setStyleSheet("background-color: #0f172a; border-radius: 8px; border: 1px solid #1e293b;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title_label = QLabel("ClassBridge Remote Learning Terminal")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")

        status_badge = QLabel(" CONNECTED ")
        status_badge.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; font-size: 12px; border-radius: 4px; padding: 4px 8px;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(status_badge)

        # Main Viewport (Video Container)
        viewport_layout = QHBoxLayout()
        viewport_layout.setSpacing(12)

        # Teacher View Container (3x width)
        self.teacher_card = QFrame()
        self.teacher_card.setStyleSheet("background-color: #090d16; border-radius: 12px; border: 1px solid #1e293b;")
        teacher_layout = QVBoxLayout(self.teacher_card)
        teacher_layout.setContentsMargins(0, 0, 0, 0)

        self.teacher_video = QLabel("Waiting for Teacher Stream...")
        self.teacher_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.teacher_video.setStyleSheet("color: #64748b; font-size: 16px; font-weight: 500;")
        teacher_layout.addWidget(self.teacher_video)

        # Classroom Local Camera Container (1x width)
        self.classroom_card = QFrame()
        self.classroom_card.setStyleSheet("background-color: #090d16; border-radius: 12px; border: 1px solid #1e293b;")
        classroom_layout = QVBoxLayout(self.classroom_card)
        classroom_layout.setContentsMargins(0, 0, 0, 0)

        self.classroom_video = QLabel("Initializing USB Camera...")
        self.classroom_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.classroom_video.setScaledContents(True)
        classroom_layout.addWidget(self.classroom_video)

        viewport_layout.addWidget(self.teacher_card, stretch=3)
        viewport_layout.addWidget(self.classroom_card, stretch=1)

        root_layout.addWidget(header)
        root_layout.addLayout(viewport_layout)

    def start_camera(self):
        # Start OpenCV capture thread
        self.camera_thread = LocalCameraThread(camera_index=0)
        self.camera_thread.change_pixmap_signal.connect(self.update_classroom_feed)
        self.camera_thread.start()

    def update_classroom_feed(self, qt_img, cv_frame):
        # Update UI video display
        pixmap = QPixmap.fromImage(qt_img)
        self.classroom_video.setPixmap(pixmap)
        
        # Here you can pass 'cv_frame' directly into OpenCV object detection / student counting routines!

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClassBridgeStudentApp()
    window.showMaximized()
    sys.exit(app.exec())