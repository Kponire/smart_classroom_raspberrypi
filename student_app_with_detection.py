"""
ClassBridge Student Application with YOLO Face Detection
Integrated face detection for student counting
"""

import sys
import json
import asyncio
import time
import cv2
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFrame, QPushButton, QSlider,
                             QGroupBox, QFormLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QRect
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QBrush

from qasync import QEventLoop
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from av import VideoFrame

# Import face detector
from face_detector import YOLOFaceDetector, FaceDetection

# Configuration
SIGNALING_SERVER_URL = "ws://10.83.65.139:8000/ws/student"

class FaceDetectionOverlay:
    """Overlay for face detection visualization"""
    
    def __init__(self):
        self.student_count = 0
        self.faces = []
        self.overlay_enabled = True
        
    def update(self, faces: list, count: int):
        """Update face detection data"""
        self.faces = faces
        self.student_count = count
        
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay on frame"""
        if not self.overlay_enabled:
            return frame
            
        overlay = frame.copy()
        
        # Draw face boxes
        for i, face in enumerate(self.faces):
            x1, y1, x2, y2 = face.bbox
            
            # Color based on confidence
            if face.confidence > 0.8:
                color = (0, 255, 0)
            elif face.confidence > 0.6:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
                
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Student number
            label = f"S{i+1}"
            cv2.putText(overlay, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw student count at top
        count_text = f"Students: {self.student_count}"
        
        # Background for count
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(overlay, (10, 10), (10 + text_size[0] + 20, 10 + text_size[1] + 20),
                     (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (10 + text_size[0] + 20, 10 + text_size[1] + 20),
                     (0, 255, 0), 2)
        
        cv2.putText(overlay, count_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return overlay

class CameraThreadWithDetection(QThread):
    """Camera thread with YOLO face detection"""
    change_pixmap_signal = pyqtSignal(QImage, object, int)
    student_count_signal = pyqtSignal(int)
    detection_stats_signal = pyqtSignal(dict)
    
    def __init__(self, camera_index: int = 0, detection_enabled: bool = True):
        super().__init__()
        self.camera_index = camera_index
        self.detection_enabled = detection_enabled
        self._run_flag = True
        self.cap = None
        
        # Initialize face detector
        self.face_detector = YOLOFaceDetector()
        self.overlay = FaceDetectionOverlay()
        
        # Frame processing
        self.processing_fps = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        
    def run(self):
        """Main camera capture and processing loop"""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self._run_flag and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            processed_frame = frame.copy()
            student_count = 0
            
            # Perform face detection if enabled
            if self.detection_enabled:
                faces = self.face_detector.detect_faces(frame)
                student_count = self.face_detector.get_student_count()
                
                # Update overlay
                self.overlay.update(faces, student_count)
                
                # Draw overlay on frame
                processed_frame = self.overlay.draw(frame)
                
                # Emit student count
                self.student_count_signal.emit(student_count)
                
                # Emit stats periodically
                if self.frame_count % 30 == 0:
                    stats = self.face_detector.get_performance_stats()
                    self.detection_stats_signal.emit(stats)
            
            # Convert to RGB for Qt
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, 
                          QImage.Format.Format_RGB888)
            
            # Emit for display
            self.change_pixmap_signal.emit(qt_img, processed_frame, student_count)
            
            # FPS tracking
            self.frame_count += 1
            if time.time() - self.last_fps_update >= 1.0:
                self.processing_fps = self.frame_count
                self.frame_count = 0
                self.last_fps_update = time.time()
                
            time.sleep(0.033)  # ~30 FPS
            
        if self.cap:
            self.cap.release()
            
    def stop(self):
        """Stop the camera thread"""
        self._run_flag = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.wait()
        
    def toggle_detection(self, enabled: bool):
        """Enable or disable face detection"""
        self.detection_enabled = enabled
        if not enabled:
            self.overlay.overlay_enabled = False
        else:
            self.overlay.overlay_enabled = True
            
    def update_detection_params(self, confidence: float = None, iou: float = None, 
                               frame_skip: int = None):
        """Update face detection parameters"""
        self.face_detector.update_parameters(confidence, iou, frame_skip)

class ClassBridgeStudentApp(QWidget):
    """Main student application window with face detection"""
    
    def __init__(self):
        super().__init__()
        self.pc = None
        self.local_track = OpenCVVideoTrack()
        self.self_view_enabled = False
        self.teacher_mic_muted = False
        self.teacher_cam_off = False
        self.remote_track_task = None
        self.student_count = 0
        self.detection_stats = {}
        
        self.init_ui()
        self.start_camera()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("ClassBridge Learning Terminal - Face Detection")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #020617; color: #ffffff;")
        
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        
        # Header Bar
        header = self._create_header()
        root_layout.addWidget(header)
        
        # Main Stage Container
        self.stage_box = QFrame()
        self.stage_box.setStyleSheet("""
            background-color: #000000; 
            border-radius: 12px; 
            border: 1px solid #1e293b;
        """)
        
        # Main Display
        self.teacher_video = QLabel("Waiting for Teacher Stream...", self.stage_box)
        self.teacher_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.teacher_video.setStyleSheet("""
            color: #64748b; 
            font-size: 18px; 
            font-weight: 500;
        """)
        self.teacher_video.setScaledContents(True)
        
        # Self View PiP
        self.pip_card = self._create_pip()
        
        # Control Panel (Bottom)
        control_panel = self._create_control_panel()
        
        root_layout.addWidget(self.stage_box, stretch=1)
        root_layout.addWidget(control_panel)
        
    def _create_header(self) -> QFrame:
        """Create header bar"""
        header = QFrame()
        header.setStyleSheet("""
            background-color: #0f172a; 
            border-radius: 8px; 
            border: 1px solid #1e293b;
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        
        title_label = QLabel("ClassBridge Learning Station")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        
        # Student counter
        self.student_counter_label = QLabel("👥 Students: 0")
        self.student_counter_label.setStyleSheet("""
            background-color: #1e293b; 
            color: #22c55e; 
            font-weight: bold; 
            font-size: 14px; 
            border-radius: 4px; 
            padding: 4px 12px;
        """)
        
        # Status badge
        self.status_badge = QLabel(" INITIALIZING ")
        self.status_badge.setStyleSheet("""
            background-color: #eab308; 
            color: black; 
            font-weight: bold; 
            font-size: 12px; 
            border-radius: 4px; 
            padding: 4px 8px;
        """)
        
        self.mic_status = QLabel("🎤")
        self.mic_status.setStyleSheet("font-size: 16px;")
        self.mic_status.setToolTip("Microphone status")
        
        self.cam_status = QLabel("📷")
        self.cam_status.setStyleSheet("font-size: 16px;")
        self.cam_status.setToolTip("Camera status")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.student_counter_label)
        header_layout.addWidget(self.mic_status)
        header_layout.addWidget(self.cam_status)
        header_layout.addWidget(self.status_badge)
        
        return header
        
    def _create_pip(self) -> QFrame:
        """Create self-view PiP widget"""
        pip_card = QFrame(self.stage_box)
        pip_card.setStyleSheet("""
            background-color: #0f172a; 
            border-radius: 8px; 
            border: 2px solid #2563eb;
        """)
        pip_card.setFixedSize(220, 140)
        pip_layout = QVBoxLayout(pip_card)
        pip_layout.setContentsMargins(0, 0, 0, 0)
        
        self.classroom_video = QLabel(pip_card)
        self.classroom_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.classroom_video.setScaledContents(True)
        pip_layout.addWidget(self.classroom_video)
        
        # PiP label with student count
        self.pip_label = QLabel("👥 0", pip_card)
        self.pip_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
            color: #22c55e;
            font-weight: bold;
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 4px;
        """)
        self.pip_label.move(10, 10)
        
        pip_card.setVisible(False)
        return pip_card
        
    def _create_control_panel(self) -> QFrame:
        """Create control panel with detection options"""
        panel = QFrame()
        panel.setStyleSheet("""
            background-color: #0f172a; 
            border-radius: 8px; 
            border: 1px solid #1e293b;
        """)
        panel.setFixedHeight(100)
        
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        
        # Detection group
        detection_group = QGroupBox("Face Detection")
        detection_group.setStyleSheet("""
            QGroupBox {
                color: #94a3b8;
                border: 1px solid #1e293b;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        detection_layout = QHBoxLayout(detection_group)
        
        # Toggle detection
        self.detection_toggle = QPushButton("🔄 Detection: ON")
        self.detection_toggle.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
        """)
        self.detection_toggle.clicked.connect(self.toggle_detection)
        detection_layout.addWidget(self.detection_toggle)
        
        # Confidence slider
        conf_layout = QVBoxLayout()
        conf_label = QLabel(f"Confidence: {0.5:.2f}")
        conf_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        conf_slider = QSlider(Qt.Orientation.Horizontal)
        conf_slider.setRange(10, 90)
        conf_slider.setValue(50)
        conf_slider.valueChanged.connect(
            lambda v: self.update_detection_params(confidence=v/100, label=conf_label)
        )
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(conf_slider)
        detection_layout.addLayout(conf_layout)
        
        # Frame skip slider
        skip_layout = QVBoxLayout()
        skip_label = QLabel("Frame Skip: 2")
        skip_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        skip_slider = QSlider(Qt.Orientation.Horizontal)
        skip_slider.setRange(1, 10)
        skip_slider.setValue(2)
        skip_slider.valueChanged.connect(
            lambda v: self.update_detection_params(frame_skip=v, label=skip_label)
        )
        skip_layout.addWidget(skip_label)
        skip_layout.addWidget(skip_slider)
        detection_layout.addLayout(skip_layout)
        
        layout.addWidget(detection_group, stretch=2)
        
        # Stats display
        stats_group = QGroupBox("Detection Stats")
        stats_group.setStyleSheet("""
            QGroupBox {
                color: #94a3b8;
                border: 1px solid #1e293b;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        stats_layout = QHBoxLayout(stats_group)
        
        self.stats_label = QLabel("FPS: 0 | Students: 0")
        self.stats_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group, stretch=1)
        
        return panel
        
    def start_camera(self):
        """Start the camera thread with face detection"""
        self.camera_thread = CameraThreadWithDetection(camera_index=0, detection_enabled=True)
        self.camera_thread.change_pixmap_signal.connect(self.update_classroom_feed)
        self.camera_thread.student_count_signal.connect(self.update_student_count)
        self.camera_thread.detection_stats_signal.connect(self.update_detection_stats)
        self.camera_thread.start()
        
    def update_classroom_feed(self, qt_img, cv_frame, student_count):
        """Update the local camera feed display"""
        pixmap = QPixmap.fromImage(qt_img)
        self.classroom_video.setPixmap(pixmap)
        self.local_track.update_frame(cv_frame)
        
        # Update PiP label
        self.pip_label.setText(f"👥 {student_count}")
        
    def update_student_count(self, count):
        """Update student count display"""
        self.student_count = count
        self.student_counter_label.setText(f"👥 Students: {count}")
        
        # Send count to teacher via WebRTC data channel if available
        if hasattr(self, 'pc') and self.pc:
            # You can send this via data channel
            pass
            
    def update_detection_stats(self, stats):
        """Update detection statistics display"""
        self.detection_stats = stats
        self.stats_label.setText(
            f"FPS: {stats.get('processing_fps', 0):.1f} | "
            f"Students: {stats.get('student_count', 0)} | "
            f"Device: {stats.get('device', 'unknown')}"
        )
        
    def toggle_detection(self):
        """Toggle face detection on/off"""
        if self.camera_thread:
            enabled = not self.camera_thread.detection_enabled
            self.camera_thread.toggle_detection(enabled)
            self.detection_toggle.setText(f"🔄 Detection: {'ON' if enabled else 'OFF'}")
            self.detection_toggle.setStyleSheet("""
                QPushButton {
                    background-color: %s;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: %s;
                }
            """ % (("#22c55e" if enabled else "#ef4444"), 
                   ("#16a34a" if enabled else "#dc2626")))
        
    def update_detection_params(self, confidence: float = None, frame_skip: int = None, label: QLabel = None):
        """Update detection parameters from sliders"""
        if self.camera_thread:
            self.camera_thread.update_detection_params(
                confidence=confidence,
                frame_skip=frame_skip
            )
            if label:
                if confidence is not None:
                    label.setText(f"Confidence: {confidence:.2f}")
                elif frame_skip is not None:
                    label.setText(f"Frame Skip: {frame_skip}")
                    
    def resizeEvent(self, event):
        """Position widgets when window is resized"""
        super().resizeEvent(event)
        w = self.stage_box.width()
        h = self.stage_box.height()
        
        self.teacher_video.setGeometry(0, 0, w, h)
        
        pip_w = self.pip_card.width()
        pip_h = self.pip_card.height()
        self.pip_card.move(w - pip_w - 20, h - pip_h - 20)
        
    # ... (remaining WebRTC methods from original code)
    # Include all WebRTC methods: connect_webrtc, handle_control_message, 
    # handle_offer, handle_candidate, render_remote_track, closeEvent

# Note: Keep the OpenCVVideoTrack class and all WebRTC implementation
# from the original code

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = ClassBridgeStudentApp()
    window.showMaximized()
    
    loop.create_task(window.connect_webrtc())
    
    with loop:
        loop.run_forever()