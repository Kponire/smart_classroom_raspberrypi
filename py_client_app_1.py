import sys
import json
import asyncio
import time
import cv2
import numpy as np

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFrame, QLineEdit, QPushButton)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSettings
from PyQt6.QtGui import QImage, QPixmap

from qasync import QEventLoop
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from av import VideoFrame
from aiortc.contrib.signaling import object_from_dict

class OpenCVVideoTrack(VideoStreamTrack):
    """Custom video track that sends frames from OpenCV"""
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self._frame_lock = asyncio.Lock()

    def update_frame(self, frame):
        self.current_frame = frame

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        async with self._frame_lock:
            if self.current_frame is not None:
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                av_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            else:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                av_frame = VideoFrame.from_ndarray(blank_frame, format="rgb24")
            
            av_frame.pts = pts
            av_frame.time_base = time_base
            return av_frame

class LocalCameraThread(QThread):
    """Thread for capturing camera frames"""
    change_pixmap_signal = pyqtSignal(QImage, object)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._run_flag = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        while self._run_flag and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img, frame)
            time.sleep(0.033)

        if self.cap:
            self.cap.release()

    def stop(self):
        self._run_flag = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.wait()

class ClassBridgeStudentApp(QWidget):
    """Main student application window"""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("ClassBridge", "StudentApp")
        self.server_ip = self.settings.value("server_ip", "10.148.101.130")

        self.pc = None
        self.ws = None
        self.local_track = OpenCVVideoTrack()
        self.self_view_enabled = False
        self.teacher_mic_muted = False
        self.teacher_cam_off = False
        self.remote_track_task = None
        self.webrtc_task = None

        self.init_ui()
        self.start_camera()

    def init_ui(self):
        self.setWindowTitle("ClassBridge Remote Learning Terminal")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #020617; color: #ffffff;")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # Header Bar
        header = QFrame()
        header.setStyleSheet("""
            background-color: #0f172a; 
            border-radius: 8px; 
            border: 1px solid #1e293b;
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title_label = QLabel("ClassBridge Station")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")

        # IP Input Configuration Widgets
        ip_label = QLabel("Server IP:")
        ip_label.setStyleSheet("font-size: 13px; color: #94a3b8;")
        
        self.ip_input = QLineEdit(self.server_ip)
        self.ip_input.setFixedWidth(130)
        self.ip_input.setStyleSheet("""
            background-color: #1e293b; 
            color: white; 
            border: 1px solid #334155; 
            border-radius: 4px; 
            padding: 4px 8px;
            font-size: 13px;
        """)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; 
                color: white; 
                font-weight: bold; 
                border-radius: 4px; 
                padding: 5px 12px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        self.connect_btn.clicked.connect(self.on_ip_update)

        # Status indicators
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
        self.cam_status = QLabel("📷")
        self.cam_status.setStyleSheet("font-size: 16px;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(ip_label)
        header_layout.addWidget(self.ip_input)
        header_layout.addWidget(self.connect_btn)
        header_layout.addSpacing(15)
        header_layout.addWidget(self.mic_status)
        header_layout.addWidget(self.cam_status)
        header_layout.addWidget(self.status_badge)

        # Main Stage Container
        self.stage_box = QFrame()
        self.stage_box.setStyleSheet("""
            background-color: #000000; 
            border-radius: 12px; 
            border: 1px solid #1e293b;
        """)
        
        self.teacher_video = QLabel("Waiting for Teacher Stream...", self.stage_box)
        self.teacher_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.teacher_video.setStyleSheet("color: #64748b; font-size: 18px; font-weight: 500;")
        self.teacher_video.setScaledContents(True)

        self.pip_card = QFrame(self.stage_box)
        self.pip_card.setStyleSheet("""
            background-color: #0f172a; 
            border-radius: 8px; 
            border: 2px solid #2563eb;
        """)
        self.pip_card.setFixedSize(220, 140)
        pip_layout = QVBoxLayout(self.pip_card)
        pip_layout.setContentsMargins(0, 0, 0, 0)

        self.classroom_video = QLabel(self.pip_card)
        self.classroom_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.classroom_video.setScaledContents(True)
        pip_layout.addWidget(self.classroom_video)
        self.pip_card.setVisible(False)

        root_layout.addWidget(header)
        root_layout.addWidget(self.stage_box, stretch=1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.stage_box.width()
        h = self.stage_box.height()
        self.teacher_video.setGeometry(0, 0, w, h)
        pip_w, pip_h = self.pip_card.width(), self.pip_card.height()
        self.pip_card.move(w - pip_w - 20, h - pip_h - 20)

    def start_camera(self):
        self.camera_thread = LocalCameraThread(camera_index=0)
        self.camera_thread.change_pixmap_signal.connect(self.update_classroom_feed)
        self.camera_thread.start()

    def update_classroom_feed(self, qt_img, cv_frame):
        self.classroom_video.setPixmap(QPixmap.fromImage(qt_img))
        self.local_track.update_frame(cv_frame)

    def on_ip_update(self):
        new_ip = self.ip_input.text().strip()
        if new_ip:
            self.server_ip = new_ip
            self.settings.setValue("server_ip", new_ip)
            print(f"Server IP updated to: {new_ip}")
            self.reconnect_webrtc()

    def reconnect_webrtc(self):
        if self.webrtc_task and not self.webrtc_task.done():
            self.webrtc_task.cancel()
        self.webrtc_task = asyncio.create_task(self.connect_webrtc())

    async def connect_webrtc(self):
        await self.cleanup_peer_connection()

        self.pc = RTCPeerConnection()
        self.pc.addTrack(self.local_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                if self.remote_track_task is None or self.remote_track_task.done():
                    self.remote_track_task = asyncio.create_task(self.render_remote_track(track))

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            state = self.pc.iceConnectionState
            print(f"Student ICE Connection State: {state}")
            if state in ["connected", "completed"]:
                self.status_badge.setText(" LIVE ")
                self.status_badge.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
            elif state in ["disconnected", "failed"]:
                self.status_badge.setText(" RECONNECTING ")
                self.status_badge.setStyleSheet("background-color: #eab308; color: black; font-weight: bold;")
                # Clear old video display notice
                self.teacher_video.setText("Connection lost\nWaiting for stream recovery...")

        signaling_url = f"ws://{self.server_ip}:8000/ws/student"
        try:
            async with websockets.connect(signaling_url) as ws:
                self.ws = ws
                self.status_badge.setText(" READY ")
                self.status_badge.setStyleSheet("background-color: #22c55e; color: white; font-weight: bold;")

                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("type") == "control":
                        await self.handle_control_message(data)
                    elif "offer" in data:
                        await self.handle_offer(data, ws)
                    elif "candidate" in data and data["candidate"]:
                        await self.handle_candidate(data)

        except (asyncio.CancelledError, websockets.ConnectionClosed):
            self.status_badge.setText(" DISCONNECTED ")
            self.status_badge.setStyleSheet("background-color: #64748b; color: white; font-weight: bold;")
        except Exception as e:
            print(f"Signaling error: {e}")
            self.status_badge.setText(" OFFLINE ")
            self.status_badge.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
            await asyncio.sleep(3)
            self.webrtc_task = asyncio.create_task(self.connect_webrtc())
    
    async def cleanup_peer_connection(self):
        """Safely cleans up existing RTCPeerConnection and running rendering tasks."""
        if self.remote_track_task and not self.remote_track_task.done():
            self.remote_track_task.cancel()
            self.remote_track_task = None

        if self.pc:
            await self.pc.close()
            self.pc = None

    async def handle_control_message(self, data):
        action = data.get("action")
        if action == "toggle-self-view":
            self.self_view_enabled = data.get("enabled", False)
            self.pip_card.setVisible(self.self_view_enabled)
        elif action == "toggle-mic":
            self.teacher_mic_muted = data.get("muted", False)
            self.mic_status.setText("🔇" if self.teacher_mic_muted else "🎤")
        elif action == "toggle-camera":
            self.teacher_cam_off = data.get("off", False)
            self.cam_status.setText("📷❌" if self.teacher_cam_off else "📷")

    async def handle_offer(self, data, ws):
        try:
            # Re-initialize peer connection if current one failed or closed
            if self.pc is None or self.pc.iceConnectionState in ["failed", "closed"]:
                await self.cleanup_peer_connection()
                self.pc = RTCPeerConnection()
                self.pc.addTrack(self.local_track)

            offer = RTCSessionDescription(sdp=data["offer"]["sdp"], type=data["offer"]["type"])
            await self.pc.setRemoteDescription(offer)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            await ws.send(json.dumps({
                "answer": {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}
            }))
            self.status_badge.setText(" LIVE ")
            self.status_badge.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
        except Exception as e:
            print(f"Error handling offer: {e}")
    
    async def handle_candidate(self, data):
        try:
            cand_data = data.get("candidate")
            if cand_data:
                ice_obj = {
                    "candidate": cand_data.get("candidate"),
                    "sdpMid": cand_data.get("sdpMid"),
                    "sdpMLineIndex": cand_data.get("sdpMLineIndex")
                }
                candidate = object_from_dict({"type": "candidate", "candidate": ice_obj})
                await self.pc.addIceCandidate(candidate)
        except Exception as e:
            print(f"Error adding ICE candidate: {e}")

    async def render_remote_track(self, track):
        try:
            while True:
                # Add a timeout so track.recv() doesn't hang indefinitely on a dead network stream
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                img = frame.to_ndarray(format="bgr24")
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.teacher_video.setPixmap(QPixmap.fromImage(qt_img))
        except asyncio.TimeoutError:
            print("Remote video frame timeout. Network may have stalled.")
            self.teacher_video.setText("Video stream stalled\nReconnecting...")
        except Exception as e:
            print(f"Remote track error: {e}")
            self.teacher_video.setText("Video stream lost\nReconnecting...")

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        if self.pc:
            asyncio.create_task(self.pc.close())
        if self.remote_track_task and not self.remote_track_task.done():
            self.remote_track_task.cancel()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = ClassBridgeStudentApp()
    window.showMaximized()

    window.webrtc_task = loop.create_task(window.connect_webrtc())

    with loop:
        loop.run_forever()