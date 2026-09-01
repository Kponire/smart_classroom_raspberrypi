import sys
import json
import asyncio
import cv2
import numpy as np

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from qasync import QEventLoop
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from av import VideoFrame

# Configuration
SIGNALING_SERVER_URL = "ws://10.83.65.139:8000/ws/student"

class OpenCVVideoTrack(VideoStreamTrack):
    """Custom video track that sends frames from OpenCV"""
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self._frame_lock = asyncio.Lock()

    def update_frame(self, frame):
        """Update the current frame to be sent"""
        self.current_frame = frame

    async def recv(self):
        """Receive the next video frame"""
        pts, time_base = await self.next_timestamp()
        
        async with self._frame_lock:
            if self.current_frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                av_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            else:
                # Blank frame if no camera feed
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
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        while self._run_flag and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB for Qt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, 
                              QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img, frame)
            QThread.msleep(33)  # ~30 FPS

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
        self.pc = None
        self.local_track = OpenCVVideoTrack()
        self.self_view_enabled = False
        self.teacher_mic_muted = False
        self.teacher_cam_off = False
        self.remote_track_task = None
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

        title_label = QLabel("ClassBridge Learning Station")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")

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
        self.mic_status.setToolTip("Microphone status")
        
        self.cam_status = QLabel("📷")
        self.cam_status.setStyleSheet("font-size: 16px;")
        self.cam_status.setToolTip("Camera status")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
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
        
        # Main Display: Large view showing Teacher Feed or Screen Share
        self.teacher_video = QLabel("Waiting for Teacher Stream...", self.stage_box)
        self.teacher_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.teacher_video.setStyleSheet("""
            color: #64748b; 
            font-size: 18px; 
            font-weight: 500;
        """)
        self.teacher_video.setScaledContents(True)

        # Self View PiP Frame (Bottom Right Corner)
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

        # Show local camera self-view by default if enabled
        self.pip_card.setVisible(False)

        root_layout.addWidget(header)
        root_layout.addWidget(self.stage_box, stretch=1)

    def resizeEvent(self, event):
        """Position widgets when window is resized"""
        super().resizeEvent(event)
        w = self.stage_box.width()
        h = self.stage_box.height()
        
        # Expand teacher stream across the full stage box
        self.teacher_video.setGeometry(0, 0, w, h)
        
        # Position self-view in bottom-right corner with padding
        pip_w = self.pip_card.width()
        pip_h = self.pip_card.height()
        self.pip_card.move(w - pip_w - 20, h - pip_h - 20)

    def start_camera(self):
        """Start the local camera thread"""
        self.camera_thread = LocalCameraThread(camera_index=0)
        self.camera_thread.change_pixmap_signal.connect(self.update_classroom_feed)
        self.camera_thread.start()

    def update_classroom_feed(self, qt_img, cv_frame):
        """Update the local camera feed display"""
        pixmap = QPixmap.fromImage(qt_img)
        self.classroom_video.setPixmap(pixmap)
        self.local_track.update_frame(cv_frame)

    async def connect_webrtc(self):
        """Connect to the signaling server and establish WebRTC connection"""
        self.pc = RTCPeerConnection()
        
        # Add local video track
        self.pc.addTrack(self.local_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                if self.remote_track_task is None or self.remote_track_task.done():
                    self.remote_track_task = asyncio.create_task(
                        self.render_remote_track(track)
                    )

        try:
            async with websockets.connect(SIGNALING_SERVER_URL) as ws:
                self.status_badge.setText(" READY ")
                self.status_badge.setStyleSheet("""
                    background-color: #22c55e; 
                    color: white; 
                    font-weight: bold; 
                    font-size: 12px; 
                    border-radius: 4px; 
                    padding: 4px 8px;
                """)

                async for msg in ws:
                    data = json.loads(msg)

                    # Handle control messages from teacher
                    if data.get("type") == "control":
                        await self.handle_control_message(data)

                    # Handle WebRTC signaling
                    elif "offer" in data:
                        await self.handle_offer(data, ws)
                    
                    elif "candidate" in data and data["candidate"]:
                        await self.handle_candidate(data)

        except websockets.ConnectionClosed:
            print("WebSocket connection closed")
            self.status_badge.setText(" DISCONNECTED ")
            self.status_badge.setStyleSheet("""
                background-color: #64748b; 
                color: white; 
                font-weight: bold; 
                font-size: 12px; 
                border-radius: 4px; 
                padding: 4px 8px;
            """)
        except Exception as e:
            print(f"Signaling error: {e}")
            self.status_badge.setText(" ERROR ")
            self.status_badge.setStyleSheet("""
                background-color: #dc2626; 
                color: white; 
                font-weight: bold; 
                font-size: 12px; 
                border-radius: 4px; 
                padding: 4px 8px;
            """)
            # Retry connection after delay
            await asyncio.sleep(5)
            asyncio.create_task(self.connect_webrtc())

    async def handle_control_message(self, data):
        """Handle control messages from teacher"""
        action = data.get("action")
        
        if action == "toggle-self-view":
            self.self_view_enabled = data.get("enabled", False)
            self.pip_card.setVisible(self.self_view_enabled)
            print(f"Self-view toggled: {self.self_view_enabled}")
            
        elif action == "toggle-mic":
            self.teacher_mic_muted = data.get("muted", False)
            self.mic_status.setText("🔇" if self.teacher_mic_muted else "🎤")
            self.mic_status.setToolTip(
                "Microphone muted by teacher" if self.teacher_mic_muted else "Microphone active"
            )
            print(f"Mic toggled: {'muted' if self.teacher_mic_muted else 'unmuted'}")
            
        elif action == "toggle-camera":
            self.teacher_cam_off = data.get("off", False)
            self.cam_status.setText("📷❌" if self.teacher_cam_off else "📷")
            self.cam_status.setToolTip(
                "Camera off by teacher" if self.teacher_cam_off else "Camera active"
            )
            print(f"Camera toggled: {'off' if self.teacher_cam_off else 'on'}")

    async def handle_offer(self, data, ws):
        """Handle WebRTC offer from teacher"""
        try:
            offer = RTCSessionDescription(
                sdp=data["offer"]["sdp"], 
                type=data["offer"]["type"]
            )
            await self.pc.setRemoteDescription(offer)
            
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            await ws.send(json.dumps({
                "answer": {
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type
                }
            }))
            
            self.status_badge.setText(" LIVE ")
            self.status_badge.setStyleSheet("""
                background-color: #dc2626; 
                color: white; 
                font-weight: bold; 
                font-size: 12px; 
                border-radius: 4px; 
                padding: 4px 8px;
            """)
            print("WebRTC connection established")
            
        except Exception as e:
            print(f"Error handling offer: {e}")

    async def handle_candidate(self, data):
        """Handle ICE candidate from teacher - FIXED for aiortc API"""
        try:
            # Extract candidate info - aiortc expects these specific fields
            candidate_info = data["candidate"]
            
            # Create RTCIceCandidate with correct parameter names
            candidate = RTCIceCandidate(
                sdpMid=candidate_info.get("sdpMid"),
                sdpMLineIndex=candidate_info.get("sdpMLineIndex"),
                candidate=candidate_info.get("candidate")
            )
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            print(f"Error adding ICE candidate: {e}")

    async def render_remote_track(self, track):
        """Render remote video track from teacher"""
        try:
            while True:
                frame = await track.recv()
                # Convert frame to QImage
                img = frame.to_ndarray(format="bgr24")
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_img.data, w, h, bytes_per_line, 
                              QImage.Format.Format_RGB888)
                
                # Update the display
                self.teacher_video.setPixmap(QPixmap.fromImage(qt_img))
                
        except Exception as e:
            print(f"Remote track error: {e}")
            # Show error message on the display
            self.teacher_video.setText("Video stream lost\nReconnecting...")

    def closeEvent(self, event):
        """Clean up when closing the application"""
        print("Closing application...")
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

    # Start WebRTC connection
    loop.create_task(window.connect_webrtc())

    with loop:
        loop.run_forever()