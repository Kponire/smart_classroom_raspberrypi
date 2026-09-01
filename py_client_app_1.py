import sys
import json
import asyncio
import cv2
import numpy as np

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

from qasync import QEventLoop
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from av import VideoFrame

SIGNALING_SERVER_URL = "ws://10.83.65.139:8000/ws/student"

class OpenCVVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.current_frame = None

    def update_frame(self, frame):
        self.current_frame = frame

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img, frame)
            self.msleep(33)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class ClassBridgeStudentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.pc = None
        self.local_track = OpenCVVideoTrack()
        self.remote_tracks = []
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
        header.setStyleSheet("background-color: #0f172a; border-radius: 8px; border: 1px solid #1e293b;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title_label = QLabel("ClassBridge Learning Station")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")

        self.status_badge = QLabel(" INITIALIZING ")
        self.status_badge.setStyleSheet("background-color: #eab308; color: black; font-weight: bold; font-size: 12px; border-radius: 4px; padding: 4px 8px;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)

        # Stage Box (Main Display container)
        self.stage_box = QFrame()
        self.stage_box.setStyleSheet("background-color: #000000; border-radius: 12px; border: 1px solid #1e293b;")
        
        # Primary Screen View (Teacher Camera or Shared Screen)
        self.main_video = QLabel("Waiting for Teacher Stream...", self.stage_box)
        self.main_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_video.setStyleSheet("color: #64748b; font-size: 18px; font-weight: 500;")
        self.main_video.setScaledContents(True)

        # Teacher Camera Overlay Box (Shown at bottom-left when Screen Share is active)
        self.teacher_pip_card = QFrame(self.stage_box)
        self.teacher_pip_card.setStyleSheet("background-color: #0f172a; border-radius: 8px; border: 2px solid #059669;")
        self.teacher_pip_card.setFixedSize(240, 150)
        teacher_pip_layout = QVBoxLayout(self.teacher_pip_card)
        teacher_pip_layout.setContentsMargins(0, 0, 0, 0)

        self.teacher_pip_video = QLabel(self.teacher_pip_card)
        self.teacher_pip_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.teacher_pip_video.setScaledContents(True)
        teacher_pip_layout.addWidget(self.teacher_pip_video)
        self.teacher_pip_card.setVisible(False)

        # Student Self-View Box (Shown at bottom-right if toggled ON by Teacher)
        self.student_pip_card = QFrame(self.stage_box)
        self.student_pip_card.setStyleSheet("background-color: #0f172a; border-radius: 8px; border: 2px solid #2563eb;")
        self.student_pip_card.setFixedSize(220, 140)
        student_pip_layout = QVBoxLayout(self.student_pip_card)
        student_pip_layout.setContentsMargins(0, 0, 0, 0)

        self.student_video = QLabel(self.student_pip_card)
        self.student_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.student_video.setScaledContents(True)
        student_pip_layout.addWidget(self.student_video)
        self.student_pip_card.setVisible(False)

        root_layout.addWidget(header)
        root_layout.addWidget(self.stage_box, stretch=1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.stage_box.width()
        h = self.stage_box.height()
        
        # Expand primary display to full stage bounds
        self.main_video.setGeometry(0, 0, w, h)
        
        # Position Teacher Camera PiP at Bottom-Left corner
        self.teacher_pip_card.move(20, h - self.teacher_pip_card.height() - 20)

        # Position Student Self-View PiP at Bottom-Right corner
        self.student_pip_card.move(w - self.student_pip_card.width() - 20, h - self.student_pip_card.height() - 20)

    def start_camera(self):
        self.camera_thread = LocalCameraThread(camera_index=0)
        self.camera_thread.change_pixmap_signal.connect(self.update_classroom_feed)
        self.camera_thread.start()

    def update_classroom_feed(self, qt_img, cv_frame):
        pixmap = QPixmap.fromImage(qt_img)
        self.student_video.setPixmap(pixmap)
        self.local_track.update_frame(cv_frame)

    async def connect_webrtc(self):
        self.pc = RTCPeerConnection()
        self.pc.addTrack(self.local_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self.remote_tracks.append(track)
                track_index = len(self.remote_tracks)
                asyncio.create_task(self.render_remote_track(track, track_index))

        try:
            async with websockets.connect(SIGNALING_SERVER_URL) as ws:
                self.status_badge.setText(" READY ")
                self.status_badge.setStyleSheet("background-color: #22c55e; color: white;")

                async for msg in ws:
                    data = json.loads(msg)

                    # FIX #3: WebSocket payload listener for Student Self-View toggle
                    if data.get("type") == "control" and data.get("action") == "toggle-self-view":
                        should_show = data.get("enabled", False)
                        self.student_pip_card.setVisible(should_show)

                    elif "offer" in data:
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
                        self.status_badge.setStyleSheet("background-color: #dc2626; color: white;")

                    elif "candidate" in data and data["candidate"]:
                        candidate = RTCIceCandidate(
                            sdpMid=data["candidate"].get("sdpMid"),
                            sdpMLineIndex=data["candidate"].get("sdpMLineIndex"),
                            candidate=data["candidate"].get("candidate")
                        )
                        await self.pc.addIceCandidate(candidate)

        except Exception as e:
            print(f"Signaling error: {e}")
            self.status_badge.setText(" DISCONNECTED ")
            self.status_badge.setStyleSheet("background-color: #64748b; color: white;")

    async def render_remote_track(self, track, track_index):
        """
        FIX #1: Dual-Stream Renderer.
        Track 1: Teacher Camera (Rendered on main stage when solo, or PiP when sharing).
        Track 2: Teacher Screen Share (Rendered on main stage).
        """
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img)

                # Track 1: Teacher Webcam Stream
                if track_index == 1:
                    if len(self.remote_tracks) == 1:
                        # Only webcam active -> render on Main Display
                        self.main_video.setPixmap(pixmap)
                        self.teacher_pip_card.setVisible(False)
                    else:
                        # Screen sharing active -> render webcam in Bottom-Left PiP
                        self.teacher_pip_video.setPixmap(pixmap)
                        self.teacher_pip_card.setVisible(True)

                # Track 2: Teacher Screen Share Stream
                elif track_index == 2:
                    self.main_video.setPixmap(pixmap)

            except Exception as e:
                # Screen sharing stopped or track ended
                if track_index == 2:
                    self.teacher_pip_card.setVisible(False)
                if track in self.remote_tracks:
                    self.remote_tracks.remove(track)
                break

    def closeEvent(self, event):
        self.camera_thread.stop()
        if self.pc:
            asyncio.create_task(self.pc.close())
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = ClassBridgeStudentApp()
    window.showMaximized()

    loop.create_task(window.connect_webrtc())

    with loop:
        loop.run_forever()