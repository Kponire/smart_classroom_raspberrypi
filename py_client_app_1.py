import sys
import json
import asyncio
import cv2
import numpy as np

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFrame, QStackedLayout)
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

        # Main Stage Container (Overlaid Google Meet format)
        self.stage_box = QFrame()
        self.stage_box.setStyleSheet("background-color: #000000; border-radius: 12px; border: 1px solid #1e293b;")
        
        # Main Display: Large view showing Teacher Feed or Screen Share
        self.teacher_video = QLabel("Waiting for Teacher Stream...", self.stage_box)
        self.teacher_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.teacher_video.setStyleSheet("color: #64748b; font-size: 18px; font-weight: 500;")
        self.teacher_video.setScaledContents(True)

        # Self View PiP Frame (Bottom Right Corner)
        self.pip_card = QFrame(self.stage_box)
        self.pip_card.setStyleSheet("background-color: #0f172a; border-radius: 8px; border: 2px solid #2563eb;")
        self.pip_card.setFixedSize(220, 140)
        pip_layout = QVBoxLayout(self.pip_card)
        pip_layout.setContentsMargins(0, 0, 0, 0)

        self.classroom_video = QLabel(self.pip_card)
        self.classroom_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.classroom_video.setScaledContents(True)
        pip_layout.addWidget(self.classroom_video)

        # Hide local camera self-view by default
        self.pip_card.setVisible(False)

        root_layout.addWidget(header)
        root_layout.addWidget(self.stage_box, stretch=1)

    def resizeEvent(self, event):
        """Keep layout responsively positioned in full screen stage."""
        super().resizeEvent(event)
        w = self.stage_box.width()
        h = self.stage_box.height()
        
        # Expand teacher stream across the full stage box
        self.teacher_video.setGeometry(0, 0, w, h)
        
        # Position self-view in bottom-right corner with a 20px padding offset
        pip_w = self.pip_card.width()
        pip_h = self.pip_card.height()
        self.pip_card.move(w - pip_w - 20, h - pip_h - 20)

    def start_camera(self):
        self.camera_thread = LocalCameraThread(camera_index=0)
        self.camera_thread.change_pixmap_signal.connect(self.update_classroom_feed)
        self.camera_thread.start()

    def update_classroom_feed(self, qt_img, cv_frame):
        pixmap = QPixmap.fromImage(qt_img)
        self.classroom_video.setPixmap(pixmap)
        self.local_track.update_frame(cv_frame)

    async def connect_webrtc(self):
        self.pc = RTCPeerConnection()
        self.pc.addTrack(self.local_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.create_task(self.render_remote_track(track))

        try:
            async with websockets.connect(SIGNALING_SERVER_URL) as ws:
                self.status_badge.setText(" READY ")
                self.status_badge.setStyleSheet("background-color: #22c55e; color: white;")

                async for msg in ws:
                    data = json.loads(msg)

                    # Toggle self-view state dynamically when teacher flips the switch
                    if data.get("type") == "control" and data.get("action") == "toggle-self-view":
                        should_show = data.get("enabled", False)
                        self.pip_card.setVisible(should_show)

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

    async def render_remote_track(self, track):
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                self.teacher_video.setPixmap(QPixmap.fromImage(qt_img))
            except Exception as e:
                print(f"Remote track error: {e}")
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