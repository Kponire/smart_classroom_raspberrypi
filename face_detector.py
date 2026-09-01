"""
YOLO Face Detection Module for Student Counting
Uses YOLOv8n-face model for accurate face detection
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaceDetection:
    """Data class for face detection results"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]
    area: int
    
class YOLOFaceDetector:
    """YOLO-based face detector for student counting"""
    
    def __init__(self, model_path: str = "yolov8n-face.pt", device: str = None):
        """
        Initialize YOLO face detector
        
        Args:
            model_path: Path to YOLO model weights
            device: 'cpu' or 'cuda' (auto-detect if None)
        """
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model with weights
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"YOLO model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to a known face detection model
            self.model = YOLO('yolov8n-face.pt')  # Will download if not present
            logger.info("Downloaded default YOLO face model")
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.frame_skip = 2  # Process every Nth frame
        self.frame_counter = 0
        
        # Student tracking
        self.student_count = 0
        self.detected_faces = []
        self.tracked_students = {}
        self.next_student_id = 1
        
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame using YOLO
        
        Args:
            frame: BGR image from camera
            
        Returns:
            List of FaceDetection objects
        """
        if frame is None or frame.size == 0:
            return []
            
        # Skip frames for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return self.detected_faces
        
        try:
            # Run YOLO inference
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               device=self.device,
                               verbose=False)
            
            faces = []
            
            if results and len(results) > 0:
                for result in results[0]:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            confidence = float(box.conf[0])
                            
                            # Calculate center and area
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            area = (x2 - x1) * (y2 - y1)
                            
                            face = FaceDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                center=(center_x, center_y),
                                area=area
                            )
                            faces.append(face)
            
            # Filter and merge overlapping detections
            faces = self._filter_detections(faces)
            self.detected_faces = faces
            
            # Update student count
            self.student_count = len(faces)
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return self.detected_faces
    
    def _filter_detections(self, faces: List[FaceDetection]) -> List[FaceDetection]:
        """
        Filter and merge overlapping face detections using Non-Maximum Suppression
        
        Args:
            faces: List of FaceDetection objects
            
        Returns:
            Filtered list of FaceDetection objects
        """
        if len(faces) <= 1:
            return faces
            
        # Sort by confidence descending
        faces.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        used_indices = set()
        
        for i, face1 in enumerate(faces):
            if i in used_indices:
                continue
                
            x1_1, y1_1, x2_1, y2_1 = face1.bbox
            area1 = face1.area
            
            # Check for overlaps with other faces
            overlapping = False
            for j, face2 in enumerate(faces):
                if j <= i or j in used_indices:
                    continue
                    
                x1_2, y1_2, x2_2, y2_2 = face2.bbox
                area2 = face2.area
                
                # Calculate intersection
                x_left = max(x1_1, x1_2)
                y_top = max(y1_1, y1_2)
                x_right = min(x2_1, x2_2)
                y_bottom = min(y2_1, y2_2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    iou = intersection_area / min(area1, area2)
                    
                    if iou > self.iou_threshold:
                        used_indices.add(j)
                        overlapping = True
                        
            if not overlapping:
                filtered.append(face1)
                used_indices.add(i)
                
        return filtered
    
    def draw_detections(self, frame: np.ndarray, faces: Optional[List[FaceDetection]] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input BGR image
            faces: List of FaceDetection objects (uses stored detections if None)
            
        Returns:
            Annotated image
        """
        if faces is None:
            faces = self.detected_faces
            
        annotated_frame = frame.copy()
        
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            
            # Draw bounding box with color based on confidence
            confidence = face.confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with student number
            label = f"Student {i+1}: {confidence:.2f}"
            
            # Background for label
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(annotated_frame, face.center, 3, (0, 255, 0), -1)
        
        return annotated_frame
    
    def get_student_count(self) -> int:
        """Get current student count"""
        return self.student_count
    
    def update_parameters(self, confidence_threshold: float = None, 
                         iou_threshold: float = None, 
                         frame_skip: int = None):
        """Update detection parameters"""
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.1, min(1.0, confidence_threshold))
        if iou_threshold is not None:
            self.iou_threshold = max(0.1, min(1.0, iou_threshold))
        if frame_skip is not None:
            self.frame_skip = max(1, frame_skip)
            
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'student_count': self.student_count,
            'faces_detected': len(self.detected_faces),
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'frame_skip': self.frame_skip,
            'timestamp': datetime.now().isoformat()
        }