import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import streamlit as st


class TrafficSignDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLOv8 detector for traffic signs

        Args:
            model_path: Path to YOLO model weights
        """
        self.model_path = model_path
        self.model = None
        self.class_names = self._get_traffic_sign_classes()
        self.load_model()

    def _get_traffic_sign_classes(self):
        """Return traffic sign class names (GTSDB/GTSRB classes)"""
        return {
            0: 'Speed limit (20km/h)',
            1: 'Speed limit (30km/h)',
            2: 'Speed limit (50km/h)',
            3: 'Speed limit (60km/h)',
            4: 'Speed limit (70km/h)',
            5: 'Speed limit (80km/h)',
            6: 'End of speed limit (80km/h)',
            7: 'Speed limit (100km/h)',
            8: 'Speed limit (120km/h)',
            9: 'No passing',
            10: 'No passing for trucks',
            11: 'Stop',
            12: 'Yield',
            13: 'Priority road',
            14: 'Roundabout',
            15: 'No vehicles',
            16: 'Trucks prohibited',
            17: 'No entry',
            18: 'General caution',
            19: 'Dangerous curve left',
            20: 'Dangerous curve right',
            21: 'Double curve',
            22: 'Bumpy road',
            23: 'Slippery road',
            24: 'Road narrows',
            25: 'Road work',
            26: 'Traffic signals',
            27: 'Pedestrians',
            28: 'Children crossing',
            29: 'Bicycles crossing',
            30: 'Beware of ice/snow',
            31: 'Wild animals crossing',
            32: 'End of all speed limits',
            33: 'Turn right',
            34: 'Turn left',
            35: 'Ahead only',
            36: 'Go straight or right',
            37: 'Go straight or left',
            38: 'Keep right',
            39: 'Keep left',
            40: 'Roundabout mandatory',
            41: 'End of no passing',
            42: 'End of no passing by trucks'
        }

    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            st.info(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("Downloading YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')
            st.success("✅ Model downloaded and loaded successfully")

    def detect(self, image, confidence=0.5, nms_threshold=0.45):
        """
        Perform traffic sign detection on image

        Args:
            image: Input image (numpy array)
            confidence: Confidence threshold for detections
            nms_threshold: Non-Maximum Suppression threshold

        Returns:
            detections: List of detection dictionaries
            annotated_image: Image with bounding boxes and labels
        """
        if self.model is None:
            self.load_model()

        # Run inference
        results = self.model(image, conf=confidence, iou=nms_threshold)

        detections = []
        annotated_image = image.copy()

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence_score = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Map class id to name
                    class_name = self.class_names.get(class_id, f'Class_{class_id}')

                    # Store detection
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence_score,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)

        # Draw detections on image
        annotated_image = self.draw_detections(annotated_image, detections)

        return detections, annotated_image

    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image

        Args:
            image: Input image
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # Define colors
            color = (0, 255, 0)  # Green for traffic signs
            text_color = (255, 255, 255)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"{class_name}: {confidence:.2f}"

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2
            )

        return image

    def predict_batch(self, images, confidence=0.5):
        """
        Perform batch prediction on multiple images

        Args:
            images: List of images
            confidence: Confidence threshold

        Returns:
            List of detection results
        """
        results = []
        for image in images:
            detections, annotated = self.detect(image, confidence)
            results.append({
                'detections': detections,
                'annotated_image': annotated
            })
        return results

    def export_onnx(self, save_path='model.onnx'):
        """
        Export model to ONNX format

        Args:
            save_path: Path to save ONNX model
        """
        if self.model is None:
            self.load_model()

        # Export to ONNX
        self.model.export(format='onnx', imgsz=640)
        st.success(f"✅ Model exported to ONNX format at {save_path}")
        return save_path

    def export_tflite(self, save_path='model.tflite'):
        """
        Export model to TFLite format

        Args:
            save_path: Path to save TFLite model
        """
        if self.model is None:
            self.load_model()

        # Export to TFLite
        self.model.export(format='tflite', imgsz=640)
        st.success(f"✅ Model exported to TFLite format at {save_path}")
        return save_path

    def optimize_for_inference(self, device='auto'):
        """
        Optimize model for faster inference

        Args:
            device: 'cuda', 'cpu', or 'auto'
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)

        # Additional optimizations
        if device == 'cuda':
            # Use half precision for faster inference
            self.model.half()
            st.info("🚀 Model optimized for CUDA with half precision")
        else:
            st.info("💻 Model running on CPU")

        return device