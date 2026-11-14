import contextlib
import os
from typing import Any, Dict, List

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

from object_detector_base import ObjectDetector
from logging_config import create_logger

logger = create_logger(__name__)


@contextlib.contextmanager
def suppress_stdout_stderr():
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_fds = [os.dup(2), os.dup(1)]
        os.dup2(null_fd, 2)
        os.dup2(null_fd, 1)
        yield
    finally:
        os.dup2(save_fds[0], 2)
        os.dup2(save_fds[1], 1)
        os.close(null_fd)
        for fd in save_fds:
            os.close(fd)


class YoloObjectDetector(ObjectDetector):
    ALLOWED_CLASSES = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    }

    def __init__(self, model_name: str = "yolo11x.pt", conf_threshold: float = 0.2, enable_gpu: bool = False):
        """
        Initialize YOLO detector with optional GPU acceleration.
        
        Args:
            model_name: YOLO model name (e.g., 'yolo11x.pt')
            conf_threshold: Confidence threshold for detections
            enable_gpu: Whether to use GPU acceleration (MPS on Mac, CUDA on others)
        """
        super().__init__(model_name, conf_threshold)
        
        # Determine best available device
        self.device = self._get_best_device(enable_gpu)
        
        # Initialize model
        self.model = YOLO(model_name)
        
        # Move model to device if GPU acceleration is enabled
        if self.device != "cpu":
            self.model.to(self.device)

        # Get all class names from the model
        self.all_class_names = self.model.names

        # Filter allowed classes based on the model's available classes
        self.filtered_class_indices = {
            class_idx: class_name
            for class_idx, class_name in self.all_class_names.items()
            if class_name in self.ALLOWED_CLASSES
        }
        
        logger.info(f"YOLO detector initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Filtered classes: {len(self.filtered_class_indices)}")

    def _get_best_device(self, enable_gpu: bool) -> str:
        """Determine the best available device for inference."""
        if not enable_gpu:
            logger.info("GPU acceleration disabled - using CPU")
            return "cpu"
        
        # Check for MPS (Mac M-series)
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) available - using GPU acceleration")
            return "mps"
        
        # Check for CUDA
        if torch.cuda.is_available():
            logger.info("CUDA available - using GPU acceleration")
            return "cuda"
        
        logger.info("No GPU acceleration available - using CPU")
        return "cpu"

    @classmethod
    def get_allowed_classes(cls) -> List[str]:
        return sorted(list(cls.ALLOWED_CLASSES))

    def get_model_classes(self) -> List[str]:
        return [self.all_class_names[i] for i in sorted(self.all_class_names.keys())]

    def detect_objects(
        self, video_path: str, num_frames: int = 10
    ) -> List[Dict[str, Any]]:
        with suppress_stdout_stderr():
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stride = max(1, frame_count // num_frames)
        frames_to_sample = range(0, frame_count, stride)[:num_frames]

        all_detections = []
        for frame_num in tqdm(
            frames_to_sample,
            desc=f"Processing {len(frames_to_sample)} frames of {video_path}",
            unit="frame",
            colour="green",
            position=1,
            leave=False,
        ):
            with suppress_stdout_stderr():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
            if not ret:
                continue

            results = self.model(frame, conf=self.conf_threshold, verbose=False, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_idx = int(box.cls)
                    if class_idx in self.filtered_class_indices:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        all_detections.append(
                            {
                                "type": "DETECTED_OBJECT",
                                "value": self.filtered_class_indices[class_idx],
                                "confidence": f"{float(box.conf):.2f}",
                                "frame_number": int(frame_num),
                                "bounding_box": {
                                    "x1": int(x1),
                                    "y1": int(y1),
                                    "x2": int(x2),
                                    "y2": int(y2),
                                },
                            }
                        )

        with suppress_stdout_stderr():
            cap.release()

        return all_detections

    def track_objects(
        self, video_path: str, num_frames: int = 10
    ) -> List[Dict[str, Any]]:
        with suppress_stdout_stderr():
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stride = max(1, frame_count // num_frames)
        frames_to_sample = range(0, frame_count, stride)[:num_frames]

        all_tracks = []
        prev_track_ids = set()

        if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers:
            self.model.predictor.trackers[0].reset()

        for frame_num in tqdm(
            frames_to_sample,
            desc=f"Tracking objects across {len(frames_to_sample)} frames of {video_path}",
            unit="frame",
            colour="green",
            position=1,
            leave=False,
        ):
            with suppress_stdout_stderr():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
            if not ret:
                continue

            results = self.model.track(
                frame, persist=True, conf=self.conf_threshold, verbose=False, device=self.device
            )

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_idx = int(box.cls)
                    if class_idx in self.filtered_class_indices:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        track_id = None
                        if hasattr(box, "id") and box.id is not None:
                            track_id = int(box.id)

                        is_new_track = False
                        if track_id is not None and track_id not in prev_track_ids:
                            is_new_track = True
                            prev_track_ids.add(track_id)

                        detection = {
                            "type": "TRACKED_OBJECT",
                            "value": self.filtered_class_indices[class_idx],
                            "confidence": f"{float(box.conf):.2f}",
                            "frame_number": int(frame_num),
                            "bounding_box": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                            },
                        }

                        if track_id is not None:
                            detection["track_id"] = track_id
                            detection["is_new_track"] = is_new_track

                        all_tracks.append(detection)

        with suppress_stdout_stderr():
            cap.release()

        return all_tracks
