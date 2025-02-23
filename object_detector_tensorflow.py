# WARNING: As of 2/19/2025, the tensorflow library is not yet supported by Python 3.13.
# This module should not be used until tensorflow adds support for Python 3.13.

import glob
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from object_detector_base import ObjectDetector

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TensorFlowObjectDetector(ObjectDetector):
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
        "street sign",
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
        "hat",
        "backpack",
        "umbrella",
        "shoe",
        "eye glasses",
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
        "plate",
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
        "mirror",
        "dining table",
        "window",
        "desk",
        "toilet",
        "door",
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
        "blender",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "hair brush",
    }

    def __init__(
        self,
        model_name: str = "efficientdet_d0_coco17_tpu-32",
        conf_threshold: float = 0.2,
    ):
        super().__init__(model_name, conf_threshold)
        self.model = self._load_model()

        # Load COCO class names
        script_dir = os.path.dirname(os.path.abspath(__file__))
        coco_classes_path = os.path.join(script_dir, "coco_classes.txt")
        with open(coco_classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.filtered_class_indices = {
            idx: name
            for idx, name in enumerate(self.class_names)
            if name in self.ALLOWED_CLASSES
        }

    def _load_model(self):
        model_file = tf.keras.utils.get_file(
            fname=f"{self.model_name}.tar.gz",
            origin=f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{self.model_name}.tar.gz",
            extract=True,
            cache_subdir="models",
        )
        model_dir = os.path.dirname(model_file)

        # Find the saved_model directory
        saved_model_dirs = glob.glob(
            os.path.join(model_dir, "**", "saved_model"), recursive=True
        )
        if not saved_model_dirs:
            raise FileNotFoundError(
                f"Could not find saved_model directory in {model_dir}"
            )

        saved_model_path = saved_model_dirs[0]
        print(f"Loading model from: {saved_model_path}")

        try:
            model = tf.saved_model.load(saved_model_path)
            return model.signatures["serving_default"]
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @classmethod
    def get_allowed_classes(cls) -> List[str]:
        return sorted(list(cls.ALLOWED_CLASSES))

    def get_model_classes(self) -> List[str]:
        return self.class_names

    def detect_objects(
        self, video_path: str, num_frames: int = 10
    ) -> List[Dict[str, Any]]:
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = input_tensor[tf.newaxis, ...]

            output_dict = self.model(input_tensor)

            boxes = output_dict["detection_boxes"][0].numpy()
            classes = output_dict["detection_classes"][0].numpy().astype(np.int32)
            scores = output_dict["detection_scores"][0].numpy()

            for box, class_id, score in zip(boxes, classes, scores):
                if (
                    score >= self.conf_threshold
                    and (class_id - 1) in self.filtered_class_indices
                ):
                    y1, x1, y2, x2 = box
                    all_detections.append(
                        {
                            "type": "OBJECT",
                            "value": self.filtered_class_indices[class_id - 1],
                            "confidence": f"{float(score):.2f}",
                            "frame_number": int(frame_num),
                            "bounding_box": {
                                "x1": int(x1 * frame.shape[1]),
                                "y1": int(y1 * frame.shape[0]),
                                "x2": int(x2 * frame.shape[1]),
                                "y2": int(y2 * frame.shape[0]),
                            },
                        }
                    )

        cap.release()
        return all_detections

    def track_objects(
        self, video_path: str, num_frames: int = 10
    ) -> List[Dict[str, Any]]:
        return self.detect_objects(video_path, num_frames)
