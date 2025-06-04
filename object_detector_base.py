from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ObjectDetector(ABC):
    def __init__(self, model_name: str, conf_threshold: float = 0.2):
        self.model_name = model_name
        self.conf_threshold = conf_threshold

    @classmethod
    @abstractmethod
    def get_allowed_classes(cls) -> List[str]:
        """Return a list of allowed class names for detection"""
        pass

    @abstractmethod
    def get_model_classes(self) -> List[str]:
        """Return a list of all class names supported by the model"""
        pass

    @abstractmethod
    def detect_objects(
        self, video_path: str, num_frames: int = 10
    ) -> List[Dict[str, Any]]:
        """Detect objects in video frames"""
        pass

    @abstractmethod
    def track_objects(
        self, video_path: str, num_frames: int = 10
    ) -> List[Dict[str, Any]]:
        """Track objects across video frames"""
        pass


class ObjectDetectorFactory:
    @staticmethod
    def create_detector(model: str, conf_threshold: float = 0.2, batch_size: int = 8) -> ObjectDetector:
        from object_detector_florence2 import Florence2ObjectDetector

        # from object_detector_tensorflow import TensorFlowObjectDetector
        from object_detector_yolo import YoloObjectDetector
        from object_detector_yolo_optimized import OptimizedYoloObjectDetector
        from tag_processor import Model

        match model:
            # case Model.TENSORFLOW.value:
            #    return TensorFlowObjectDetector(
            #        model_name="efficientdet_d0_coco17_tpu-32",
            #        conf_threshold=conf_threshold,
            #    )
            case Model.FLORENCE2.value:
                return Florence2ObjectDetector(
                    model_name="microsoft/Florence-2-large",
                    conf_threshold=conf_threshold,
                )
            
            # Original YOLO detectors
            case Model.YOLO11N.value:
                return YoloObjectDetector(
                    model_name="yolo11n.pt", conf_threshold=conf_threshold
                )
            case Model.YOLO11S.value:
                return YoloObjectDetector(
                    model_name="yolo11s.pt", conf_threshold=conf_threshold
                )
            case Model.YOLO11M.value:
                return YoloObjectDetector(
                    model_name="yolo11m.pt", conf_threshold=conf_threshold
                )
            case Model.YOLO11L.value:
                return YoloObjectDetector(
                    model_name="yolo11l.pt", conf_threshold=conf_threshold
                )
            case Model.YOLO11X.value:
                return YoloObjectDetector(
                    model_name="yolo11x.pt", conf_threshold=conf_threshold
                )
            
            # Optimized YOLO detectors with GPU acceleration and batch processing
            case Model.YOLO11N_OPTIMIZED.value:
                return OptimizedYoloObjectDetector(
                    model_name="yolo11n.pt", 
                    conf_threshold=conf_threshold,
                    batch_size=batch_size
                )
            case Model.YOLO11S_OPTIMIZED.value:
                return OptimizedYoloObjectDetector(
                    model_name="yolo11s.pt", 
                    conf_threshold=conf_threshold,
                    batch_size=batch_size
                )
            case Model.YOLO11M_OPTIMIZED.value:
                return OptimizedYoloObjectDetector(
                    model_name="yolo11m.pt", 
                    conf_threshold=conf_threshold,
                    batch_size=batch_size
                )
            case Model.YOLO11L_OPTIMIZED.value:
                return OptimizedYoloObjectDetector(
                    model_name="yolo11l.pt", 
                    conf_threshold=conf_threshold,
                    batch_size=batch_size
                )
            case Model.YOLO11X_OPTIMIZED.value:
                return OptimizedYoloObjectDetector(
                    model_name="yolo11x.pt", 
                    conf_threshold=conf_threshold,
                    batch_size=batch_size
                )
            
            case _:
                raise ValueError(f"Invalid model: {model}")
