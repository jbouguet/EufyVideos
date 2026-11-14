from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from object_detector_base import ObjectDetector


class Florence2ObjectDetector(ObjectDetector):
    ALLOWED_CLASSES = {
        "person",
        "dog",
        "cat",
        "bird",
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "dining table",
        "chair",
        "umbrella",
        "potted plant",
    }

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        conf_threshold: float = 0.2,
    ):
        super().__init__(model_name, conf_threshold)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading Florence-2 model from {model_name}")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
        print("Florence-2 model loaded successfully")

    @classmethod
    def get_allowed_classes(cls) -> List[str]:
        return sorted(list(cls.ALLOWED_CLASSES))

    def get_model_classes(self) -> List[str]:
        return self.get_allowed_classes()

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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            prompt = "<OD>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                self.device, self.torch_dtype
            )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            try:
                parsed_answer = self.processor.post_process_generation(
                    generated_text, task="<OD>", image_size=(image.width, image.height)
                )

                if isinstance(parsed_answer, dict) and "<OD>" in parsed_answer:
                    od_result = parsed_answer["<OD>"]
                    bboxes = od_result.get("bboxes", [])
                    labels = od_result.get("labels", [])

                    for bbox, label in zip(bboxes, labels):
                        if label in self.ALLOWED_CLASSES:
                            x1, y1, x2, y2 = bbox
                            all_detections.append(
                                {
                                    "type": "OBJECT",
                                    "value": label,
                                    "confidence": "1.00",  # The model doesn't provide confidence scores
                                    "frame_number": int(frame_num),
                                    "bounding_box": {
                                        "x1": int(x1),
                                        "y1": int(y1),
                                        "x2": int(x2),
                                        "y2": int(y2),
                                    },
                                }
                            )
                else:
                    print(f"Unexpected parsed_answer format: {parsed_answer}")
            except Exception as e:
                print(f"Error processing frame {frame_num}: {str(e)}")
                print(f"Generated text: {generated_text}")

        cap.release()
        return all_detections

    def track_objects(
        self, video_path: str, num_frames: int = 10
    ) -> List[Dict[str, Any]]:
        return self.detect_objects(video_path, num_frames)
