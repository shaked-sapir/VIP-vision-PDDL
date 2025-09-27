from typing import List, Dict, Tuple

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection, BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms as T

import cv2
import numpy as np

from src.types import ObjectLabel
from src.utils.visualize import to_int_rgb, find_exact_rgb_color_mask, NormalizedRGB
from src.object_detection.base_object_detector import ObjectDetector
from src.object_detection.bounded_object import BoundedObject


class CVObjectDetector(ObjectDetector):
    def __init__(self, object_descriptions: List[str] = None):
        self._processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self._model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval()

        # self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").eval()
        # Define object descriptions as prompts
        self._object_descriptions = object_descriptions

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def object_descriptions(self):
        return self._object_descriptions

    def caption(self, image: cv2.typing.MatLike, **kwargs) -> str:
        prompt = "Describe the image using object names and their spatial relationships."
        inputs = self.blip_processor(image, prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.blip_model.generate(**inputs, max_new_tokens=100)

        return self.blip_processor.decode(output[0], skip_special_tokens=True)

    def detect(self, image: cv2.typing.MatLike, **kwargs) -> List[BoundedObject]:

        inputs = self.processor(text=self.object_descriptions, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        height, width = image.size
        target_sizes = torch.tensor([[height, width]], dtype=torch.float)  # 👈 float, not complex
        # target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)[0]

        detected_objects = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= 0.1:
                label = self.object_descriptions[label_idx.item()]
                bbox = box.tolist()
                detected_objects.append(
                    BoundedObject(
                        obj_type=label.split()[1],
                        name=label.split()[0],
                        x_anchor=bbox[0],
                        y_anchor=bbox[1],
                        width=bbox[2] - bbox[0],
                        height=bbox[3] - bbox[1],
                        confidence=float(score)
                    )
                )
        return detected_objects
