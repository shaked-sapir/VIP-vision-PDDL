"""Mixin providing LLM-based object detector and fluent classifier initialization."""

from pathlib import Path
from typing import Dict, List, Type

from pddl_plus_parser.lisp_parsers import DomainParser

from src.fluent_classification.base_fluent_classifier import FluentClassifier
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.object_detection.base_object_detector import ObjectDetector


class LLMVisualComponentsMixin:
    """Mixin that wires up LLM-based detector + classifier via class attributes.

    Subclasses set:
        detector_class: The LLM object detector class for the domain.
        classifier_class: The LLM fluent classifier class for the domain.

    Uses cooperative __init__ — pass pddl_domain_file and vendor as keyword args.
    """

    detector_class: Type[ObjectDetector]
    classifier_class: Type[FluentClassifier]

    def __init__(self, *, pddl_domain_file: Path, vendor: str = "openai", **kwargs):
        super().__init__(**kwargs)
        self.vendor = vendor
        self.domain = DomainParser(pddl_domain_file, partial_parsing=True).parse_domain()

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        """Override point for work before visual component init (e.g. ensure trajectory JSON)."""
        pass

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """Standard LLM init: run pre-hook, detect objects, create classifier."""
        self._pre_init_hook(init_state_image_path)

        self.object_detector = self.detector_class(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor, model_type="object_detection"),
            init_state_image_path=init_state_image_path,
        )
        detected_objects: Dict[str, List[str]] = self.object_detector.detect(
            str(init_state_image_path))

        self.fluent_classifier = self.classifier_class(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor, model_type="fluent_classification"),
            type_to_objects=detected_objects,
            init_state_image_path=init_state_image_path,
        )
