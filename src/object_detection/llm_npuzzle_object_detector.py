from pathlib import Path
from typing import List

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.llms.domains.n_puzzle.prompts import npuzzle_object_detection_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMNpuzzleObjectDetector(LLMObjectDetector):
    """
    LLM-based object detector for the Slide (8-puzzle) domain.

    The LLM returns objects as "d1:disc", "p1:peg", etc.
    This class maps them to the format expected by the fluent classifier:
    - d1, d2, d3, ... (discs stay the same)
    - p1 → peg1, p2 → peg2, p3 → peg3 (pegs are renamed)
    """

    def __init__(
            self,
            llm_backend: ImageLLMBackend,
            init_state_image_path: Path,
            temperature: float = None,
            inference_mode: bool = False
    ):
        super().__init__(
            llm_backend=llm_backend,
            init_state_image_path=init_state_image_path,
            temperature=temperature,
            inference_mode=inference_mode
        )

        self.imaged_obj_to_gym_obj_name = {
            **{f"t_{i}": f"t_{i}" for i in range(1, 25)},
            **{f"p_{i}_{j}": f"p_{i}_{j}" for i in range(1, 6) for j in range(1, 6)}  # positions up to 5x5
        }

        self.fewshot_examples = [
            (init_state_image_path, self._typed_objects(self.extract_objects_from_gt_state()))
        ]

    @staticmethod
    def _typed_objects(objects: List[str]) -> List[str]:
        """The GT state's objects in the typed n_puzzle naming (``p_i_j:position``, ``t_k:tile``).

        Accepts the typed naming as is and translates the untyped slidetile
        naming (``x1``, ``y1``, ``t1``).
        """
        columns = sorted(obj.split(":")[0] for obj in objects if obj.startswith('x'))
        if not columns:
            return sorted(objects)
        grid_dim = int(columns[-1][1:])
        typed = [f"p_{i}_{j}:position" for i in range(1, grid_dim + 1) for j in range(1, grid_dim + 1)]
        tiles = sorted(obj for obj in objects if obj.startswith('t'))
        return typed + ["t_" + obj.split(":")[0][1:] + ":tile" for obj in tiles]

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for Hanoi object detection."""
        return npuzzle_object_detection_prompt

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract object detections from LLM response."""
        # Pattern to match object detection in "<name>:<type>" format
        # where name can include digits (e.g., "t_1:tile", "p_1_2:position")
        return r'(?:t_\d+:tile|p_\d+_\d+:position)'
