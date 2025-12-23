from pathlib import Path

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

        current_objects = self.extract_objects_from_gt_state()
        new_objects = []

        grid_dim_objects = sorted(obj.split(":")[0] for obj in current_objects if obj.startswith('x'))
        grid_dim = int(grid_dim_objects[-1][1:])
        for i in range(1, grid_dim + 1):
            for j in range(1, grid_dim + 1):
                new_objects.append(f"p_{i}_{j}:position")
        tile_objects = sorted(obj for obj in current_objects if obj.startswith('t'))
        new_objects.extend(["t_" + obj.split(":")[0][1:] + ":tile" for obj in tile_objects])
        self.fewshot_examples = [(init_state_image_path, new_objects)]

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for Hanoi object detection."""
        return npuzzle_object_detection_prompt

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract object detections from LLM response."""
        # Pattern to match object detection in "<name>:<type>" format
        # where name can include digits (e.g., "t_1:tile", "p_1_2:position")
        return r'(?:t_\d+:tile|p_\d+_\d+:position)'
