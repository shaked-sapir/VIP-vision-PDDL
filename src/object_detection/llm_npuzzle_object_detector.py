from pathlib import Path

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

    def __init__(self, openai_apikey: str, model: str, temperature: float, init_state_image_path: Path):
        super().__init__(
            openai_apikey=openai_apikey,
            model=model,
            temperature=temperature,
            init_state_image_path=init_state_image_path
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
        # where name can include digits (e.g., "d1:disc", "p1:peg")
        return r'(?:t_\d+:tile|p_\d+_\d+:position)'

    # def detect(self, image: Union[Path, str], *args, **kwargs) -> Dict[str, List[str]]:
    #     """
    #     Detect objects in image and map peg names from p1, p2, p3 to peg1, peg2, peg3.
    #
    #     :param image: Path to the image
    #     :return: Dictionary mapping type to list of object names
    #     """
    #     # Call parent's detect method
    #     detected_objects = super().detect(image, *args, **kwargs)
    #
    #     # Map peg names: p1 → peg1, p2 → peg2, etc.
    #     if 'peg' in detected_objects:
    #         mapped_pegs = []
    #         for peg_name in detected_objects['peg']:
    #             if peg_name.startswith('p') and peg_name[1:].isdigit():
    #                 # Convert p1 → peg1
    #                 mapped_name = f"peg{peg_name[1:]}"
    #                 mapped_pegs.append(mapped_name)
    #             else:
    #                 # Keep as-is if already in correct format
    #                 mapped_pegs.append(peg_name)
    #         detected_objects['peg'] = mapped_pegs
    #
    #     return detected_objects
