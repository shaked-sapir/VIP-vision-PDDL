import re

from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier
from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMBlocksImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Blocksworld domain."""

    detector_class = LLMBlocksObjectDetector
    classifier_class = LLMBlocksFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Rename gym-format actions: pick-up→pick_up, put-down→put_down, remove robot param."""
        return (action_str.replace('pick-up', 'pick_up')
                .replace('put-down', 'put_down')
                .replace(', robot:robot', ''))

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Apply blocksworld-specific transformations to trajectory JSON."""
        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    literals = step[state_key]['literals']
                    new_literals = []
                    for lit in literals:
                        if lit == "handempty(robot:robot)":
                            new_literals.append("handempty()")
                        elif lit == "handfull(robot:robot)":
                            continue
                        else:
                            new_literals.append(lit)
                    step[state_key]['literals'] = new_literals

            if 'ground_action' in step:
                action = step['ground_action']
                action = re.sub(r'pick-up\(([^,]+):block,\s*robot:robot\)', r'pick_up(\1:block)', action)
                action = re.sub(r'put-down\(([^,]+):block,\s*robot:robot\)', r'put_down(\1:block)', action)
                action = re.sub(r'stack\(([^,]+):block,\s*([^,]+):block,\s*robot:robot\)', r'stack(\1:block, \2:block)', action)
                action = re.sub(r'unstack\(([^,]+):block,\s*([^,]+):block,\s*robot:robot\)', r'unstack(\1:block, \2:block)', action)
                step['ground_action'] = action

        return gt_trajectory_json
