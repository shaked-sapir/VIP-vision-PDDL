from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier
from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMBlocksImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Blocksworld domain.

    The gym ``blocks_operator_actions`` env emits hyphenated action names
    (``pick-up``/``put-down``) and a nullary ``handfull()`` that the canonical
    domain does not use. These helpers convert the raw gym GT trajectory into
    the canonical format: underscore action names and no ``handfull``.
    """

    detector_class = LLMBlocksObjectDetector
    classifier_class = LLMBlocksFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Rename gym action names to the canonical underscore form."""
        return action_str.replace('pick-up', 'pick_up').replace('put-down', 'put_down')

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Drop ``handfull()`` from states and canonicalize action names."""
        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    step[state_key]['literals'] = [
                        lit for lit in step[state_key]['literals']
                        if lit != "handfull()"
                    ]

            if 'ground_action' in step:
                step['ground_action'] = self._rename_ground_action(step['ground_action'])

        return gt_trajectory_json
