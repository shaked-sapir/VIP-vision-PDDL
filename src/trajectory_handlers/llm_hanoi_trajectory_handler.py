import re
from pathlib import Path

from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler
from src.utils.pddl_gym import translate_problem_pddl_text


class LLMHanoiImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Hanoi domain."""

    detector_class = LLMHanoiObjectDetector
    classifier_class = LLMHanoiFluentClassifier

    # ── Shared gym→eval schema mapping (used by both the trajectory JSON and
    #    the problem .pddl translation) ───────────────────────────────────
    _SPLIT_PREDICATES = ("smaller", "on", "clear")

    @staticmethod
    def _object_type(name: str) -> str:
        """Map a hanoi object name to its typed schema type."""
        return "peg" if name.startswith("peg") else "disc"

    @classmethod
    def _split_predicate_name(cls, pred: str, args: list[str]) -> str:
        """Split a shared hanoi predicate into its -disc/-peg variant."""
        if pred in cls._SPLIT_PREDICATES:
            suffix = "peg" if any(a.startswith("peg") for a in args) else "disc"
            return f"{pred}-{suffix}"
        return pred

    def translate_problem_pddl(self, pddl_path: Path) -> None:
        """Rewrite a generated hanoi problem .pddl into the typed eval schema."""
        text = pddl_path.read_text()
        translated = translate_problem_pddl_text(
            text,
            object_type_fn=self._object_type,
            literal_fn=lambda pred, args: (self._split_predicate_name(pred, args), args),
        )
        pddl_path.write_text(translated)

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Rename move(...) to move_peg_peg/move_disc_peg/etc based on argument types."""
        name_end = action_str.index('(')
        args_str = action_str[name_end:]
        args = args_str[1:-1].split(',')
        names = [a.split(':')[0].strip() for a in args]
        c2 = "peg" if names[1].startswith("peg") else "disc"
        c3 = "peg" if names[2].startswith("peg") else "disc"
        return f"move_{c2}_{c3}{args_str}"

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Transform hanoi trajectory from pddlgym untyped format to typed format."""

        def transform_object_type(obj: str) -> str:
            if ':default' not in obj:
                return obj
            name = obj.split(':')[0]
            if name.startswith('peg'):
                return f"{name}:peg"
            elif name.startswith('d'):
                return f"{name}:disc"
            return obj

        def contains_peg(literal: str) -> bool:
            match = re.match(r'\w+\((.*)\)', literal)
            if match:
                args = [arg.split(':')[0].strip() for arg in match.group(1).split(',')]
                return any(arg.startswith('peg') for arg in args)
            return False

        def transform_literal(lit: str) -> str:
            if lit.startswith('smaller('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('smaller(', f'smaller-{suffix}(')
            elif lit.startswith('on('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('on(', f'on-{suffix}(')
            elif lit.startswith('clear('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('clear(', f'clear-{suffix}(')
            lit = re.sub(r'(peg\d+):default', r'\1:peg', lit)
            lit = re.sub(r'(d\d+):default', r'\1:disc', lit)
            return lit

        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    step[state_key]['literals'] = [transform_literal(lit)
                                                   for lit in step[state_key]['literals']]
                if state_key in step and 'objects' in step[state_key]:
                    step[state_key]['objects'] = [transform_object_type(obj)
                                                  for obj in step[state_key]['objects']]
                if state_key in step and 'goal' in step[state_key]:
                    step[state_key]['goal'] = [transform_literal(lit)
                                               for lit in step[state_key]['goal']]

            if 'ground_action' in step:
                try:
                    action = self._rename_ground_action(step['ground_action'])
                    action = re.sub(r'(peg\d+):default', r'\1:peg', action)
                    action = re.sub(r'(d\d+):default', r'\1:disc', action)
                    step['ground_action'] = action
                except Exception as e:
                    print(f"Warning: Failed to transform action '{step['ground_action']}': {e}")

        return gt_trajectory_json
