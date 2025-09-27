import itertools
import re
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.fluent_classification.base_fluent_classifier import PredicateTruthValue


class LLMBlocksFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the blocks domain.
    Uses GPT-4 Vision to extract predicates from images of blocks world scenarios.
    """
    
    def __init__(self, openai_apikey: str):
        system_prompt = self._get_system_prompt()
        result_regex = self._get_result_regex()
        result_parse_func = self._parse_predicate
        
        super().__init__(
            openai_apikey=openai_apikey,
            system_prompt_text=system_prompt,
            result_regex=result_regex,
            result_parse_func=result_parse_func
        )
    
    @staticmethod
    def _get_system_prompt() -> str:
        """Returns the system prompt for the blocks domain."""
        return (
            "You are a visual reasoning agent for a robotic planning system in a blocks world domain. "
            "Given an image, identify the following objects:\n"
            "1. Gray-colored robot/gripper (type=robot)\n"
            "2. Brown-colored table (type=table)\n"
            "3. Colored blocks: red, blue, green, cyan, yellow, pink (type=block)\n\n"
            "Extract grounded predicates in the following forms:\n"
            "- on(block1,block2) - block1 is directly on top of block2\n"
            "- ontable(block) - block is placed on the table\n"
            "- clear(block) - no other block is on top of this block\n"
            "- handempty(robot) - robot is not holding any block\n"
            "- handfull(robot) - robot is holding a block\n"
            "- holding(block) - robot is holding this specific block\n\n"
            "For each predicate, also provide a confidence score from 0.0 to 1.0 indicating how certain you are about the predicate being true.\n"
            "Format: predicate_name confidence_score\n"
            "Example: on(red,blue) 0.9\n"
            "Return one predicate per line."
        )
    
    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract predicates from LLM response."""
        # Pattern to match predicate with confidence score
        # Examples: "on(red,blue) 0.9", "ontable(green) 0.8", "clear(cyan) 0.7"
        return r'([a-zA-Z_]+\([a-zA-Z0-9_,]+\))\s+([0-9]*\.?[0-9]+)'
    
    @staticmethod
    def _parse_predicate(match_result) -> tuple:
        """
        Parses a regex match result to extract predicate and confidence score.
        
        Args:
            match_result: Result from regex.findall() containing (predicate, confidence)
            
        Returns:
            Tuple of (predicate_string, confidence_float)
        """
        predicate_str, confidence_str = match_result
        confidence = float(confidence_str)
        return (predicate_str, confidence)
    
    def _generate_all_possible_predicates(self, objects_by_type: dict[str, list[str]]) -> set[str]:
        """
        Generates all possible predicates for the blocks domain.
        
        Args:
            objects_by_type: Dictionary mapping object types to lists of grounded object names.
                            Expected format: {'block': ['a', 'b', 'c', 'd'], 'robot': ['robot']}
            
        Returns:
            Set of all possible predicate strings
        """
        # Extract objects by type with defaults
        blocks = objects_by_type.get('block', ['a', 'b', 'c', 'd'])  # Default blocks from problem1.pddl
        robots = objects_by_type.get('robot', ['robot'])  # Default robot name
        
        # Use the first robot if multiple are provided
        robot_name = robots[0] if robots else 'robot'
        
        predicates = set()
        
        # on(block1, block2) predicates
        for block1, block2 in itertools.permutations(blocks, 2):
            predicates.add(f"on({block1},{block2})")
        
        # ontable(block) predicates
        for block in blocks:
            predicates.add(f"ontable({block})")
        
        # clear(block) predicates
        for block in blocks:
            predicates.add(f"clear({block})")
        
        # handempty(robot) predicate
        predicates.add(f"handempty({robot_name})")
        
        # handfull(robot) predicate
        predicates.add(f"handfull({robot_name})")
        
        # holding(block) predicates
        for block in blocks:
            predicates.add(f"holding({block})")
        
        return predicates
    
