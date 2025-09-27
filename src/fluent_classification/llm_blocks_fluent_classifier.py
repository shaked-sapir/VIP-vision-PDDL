from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier


class LLMBlocksFluentClassifier(LLMFluentClassifier):
    # TODO: update the system prompt to handle all colors of blocks (adjust with RGB like in the basic classifier)
    system_prompt = (
        "You are a visual reasoning agent for a robotic planning system. "
        "Given an image, consisted of the following objects: "
        "1. gray-colored gripper (type=gripper), "
        "2. brown-colored table (type=table), "
        "3. colored blocks: red, blue, green, cyan (type=block). "
        "Extract grounded binary predicates in the following forms:\n"
        "- on(x:block, y-block) - block x is directly on block y\n"
        "- ontable(x-block): true if block x is on the table\n"
        "- handfree(gripper-gripper): true if gripper does not hold anything\n"
        "- handful(gripper-gripper): gripper holds something\n"
        "- holding(x-block, gripper-gripper): gripper holds block x\n"
        "- clear(x-block): no block is on top of x\n\n"
        "Only use defined objects. Return one predicate per line."
    )
