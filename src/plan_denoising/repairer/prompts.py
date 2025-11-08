"""
Verification prompts for LLM-based fluent verification.

This module contains different prompt templates for verifying which image(s)
satisfy a given fluent predicate in PDDL planning domains.
"""


def create_binary_verification_prompt(fluent: str, domain_name: str) -> str:
    """
    Create a binary verification prompt for fluent presence.

    This prompt forces the LLM to choose exactly one of two images as satisfying
    the fluent. It's designed for situations where we know that exactly one image
    should satisfy the fluent (mutually exclusive outcomes).

    Args:
        fluent: PDDL fluent to verify (e.g., "(on red blue)")
        domain_name: Name of the planning domain (for context)

    Returns:
        System prompt string for the LLM

    Expected responses:
        - "IMAGE1" if only the first image satisfies the fluent
        - "IMAGE2" if only the second image satisfies the fluent
    """
    return f"""You are a visual reasoning expert for PDDL planning domains.

You will be shown TWO images from the {domain_name} domain.

Your task: Determine which image satisfies the following fluent (predicate):

**Fluent to verify**: {fluent}

IMPORTANT: Exactly ONE of these images satisfies this fluent. Your job is to determine which one.

Analyze each image carefully and determine which image correctly represents the fluent.

Respond with EXACTLY one of the following:
- "IMAGE1" if the first image satisfies the fluent
- "IMAGE2" if the second image satisfies the fluent

Be precise and base your answer only on clear visual evidence. You must choose one of the two images."""


def create_ternary_verification_prompt(fluent: str, domain_name: str) -> str:
    """
    Create a ternary verification prompt for fluent presence.

    This prompt allows the LLM to choose between three options: first image,
    second image, or uncertain. The "uncertain" option provides an escape hatch
    when visual evidence is ambiguous or unclear.

    Args:
        fluent: PDDL fluent to verify (e.g., "(on red blue)")
        domain_name: Name of the planning domain (for context)

    Returns:
        System prompt string for the LLM

    Expected responses:
        - "IMAGE1" if only the first image satisfies the fluent
        - "IMAGE2" if only the second image satisfies the fluent
        - "UNCERTAIN" if it's unclear which image satisfies the fluent
    """
    return f"""You are a visual reasoning expert for PDDL planning domains.

You will be shown TWO images from the {domain_name} domain.

Your task: Determine which image (if any) satisfies the following fluent (predicate):

**Fluent to verify**: {fluent}

Analyze each image carefully and determine:
1. Does IMAGE 1 satisfy this fluent?
2. Does IMAGE 2 satisfy this fluent?

Respond with EXACTLY one of the following:
- "IMAGE1" if only the first image clearly satisfies the fluent
- "IMAGE2" if only the second image clearly satisfies the fluent
- "UNCERTAIN" if it's unclear or ambiguous which image satisfies the fluent

Be precise and base your answer only on clear visual evidence. If the visual evidence
is not clear enough to make a confident determination, respond with "UNCERTAIN"."""


# TODO Later: this may be redundant, remove if not needed
def create_quaternary_verification_prompt(fluent: str, domain_name: str) -> str:
    """
    Create a quaternary verification prompt for fluent presence (original version).

    This is the original prompt that allows for four possible outcomes:
    - Only first image satisfies the fluent
    - Only second image satisfies the fluent
    - Both images satisfy the fluent
    - Neither image satisfies the fluent

    Args:
        fluent: PDDL fluent to verify (e.g., "(on red blue)")
        domain_name: Name of the planning domain (for context)

    Returns:
        System prompt string for the LLM

    Expected responses:
        - "IMAGE1" if only the first image satisfies the fluent
        - "IMAGE2" if only the second image satisfies the fluent
        - "BOTH" if both images satisfy the fluent
        - "NEITHER" if neither image satisfies the fluent
    """
    return f"""You are a visual reasoning expert for PDDL planning domains.

You will be shown TWO images from the {domain_name} domain.

Your task: Determine which image (if any) satisfies the following fluent (predicate):

**Fluent to verify**: {fluent}

Analyze each image carefully and determine:
1. Does IMAGE 1 satisfy this fluent?
2. Does IMAGE 2 satisfy this fluent?

Respond with EXACTLY one of the following:
- "IMAGE1" if only the first image satisfies the fluent
- "IMAGE2" if only the second image satisfies the fluent
- "BOTH" if both images satisfy the fluent
- "NEITHER" if neither image satisfies the fluent

Be precise and base your answer only on clear visual evidence."""
