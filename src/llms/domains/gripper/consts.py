"""Constants for the Gripper domain LLM integration."""

from typing import Dict, List

# Gripper objects are identified by text labels, not colors.
# These mappings are kept for structural consistency with other domains.
objects_to_colors: Dict[str, List[str]] = {
    "room": [],
    "ball": [],
    "gripper": [],
}

objects_to_names: dict[str, str | list[str]] = {
    "room": ["rooma:room", "roomb:room"],
    "ball": ["ball1:ball", "ball2:ball"],
    "gripper": ["left:gripper", "right:gripper"],
}

all_object_types: list[str] = list(objects_to_colors.keys())
