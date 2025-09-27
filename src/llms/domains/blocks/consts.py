from typing import Dict, List

objects_to_colors = {
    "block": ["red", "green", "blue", "cyan", "yellow", "pink"],
    "gripper": ["gray"],
    "table": ["brown"]
}

objects_to_names: dict[str, str | list[str]] = {
    "block": [f"{color}:block" for color in objects_to_colors["block"]],
    "gripper": "gripper:gripper",
    "table": "table:table",
}

all_colors: list[str] = objects_to_colors["block"] + objects_to_colors["gripper"] + objects_to_colors["table"]
all_object_types: list[str] = list(objects_to_colors.keys())
