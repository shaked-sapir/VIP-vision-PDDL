"""Constants for the Depot domain LLM integration."""

from typing import Dict, List

# Depot objects are identified by text labels, not colors.
# These mappings are kept for structural consistency with other domains.
objects_to_colors: Dict[str, List[str]] = {
    "depot": [],
    "truck": [],
    "crane": [],
    "pile": [],
    "package": [],
}

objects_to_names: dict[str, str | list[str]] = {
    "depot": ["d1:depot", "d2:depot"],
    "truck": ["t1:truck"],
    "crane": ["c1:crane", "c2:crane"],
    "pile": ["pile1:pile", "pile2:pile"],
    "package": ["p1:package", "p2:package", "p3:package"],
}

all_object_types: list[str] = list(objects_to_colors.keys())
