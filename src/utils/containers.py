import re
from collections import defaultdict
from typing import Any, List, Dict, Union


def to_list(obj: Any) -> List[Any]:
    return [obj] if not isinstance(obj, list) else obj


def serialize(obj) -> Union[dict, list]:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj  # Base case: primitive typings
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]  # Serialize lists
    elif isinstance(obj, dict):
        return {key: serialize(value) for key, value in obj.items()}  # Serialize dictionaries
    elif hasattr(obj, "__dict__"):  # If it’s an object with attributes
        return {key: serialize(value) for key, value in obj.__dict__.items()}
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def group_objects_by_key(objects: List[Any], key: str) -> Dict[str, List[Any]]:
    grouped = defaultdict(list)
    for obj in objects:
        grouped[getattr(obj, key)].append(obj)

    return dict(grouped)


def shrink_whitespaces(s: str) -> str:
    """Shrink all multiple spaces into a single space"""
    return re.sub(r'\s+', ' ', s).strip()
