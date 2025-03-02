import os
import time
from pathlib import Path


def get_project_root_dir() -> Path:
    # Traverse upwards to find the project root
    project_root = Path(__file__).resolve().parent
    while project_root.name != "VIP-vision-PDDL":  # Replace with your project root folder name
        project_root = project_root.parent
    return project_root


def create_dir_from_root(dirpath: str) -> Path:
    project_root = get_project_root_dir()
    new_dir_path = project_root / dirpath
    # new_dir_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(new_dir_path, exist_ok=True)
    if os.name != "nt":
        os.sync()
    dir_fd = os.open(new_dir_path, os.O_RDONLY)  # Open directory descriptor
    while not new_dir_path.exists():
        print(f"⏳ Waiting for {new_dir_path} to be recognized...")
        time.sleep(0.1)  # Small delay to allow OS updates
    return new_dir_path
