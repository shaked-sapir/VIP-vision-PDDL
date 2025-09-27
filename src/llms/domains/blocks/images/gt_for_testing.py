import json
from collections import defaultdict
from pathlib import Path

from src.llms.domains.blocks.consts import objects_to_names
from src.llms.domains.blocks.prompts import object_detection_system_prompt
from src.llms.facts_extraction import extract_facts_once
from src.llms.utils import state_key_to_index, object_detection_regex

gt = defaultdict(dict)

image_dir = Path(__file__).parent
image_filenames = sorted(list(image_dir.glob("*.png")))

state_keys = [image_filename.stem for image_filename in image_filenames]


def create_gt_for_testing(block_names: list[str], gripper_name: str) -> dict:
    with open(image_dir / "problem1_trajectory.json", 'r') as file:
        data = json.load(file)
        for state_key in state_keys:
            state_index = state_key_to_index(state_key)
            transition = next(s for s in data if s["step"] == state_index) if state_index != 0 \
                else next(s for s in data if s["step"] == 1)
            if state_index == 0:
                state_literals = transition["current_state"]["literals"]
            else:
                state_literals = transition["next_state"]["literals"]

            for literal in state_literals:
                gt[state_key][literal] = 1

            for block_name in block_names:
                if f"clear({block_name})" not in gt[state_key]:
                    gt[state_key][f"clear({block_name})"] = 0
                if f"ontable({block_name})" not in gt[state_key]:
                    gt[state_key][f"ontable({block_name})"] = 0
                if f"holding({block_name})" not in gt[state_key]:
                    gt[state_key][f"holding({block_name})"] = 0
            for block_name in block_names:
                for block_name2 in block_names:
                    if block_name != block_name2:
                        if f"on({block_name},{block_name2})" not in gt[state_key]:
                            gt[state_key][f"on({block_name},{block_name2})"] = 0
            if f"handempty({gripper_name})" not in gt[state_key]:
                gt[state_key][f"handempty({gripper_name})"] = 0
            if f"handfull({gripper_name})" not in gt[state_key]:
                gt[state_key][f"handfull({gripper_name})"] = 0

    # save to file for sanity checks
    with open("ground_truth.json", 'w') as file:
        json.dump(gt, file, indent=4)

    return gt
