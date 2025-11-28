from typing import Dict, List

# Define colors for Hanoi objects
# Discs will have different colors to distinguish them by size
# Smaller discs get lighter/brighter colors
objects_to_colors = {
    "disc": ["red"],  # d1 (smallest) to d6 (largest)
    "peg": ["gray"]
}

# Object names with types
# For a 3-disc problem: d1 (smallest), d2 (medium), d3 (largest)
# Pegs are named peg1, peg2, peg3
objects_to_names: dict[str, str | list[str]] = {
    "disc": [f"d{i}:disc" for i in range(1, 7)],  # Support up to 6 discs
    "peg": [f"peg{i}:peg" for i in range(1, 4)],  # Support 3 pegs
}

# Disc to color mapping (by disc number)
# Smaller discs (d1) have lighter colors
disc_to_color: dict[str, str] = {
    "d1": "red",     # Smallest disc
    "d2": "orange",  #
    "d3": "yellow",  #
    "d4": "green",   #
    "d5": "blue",    #
    "d6": "purple"   # Largest disc
}

# Peg color mapping
peg_color = "gray"

all_colors: list[str] = objects_to_colors["disc"] + objects_to_colors["peg"]
all_object_types: list[str] = list(objects_to_colors.keys())
