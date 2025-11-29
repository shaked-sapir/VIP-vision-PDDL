"""
Blocks domain configuration for benchmark experiments.

This version uses the equalized domain (no handfull predicate) to match ROSAME's definition.
"""

from pathlib import Path

DOMAIN_FILE = Path(__file__).parent / "blocks_no_handfull.pddl"
DOMAIN_NAME = "blocks"
