"""
Deterministic fluent classifier for Hanoi domain.

Classifies predicates based on object positions and sizes detected in images.
"""

from typing import Dict, Tuple
import cv2
import numpy as np

from src.fluent_classification.base_fluent_classifier import FluentClassifier, PredicateTruthValue
from src.object_detection.hanoi_object_detector import HanoiObjectDetector
from src.typings import ObjectLabel


class HanoiFluentClassifier(FluentClassifier):
    """
    Deterministic fluent classifier for the Hanoi domain.

    Determines truth values for predicates:
    - on(x, y): disc/peg x is on disc/peg y
    - clear(x): disc/peg x has nothing on top of it
    - smaller(x, y): disc y is smaller than disc/peg x
    """

    def __init__(self, object_detector: HanoiObjectDetector):
        """
        Initialize the Hanoi fluent classifier.

        :param object_detector: HanoiObjectDetector instance for detecting objects
        """
        self.object_detector = object_detector

        # Disc size ordering (smaller number = smaller disc)
        self.disc_sizes = {
            "d1": 1,  # Smallest
            "d2": 2,
            "d3": 3,
            "d4": 4,
            "d5": 5,
            "d6": 6   # Largest
        }

        # Vertical threshold for "on" relationship (in pixels)
        self.vertical_threshold = 5  # Discs/pegs must be vertically aligned within this threshold
        self.horizontal_threshold = 50  # Horizontal alignment threshold

    def classify(self, image_path: str) -> Dict[str, PredicateTruthValue]:
        """
        Classify all predicates in the Hanoi domain for the given image.

        :param image_path: Path to the image file
        :return: Dictionary mapping predicate strings to truth values
        """
        # Detect objects in the image
        objects = self.object_detector.detect_objects(image_path)

        # Separate discs and pegs
        discs = {label: pos for label, pos in objects.items() if ":disc" in str(label)}
        pegs = {label: pos for label, pos in objects.items() if ":peg" in str(label)}

        # Extract disc names (d1, d2, etc.) and positions
        disc_items = [(str(label).split(":")[0], pos) for label, pos in discs.items()]
        peg_items = [(str(label).split(":")[0], pos) for label, pos in pegs.items()]

        # Sort by vertical position (y-coordinate, higher y = lower in image)
        disc_items.sort(key=lambda item: item[1][1])  # Sort by y-coordinate

        # Initialize predicate dictionary
        predicates = {}

        # Classify 'on' predicates
        predicates.update(self._classify_on_predicates(disc_items, peg_items))

        # Classify 'clear' predicates
        predicates.update(self._classify_clear_predicates(disc_items, peg_items, predicates))

        # Classify 'smaller' predicates
        predicates.update(self._classify_smaller_predicates(disc_items, peg_items))

        return predicates

    def _classify_on_predicates(
        self,
        disc_items: list[Tuple[str, Tuple[int, int]]],
        peg_items: list[Tuple[str, Tuple[int, int]]]
    ) -> Dict[str, PredicateTruthValue]:
        """
        Classify 'on' predicates.

        A disc is 'on' another disc/peg if:
        1. They are horizontally aligned (similar x-coordinates)
        2. The first disc is directly above the second (lower y-coordinate, but close)

        :param disc_items: List of (disc_name, (x, y)) tuples
        :param peg_items: List of (peg_name, (x, y)) tuples
        :return: Dictionary of 'on' predicates and their truth values
        """
        predicates = {}

        # Check disc-on-disc relationships
        for i, (disc1, (x1, y1)) in enumerate(disc_items):
            for disc2, (x2, y2) in disc_items:
                if disc1 == disc2:
                    continue

                # Check if disc1 is directly above disc2
                horizontal_aligned = abs(x1 - x2) < self.horizontal_threshold
                vertically_adjacent = (y2 > y1) and (y2 - y1 < 100)  # disc2 below disc1

                if horizontal_aligned and vertically_adjacent:
                    # Check if there's no disc in between
                    is_directly_on = True
                    for disc3, (x3, y3) in disc_items:
                        if disc3 in [disc1, disc2]:
                            continue
                        if abs(x3 - x1) < self.horizontal_threshold and y1 < y3 < y2:
                            is_directly_on = False
                            break

                    if is_directly_on:
                        predicates[f"on({disc1}:disc,{disc2}:disc)"] = PredicateTruthValue.TRUE
                    else:
                        predicates[f"on({disc1}:disc,{disc2}:disc)"] = PredicateTruthValue.FALSE
                else:
                    predicates[f"on({disc1}:disc,{disc2}:disc)"] = PredicateTruthValue.FALSE

        # Check disc-on-peg relationships
        for disc, (dx, dy) in disc_items:
            for peg, (px, py) in peg_items:
                # Check if disc is at the bottom of this peg
                horizontal_aligned = abs(dx - px) < self.horizontal_threshold

                if horizontal_aligned:
                    # Check if disc is the lowest disc on this peg
                    is_lowest = True
                    for other_disc, (ox, oy) in disc_items:
                        if other_disc == disc:
                            continue
                        if abs(ox - px) < self.horizontal_threshold and oy > dy:
                            is_lowest = False
                            break

                    if is_lowest:
                        predicates[f"on({disc}:disc,{peg}:peg)"] = PredicateTruthValue.TRUE
                    else:
                        predicates[f"on({disc}:disc,{peg}:peg)"] = PredicateTruthValue.FALSE
                else:
                    predicates[f"on({disc}:disc,{peg}:peg)"] = PredicateTruthValue.FALSE

        return predicates

    def _classify_clear_predicates(
        self,
        disc_items: list[Tuple[str, Tuple[int, int]]],
        peg_items: list[Tuple[str, Tuple[int, int]]],
        on_predicates: Dict[str, PredicateTruthValue]
    ) -> Dict[str, PredicateTruthValue]:
        """
        Classify 'clear' predicates.

        A disc/peg is 'clear' if no disc is on top of it.

        :param disc_items: List of (disc_name, (x, y)) tuples
        :param peg_items: List of (peg_name, (x, y)) tuples
        :param on_predicates: Already classified 'on' predicates
        :return: Dictionary of 'clear' predicates and their truth values
        """
        predicates = {}

        # Check if discs are clear
        for disc, pos in disc_items:
            is_clear = True
            for other_disc, _ in disc_items:
                if other_disc == disc:
                    continue
                # Check if other_disc is on this disc
                on_pred = f"on({other_disc}:disc,{disc}:disc)"
                if on_predicates.get(on_pred) == PredicateTruthValue.TRUE:
                    is_clear = False
                    break

            predicates[f"clear({disc}:disc)"] = (
                PredicateTruthValue.TRUE if is_clear else PredicateTruthValue.FALSE
            )

        # Check if pegs are clear
        for peg, pos in peg_items:
            is_clear = True
            for disc, _ in disc_items:
                # Check if any disc is on this peg
                on_pred = f"on({disc}:disc,{peg}:peg)"
                if on_predicates.get(on_pred) == PredicateTruthValue.TRUE:
                    is_clear = False
                    break

            predicates[f"clear({peg}:peg)"] = (
                PredicateTruthValue.TRUE if is_clear else PredicateTruthValue.FALSE
            )

        return predicates

    def _classify_smaller_predicates(
        self,
        disc_items: list[Tuple[str, Tuple[int, int]]],
        peg_items: list[Tuple[str, Tuple[int, int]]]
    ) -> Dict[str, PredicateTruthValue]:
        """
        Classify 'smaller' predicates.

        These are static predicates based on disc sizes:
        - smaller(x:disc, y:disc): disc y is smaller than disc x
        - smaller(x:peg, y:disc): always true (all discs are smaller than pegs)

        :param disc_items: List of (disc_name, (x, y)) tuples
        :param peg_items: List of (peg_name, (x, y)) tuples
        :return: Dictionary of 'smaller' predicates and their truth values
        """
        predicates = {}

        # Disc-disc smaller relationships
        for disc1, _ in disc_items:
            for disc2, _ in disc_items:
                if disc1 == disc2:
                    continue

                # disc2 is smaller than disc1 if disc2's size number < disc1's size number
                size1 = self.disc_sizes.get(disc1, 999)
                size2 = self.disc_sizes.get(disc2, 999)

                if size2 < size1:
                    predicates[f"smaller({disc1}:disc,{disc2}:disc)"] = PredicateTruthValue.TRUE
                else:
                    predicates[f"smaller({disc1}:disc,{disc2}:disc)"] = PredicateTruthValue.FALSE

        # Peg-disc smaller relationships (always true)
        for peg, _ in peg_items:
            for disc, _ in disc_items:
                predicates[f"smaller({peg}:peg,{disc}:disc)"] = PredicateTruthValue.TRUE

        return predicates
