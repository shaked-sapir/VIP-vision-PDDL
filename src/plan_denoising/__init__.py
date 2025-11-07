"""Plan denoising module for fixing inconsistencies in vision-based trajectories."""

from .inconsistency_detector import InconsistencyDetector
from .detectors.base_detector import Transition
from .detectors.frame_axiom_detector import FrameAxiomDetector, FrameAxiomViolation
from .detectors.effects_detector import EffectsDetector, EffectsViolation
from .transition_extractor import TransitionExtractor
from .conflict_tree import ConflictTree, ConflictNode, RepairOperation
from .data_repairer import DataRepairer
from .plan_denoiser import PlanDenoiser

# Backward compatibility: alias DeterminismViolation as Inconsistency
Inconsistency = EffectsViolation

# Set the backward compatibility alias in conflict_tree module
from . import conflict_tree
conflict_tree.Inconsistency = EffectsViolation

__all__ = [
    # Main coordinator
    'InconsistencyDetector',

    # Detectors
    'FrameAxiomDetector',
    'EffectsDetector',

    # Violation types
    'FrameAxiomViolation',
    'EffectsViolation',
    'Inconsistency',  # Backward compatibility alias

    # Utilities
    'Transition',
    'TransitionExtractor',

    # Repair system
    'DataRepairer',
    'ConflictTree',
    'ConflictNode',
    'RepairOperation',
    'PlanDenoiser'
]
