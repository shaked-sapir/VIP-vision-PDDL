"""Pre-processing normalization for benchmark experiment data.

Normalizes PDDL files before testing to ensure compatibility with planners:
    1. Hyphen → underscore in PDDL identifiers (predicates, actions, types, objects).
    2. Domain name consistency between domain and problem files.
    3. Typing validation (both :typing requirement and :types section).

Usage:
    Called automatically by the testing pipeline before fold execution.
    Can also be called standalone for debugging:

        from benchmark.experiment_running_helpers.normalize import normalize_experiment_data
        norm_trajs, norm_domain = normalize_experiment_data(data_dir, domain_file)
"""

import re
import shutil
from pathlib import Path
from typing import Tuple


# ── Pass 1: Hyphen normalization ─────────────────────────────────────────

# Match a hyphen that has a non-whitespace character on both sides,
# but only when the surrounding token does NOT start with ':'
# (to preserve PDDL keywords like :negative-preconditions).
_HYPHEN_IN_IDENTIFIER = re.compile(
    r'(?<=\S)-(?=\S)'
)

# Tokens starting with ':' that may contain hyphens — these must be preserved.
_PDDL_KEYWORD_WITH_HYPHEN = re.compile(
    r'(:[a-zA-Z][a-zA-Z0-9-]*)'
)


def _normalize_hyphens(content: str) -> str:
    """Replace hyphens with underscores in PDDL identifiers, preserving keywords.

    Strategy:
        1. Find all :keyword tokens that contain hyphens and temporarily replace
           their hyphens with a placeholder.
        2. Replace all remaining non-whitespace-surrounded hyphens with underscores.
        3. Restore the :keyword placeholders back to hyphens.
    """
    placeholder = '\x00HYPH\x00'

    # Protect :keywords by replacing their internal hyphens with placeholder
    def _protect_keyword(match: re.Match) -> str:
        return match.group(0).replace('-', placeholder)

    protected = _PDDL_KEYWORD_WITH_HYPHEN.sub(_protect_keyword, content)

    # Replace all remaining non-whitespace-surrounded hyphens
    normalized = _HYPHEN_IN_IDENTIFIER.sub('_', protected)

    # Restore keyword hyphens
    return normalized.replace(placeholder, '-')


def _normalize_file_hyphens(file_path: Path) -> None:
    """Normalize hyphens in a single file."""
    content = file_path.read_text()
    normalized = _normalize_hyphens(content)
    if normalized != content:
        file_path.write_text(normalized)


# ── Pass 2: Domain name consistency ──────────────────────────────────────

_DOMAIN_DEF_PATTERN = re.compile(
    r'\(define\s+\(domain\s+([^\s)]+)\s*\)',
    re.IGNORECASE
)

_PROBLEM_DOMAIN_PATTERN = re.compile(
    r'\(:domain\s+([^\s)]+)\s*\)',
    re.IGNORECASE
)


def _extract_domain_name(domain_file: Path) -> str:
    """Extract the domain name from (define (domain <name>) ...)."""
    content = domain_file.read_text()
    match = _DOMAIN_DEF_PATTERN.search(content)
    if not match:
        raise ValueError(
            f"Could not find (define (domain <name>)) in {domain_file}"
        )
    return match.group(1)


def _fix_problem_domain_reference(problem_file: Path, expected_domain: str) -> None:
    """Ensure the problem file's (:domain ...) matches the domain file's name."""
    content = problem_file.read_text()
    match = _PROBLEM_DOMAIN_PATTERN.search(content)
    if not match:
        raise ValueError(
            f"Could not find (:domain <name>) in problem file {problem_file}"
        )

    current_domain = match.group(1)
    if current_domain != expected_domain:
        content = content[:match.start(1)] + expected_domain + content[match.end(1):]
        problem_file.write_text(content)


# ── Pass 3: Typing validation ───────────────────────────────────────────

_REQUIREMENTS_PATTERN = re.compile(
    r'\(:requirements\s+([^)]*)\)',
    re.IGNORECASE
)

_TYPES_PATTERN = re.compile(
    r'\(:types\s+([^)]*)\)',
    re.IGNORECASE
)


def _validate_typing(domain_file: Path) -> None:
    """Validate that the domain file declares both :typing and :types.

    Raises:
        ValueError with a specific message for each failure case.
    """
    content = domain_file.read_text()

    # Check :typing in :requirements
    req_match = _REQUIREMENTS_PATTERN.search(content)
    has_typing_req = bool(req_match and ':typing' in req_match.group(1))

    # Check :types section
    types_match = _TYPES_PATTERN.search(content)
    has_types_section = bool(types_match and types_match.group(1).strip())

    if has_typing_req and has_types_section:
        return  # All good

    if not has_typing_req and has_types_section:
        raise ValueError(
            f"Domain file {domain_file.name}: "
            f"has `:types` section but `:typing` is missing from `:requirements`"
        )

    if has_typing_req and not has_types_section:
        raise ValueError(
            f"Domain file {domain_file.name}: "
            f"has `:typing` in `:requirements` but `:types` section is missing or empty"
        )

    # Both missing
    raise ValueError(
        f"Domain file {domain_file.name}: "
        f"must include `:typing` in `:requirements` and a `:types` section "
        f"for metrics calculation"
    )


# ── Main entry point ────────────────────────────────────────────────────

def normalize_experiment_data(
    data_dir: Path,
    domain_file: Path,
    force: bool = False,
) -> Tuple[Path, Path]:
    """Normalize experiment data for planner compatibility.

    Creates a normalized copy of the trajectories and domain file, then
    applies three normalization passes:
        1. Hyphen → underscore in PDDL identifiers.
        2. Domain name consistency between domain and problem files.
        3. Typing validation.

    Args:
        data_dir: Experiment data directory containing training/trajectories/.
        domain_file: Path to the reference domain PDDL file.
        force: Re-normalize even if normalized copy already exists.

    Returns:
        Tuple of (normalized_trajectories_dir, normalized_domain_file).

    Raises:
        ValueError: If typing validation fails.
        FileNotFoundError: If expected directories or files are missing.
    """
    trajectories_dir = data_dir / 'training' / 'trajectories'
    normalized_dir = data_dir / 'training' / 'trajectories_normalized'
    normalized_domain = normalized_dir / '_domain.pddl'

    if not trajectories_dir.exists():
        raise FileNotFoundError(
            f"Trajectories directory not found: {trajectories_dir}"
        )

    # Skip if already normalized (unless forced)
    if normalized_dir.exists() and not force:
        print(f"  Normalized data already exists, skipping. Use force=True to re-normalize.")
        return normalized_dir, normalized_domain

    # Clean up previous normalization if forcing
    if normalized_dir.exists() and force:
        shutil.rmtree(normalized_dir)

    # Copy trajectories
    print(f"  Copying trajectories to normalized directory...")
    shutil.copytree(trajectories_dir, normalized_dir)

    # Copy domain file
    shutil.copy(domain_file, normalized_domain)

    # ── Pass 1: Hyphen normalization ──────────────────────────────────
    print(f"  Pass 1: Normalizing hyphens in PDDL identifiers...")

    # Normalize domain file
    _normalize_file_hyphens(normalized_domain)

    # Normalize all files in problem directories
    file_patterns = ['*.trajectory', '*.pddl', '*.masking_info', '*_trajectory.json']
    for problem_dir in normalized_dir.iterdir():
        if not problem_dir.is_dir():
            continue
        for pattern in file_patterns:
            for f in problem_dir.glob(pattern):
                _normalize_file_hyphens(f)

    # ── Pass 2: Domain name consistency ───────────────────────────────
    print(f"  Pass 2: Ensuring domain name consistency...")

    domain_name = _extract_domain_name(normalized_domain)

    for problem_dir in normalized_dir.iterdir():
        if not problem_dir.is_dir():
            continue
        for pddl_file in problem_dir.glob('*.pddl'):
            # Skip domain file copies if any
            if pddl_file.name.startswith('_'):
                continue
            _fix_problem_domain_reference(pddl_file, domain_name)

    # ── Pass 3: Typing validation ─────────────────────────────────────
    print(f"  Pass 3: Validating typing declarations...")

    _validate_typing(normalized_domain)

    print(f"  Normalization complete: {normalized_dir}")
    return normalized_dir, normalized_domain
