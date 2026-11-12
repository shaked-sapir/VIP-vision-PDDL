"""The ``gt_setup`` option of the baseline runners that fix endpoint states.

A runner built without a setup anchors as it did before setups existed; one
built with a setup fixes exactly the endpoint states the setup gives.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from src.utils.gt_setup import GtSetup

GtSetupOption = Union[str, GtSetup, None]


def runner_gt_setup(value: GtSetupOption) -> Optional[GtSetup]:
    """A runner's ``gt_setup`` argument as a setup; ``None`` stays ``None``.

    Raises:
        ValueError: on an unknown setup value.
    """
    return None if value is None else GtSetup.parse(value)


def reject_explicit_switches(gt_setup: Optional[GtSetup], **switches: object) -> None:
    """Refuse a setup combined with a switch the setup already decides.

    Args:
        gt_setup: The runner's setup, or ``None``.
        **switches: Each explicit switch by name; ``None`` means not given.

    Raises:
        ValueError: if a setup is given together with any explicit switch.
    """
    given = sorted(name for name, value in switches.items() if value is not None)
    if gt_setup is not None and given:
        raise ValueError(
            f"gt_setup '{gt_setup.value}' decides {given}; "
            f"pass the setup or the explicit switch(es), not both"
        )


def only_init_final(gt_setup: Optional[GtSetup], arm: str) -> Tuple[bool, str]:
    """``(supported, reason)`` for an arm that needs both endpoint states.

    No setup counts as supported: the arm then runs as it did before setups.
    """
    if gt_setup is None or gt_setup is GtSetup.INIT_FINAL:
        return True, ""
    return False, (
        f"{arm} takes both endpoint states as network inputs, so it runs only "
        f"under gt_setup '{GtSetup.INIT_FINAL.value}', not '{gt_setup.value}'"
    )
