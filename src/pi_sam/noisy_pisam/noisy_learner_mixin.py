"""
Mixin that adds conflict-driven noisy learning to any SAM-family learner.

Subclasses must implement three template methods:
  - _extract_discrete_effects(prev_preds, next_preds)
  - _extract_cannot_be_effects(prev_preds, next_preds)
  - _delegate_handle_effects(grounded_action, previous_state, next_state)
"""

from abc import abstractmethod
from copy import deepcopy
from typing import List, Set, Dict, Tuple

from pddl_plus_parser.models import (
    Observation, State, ActionCall, Predicate, GroundedPredicate, Action,
)
from sam_learning.core import LearnerDomain

from src.action_model.pddl2gym_parser import negate_str_predicate
from src.pi_sam.noisy_pisam.typings import (
    ParameterBoundLiteral,
    ModelLevelPatch,
    FluentLevelPatch,
    PatchOperation,
    ModelPart,
    ConflictType,
    Conflict,
)
from src.utils.pddl import get_state_grounded_predicates


class NoisyLearnerMixin:
    """
    Mixin providing:
      - Fluent-level patches (flip specific grounded fluents in trajectories).
      - Model-level patches (EFFECT + FORBID/REQUIRE).
      - Conflict detection (FORBID_EFFECT_VS_MUST, REQUIRE_EFFECT_VS_CANNOT,
        FRAME_AXIOM, and data-only inconsistencies).

    Mix this in **before** the base learner so that ``handle_effects``
    resolves to this mixin's version:

        class NoisyPisamLearner(NoisyLearnerMixin, PISAMLearner): ...
    """

    # ------------------------------------------------------------------
    # Template methods — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_discrete_effects(
        self, prev_preds: Set[GroundedPredicate], next_preds: Set[GroundedPredicate],
    ) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
        """Return (grounded_add_effects, grounded_del_effects)."""
        ...

    @abstractmethod
    def _extract_cannot_be_effects(
        self, prev_preds: Set[GroundedPredicate], next_preds: Set[GroundedPredicate],
    ) -> Set[GroundedPredicate]:
        """Return the set of grounded cannot-be-effect predicates.

        SAM implementations may ignore *next_preds*.
        """
        ...

    @abstractmethod
    def _delegate_handle_effects(
        self, grounded_action: ActionCall, previous_state: State, next_state: State,
    ) -> None:
        """Called on the no-conflict path to perform the base learner's
        normal effect update."""
        ...

    # ------------------------------------------------------------------
    # Noisy-field initialisation (call from subclass __init__)
    # ------------------------------------------------------------------

    def _init_noisy_fields(self) -> None:
        self.fluent_patches: Set[FluentLevelPatch] = set()
        self.model_patches: Set[ModelLevelPatch] = set()
        self.forbidden_effects: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.required_effects: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.conflicts: List[Conflict] = []
        self.current_observation_index: int = 0
        self.current_component_index: int = 0

    # ------------------------------------------------------------------
    # Patch management
    # ------------------------------------------------------------------

    def set_patches(
        self,
        fluent_patches: Set[FluentLevelPatch],
        model_patches: Set[ModelLevelPatch],
    ) -> None:
        self.fluent_patches = fluent_patches
        self.model_patches = model_patches
        self.conflicts = []

        self.forbidden_effects.clear()
        self.required_effects.clear()

        for patch in model_patches:
            if patch.model_part == ModelPart.EFFECT:
                if patch.operation == PatchOperation.FORBID:
                    self.forbidden_effects.setdefault(patch.action_name, set()).add(patch.pbl)
                else:  # REQUIRE
                    self.required_effects.setdefault(patch.action_name, set()).add(patch.pbl)

    # ------------------------------------------------------------------
    # Fluent-level patches
    # ------------------------------------------------------------------

    def apply_fluent_patches(self, observations: List[Observation]) -> List[Observation]:
        patched_observations = deepcopy(observations)

        for patch in self.fluent_patches:
            obs_idx = patch.observation_index
            comp_idx = patch.component_index
            try:
                if not (0 <= obs_idx < len(patched_observations)):
                    self.logger.warning(f"Fluent patch with invalid observation index: {patch}")
                    continue

                obs = patched_observations[obs_idx]
                if not (0 <= comp_idx < len(obs.components)):
                    self.logger.warning(f"Fluent patch with invalid component index: {patch}")
                    continue

                comp = obs.components[comp_idx]

                if patch.state_type == "next":
                    if comp.next_state is not None:
                        self._flip_fluent_in_state(comp.next_state, patch.fluent)
                else:  # "prev"
                    if comp.previous_state is not None:
                        self._flip_fluent_in_state(comp.previous_state, patch.fluent)
            except ValueError as ve:
                self.logger.warning(f"{ve} [{obs_idx}][{comp_idx}], Full Patch: {patch}")
                continue
        return patched_observations

    def _flip_fluent_in_state(self, state: State, fluent_str: str) -> None:
        """Flip fluent_str in the given state.

        Subclasses may override to handle missing fluents differently
        (e.g. closed-world creation).
        """
        candidates = {fluent_str, negate_str_predicate(fluent_str)}

        for gp in get_state_grounded_predicates(state):
            if gp.untyped_representation not in candidates:
                continue

            base_key = (
                gp.lifted_untyped_representation
                if gp.is_positive
                else negate_str_predicate(gp.lifted_untyped_representation)
            )

            for p in state.state_predicates[base_key]:
                if p.untyped_representation in candidates:
                    p.is_positive = not p.is_positive
                    return

        raise ValueError(
            f"Could not find fluent {fluent_str} or its negation to flip in state"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lift_and_match(
        self,
        grounded_action: ActionCall,
        grounded_pred: GroundedPredicate,
        pbl: ParameterBoundLiteral,
    ) -> bool:
        lifted_candidates = self.matcher.get_possible_literal_matches(
            grounded_action, [grounded_pred]
        )
        return any(
            isinstance(candidate, Predicate) and pbl.matches(candidate)
            for candidate in lifted_candidates
        )

    def _ground_pbl_with_action(
        self, pbl: ParameterBoundLiteral, grounded_action: ActionCall,
    ) -> str:
        action_schema: Action = self.partial_domain.actions[grounded_action.name]
        signature_param_to_object: Dict[str, str] = {
            param: obj
            for param, obj in zip(action_schema.signature.keys(), grounded_action.parameters)
        }
        grounded_args = [signature_param_to_object[p_name] for p_name in pbl.parameters]
        base = f"({pbl.predicate_name} {' '.join(grounded_args)})"
        return base if pbl.is_positive else f"(not {base})"

    def _should_negate_grounded_effect(
        self,
        grounded_predicate,
        conflict_observation_index: int,
        conflict_component_index: int,
    ) -> bool:
        """Decide whether the grounded_fluent of a REQUIRE_EFFECT_VS_CANNOT
        conflict should be reported in negated form.

        Returns True (use negated form) unless a fluent patch at
        (obs, comp+1, "prev") already flips the same predicate.
        """
        gp_repr = grounded_predicate.untyped_representation
        target_comp = conflict_component_index + 1

        for patch in self.fluent_patches:
            if (
                patch.observation_index == conflict_observation_index
                and patch.component_index == target_comp
                and patch.state_type == "prev"
                and patch.fluent == gp_repr
            ):
                return False

        return True

    # ------------------------------------------------------------------
    # Frame-axiom conflict detection
    # ------------------------------------------------------------------

    def _collect_frame_axiom_conflicts(
        self,
        grounded_action: ActionCall,
        grounded_add_effects: Set[GroundedPredicate],
        grounded_del_effects: Set[GroundedPredicate],
    ) -> List[Conflict]:
        """Detect frame-axiom violations for a single transition.

        A violation arises when a fluent changes truth value but none of its
        objects appear among the action's parameters.
        """
        action_name = grounded_action.name
        action_objs = set(grounded_action.parameters)
        local_conflicts: List[Conflict] = []

        for gp, frame_is_add in (
            [(g, False) for g in grounded_del_effects] +
            [(g, True) for g in grounded_add_effects]
        ):
            gp_objs = set(gp.object_mapping.values())
            if len(gp_objs) == 0:
                continue
            if gp_objs <= action_objs:
                continue

            pbl = ParameterBoundLiteral(
                predicate_name=gp.name,
                parameters=tuple(),
                is_positive=gp.is_positive,
            )
            conflict = Conflict(
                action_name=action_name,
                pbl=pbl,
                conflict_type=ConflictType.FRAME_AXIOM,
                observation_index=self.current_observation_index,
                component_index=self.current_component_index,
                grounded_fluent=gp.untyped_representation,
                frame_is_add=frame_is_add,
            )
            local_conflicts.append(conflict)

        return local_conflicts

    # ------------------------------------------------------------------
    # Effect handling with conflict detection
    # ------------------------------------------------------------------

    def handle_effects(
        self, grounded_action: ActionCall, previous_state: State, next_state: State,
    ) -> None:
        action_name = grounded_action.name
        observed_action = self.partial_domain.actions[action_name]

        prev_preds = get_state_grounded_predicates(previous_state)
        next_preds = get_state_grounded_predicates(next_state)

        local_conflicts: List[Conflict] = []

        # --- Must-be effects ---
        grounded_add_effects, grounded_del_effects = self._extract_discrete_effects(
            prev_preds, next_preds,
        )
        all_grounded_must = list(grounded_add_effects) + list(grounded_del_effects)

        # PRED_MATCHER_DEBUGGING
        _diag = action_name == "unstack"
        if _diag:
            print(
                f"[PRED_MATCHER_DEBUGGING] === handle_effects: {action_name}({', '.join(grounded_action.parameters)}) "
                f"obs={self.current_observation_index} comp={self.current_component_index} ==="
            )
            print(
                f"[PRED_MATCHER_DEBUGGING] grounded_add_effects: "
                f"{[gp.untyped_representation for gp in grounded_add_effects]}"
            )
            print(
                f"[PRED_MATCHER_DEBUGGING] grounded_del_effects: "
                f"{[gp.untyped_representation for gp in grounded_del_effects]}"
            )
        # END PRED_MATCHER_DEBUGGING

        # Frame-axiom conflicts
        local_conflicts.extend(
            self._collect_frame_axiom_conflicts(
                grounded_action, grounded_add_effects, grounded_del_effects,
            )
        )

        # History BEFORE this transition
        prior_must_effects: Set[Predicate] = set(observed_action.discrete_effects)
        prior_cannot_effects: Set[Predicate] = set(
            self.cannot_be_effect.get(action_name, set())
        )

        # PRED_MATCHER_DEBUGGING
        if _diag:
            print(
                f"[PRED_MATCHER_DEBUGGING] prior_must_effects (discrete_effects): "
                f"{[str(p) for p in prior_must_effects]}"
            )
            print(
                f"[PRED_MATCHER_DEBUGGING] prior_cannot_effects: "
                f"{[str(p) for p in prior_cannot_effects]}"
            )
        # END PRED_MATCHER_DEBUGGING

        # (2a) DATA-ONLY: prior cannot_be_effect + new must-be-effect
        for gp in all_grounded_must:
            possible_lifted = self.matcher.get_possible_literal_matches(grounded_action, [gp])

            # PRED_MATCHER_DEBUGGING
            if _diag:
                self.logger.debug(
                    f"[PRED_MATCHER_DEBUGGING] (2a) gp={gp.untyped_representation} "
                    f"possible_lifted={[str(p) for p in possible_lifted]} "
                    f"hits={[str(p) for p in possible_lifted if p in prior_cannot_effects]}"
                )
            # END PRED_MATCHER_DEBUGGING

            for lifted_gp in possible_lifted:
                if lifted_gp in prior_cannot_effects:
                    pbl = ParameterBoundLiteral.from_lifted_predicate(lifted_gp)
                    local_conflicts.append(Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.untyped_representation,
                    ))

        # (1a) PATCH-BASED: FORBID_EFFECT_VS_MUST
        forbid_set = self.forbidden_effects.get(action_name, set())
        for gp in all_grounded_must:
            for pbl in forbid_set:
                if self._lift_and_match(grounded_action, gp, pbl):
                    local_conflicts.append(Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.untyped_representation,
                    ))

        # --- Cannot-be effects ---
        cannot_be_effects: Set[GroundedPredicate] = self._extract_cannot_be_effects(
            prev_preds, next_preds,
        )

        # PRED_MATCHER_DEBUGGING
        if _diag:
            print(
                f"[PRED_MATCHER_DEBUGGING] cannot_be_effects: "
                f"{[gp.untyped_representation for gp in cannot_be_effects]}"
            )
        # END PRED_MATCHER_DEBUGGING

        # (2b) DATA-ONLY: prior must-be-effect + new cannot-be-effect
        for gp in cannot_be_effects:
            possible_lifted = self.matcher.get_possible_literal_matches(grounded_action, [gp])

            # PRED_MATCHER_DEBUGGING
            if _diag:
                self.logger.debug(
                    f"[PRED_MATCHER_DEBUGGING] (2b) gp={gp.untyped_representation} "
                    f"possible_lifted={[str(p) for p in possible_lifted]} "
                    f"hits={[str(p) for p in possible_lifted if p in prior_must_effects]}"
                )
            # END PRED_MATCHER_DEBUGGING

            for lifted_gp in possible_lifted:
                if lifted_gp in prior_must_effects:
                    pbl = ParameterBoundLiteral.from_lifted_predicate(lifted_gp)
                    to_negate = self._should_negate_grounded_effect(
                        gp, self.current_observation_index, self.current_component_index,
                    )
                    local_conflicts.append(Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.copy(is_negated=to_negate).untyped_representation,
                    ))

        # (1b) PATCH-BASED: REQUIRE_EFFECT_VS_CANNOT
        require_set = self.required_effects.get(action_name, set())
        for gp in cannot_be_effects:
            for pbl in require_set:
                if self._lift_and_match(grounded_action, gp, pbl):
                    to_negate = self._should_negate_grounded_effect(
                        gp, self.current_observation_index, self.current_component_index,
                    )
                    local_conflicts.append(Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.copy(is_negated=to_negate).untyped_representation,
                    ))

        # PRED_MATCHER_DEBUGGING
        if _diag:
            print(
                f"[PRED_MATCHER_DEBUGGING] local_conflicts found: {len(local_conflicts)}"
            )
            for c in local_conflicts:
                print(f"[PRED_MATCHER_DEBUGGING]   conflict: {c}")  # PRED_MATCHER_DEBUGGING
        # END PRED_MATCHER_DEBUGGING

        self.conflicts.extend(local_conflicts)
        self._delegate_handle_effects(grounded_action, previous_state, next_state)

        # PRED_MATCHER_DEBUGGING
        if _diag:
            print(
                f"[PRED_MATCHER_DEBUGGING] AFTER delegate: discrete_effects for {action_name}: "
                f"{[str(p) for p in observed_action.discrete_effects]}"
            )
            print(
                f"[PRED_MATCHER_DEBUGGING] AFTER delegate: cannot_be_effect for {action_name}: "
                f"{[str(p) for p in self.cannot_be_effect.get(action_name, set())]}"
            )
            print(f"[PRED_MATCHER_DEBUGGING] === END {action_name} ===")  # PRED_MATCHER_DEBUGGING
        # END PRED_MATCHER_DEBUGGING

    # ------------------------------------------------------------------
    # Learning loop with index tracking
    # ------------------------------------------------------------------

    def learn_action_model(
        self,
        observations: List[Observation],
        **kwargs,
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        # Defensive contract check: this mixin expects concrete learners
        # to implement per-component handling.
        # Accept concrete implementations inherited from any base class
        # (e.g. PISAMLearner/SAMLearner), not only direct overrides.
        handler = getattr(self, "handle_single_trajectory_component", None)
        if handler is None or not callable(handler):
            raise TypeError(
                f"{self.__class__.__name__} must override callable "
                f"handle_single_trajectory_component(component)"
            )

        self.logger.info("Starting noisy learner with conflict detection.")

        self.start_measure_learning_time()
        self.deduce_initial_inequality_preconditions()
        self._complete_possibly_missing_actions()

        patched_observations = self.apply_fluent_patches(observations)

        for obs_idx, observation in enumerate(patched_observations):
            self.current_observation_index = obs_idx
            self.current_trajectory_objects = observation.grounded_objects

            for comp_idx, component in enumerate(observation.components):
                self.current_component_index = comp_idx

                if not component.is_successful:
                    self.logger.warning("Skipping transition because it was not successful.")
                    continue

                self.handle_single_trajectory_component(component)

        self.construct_safe_actions()
        self._remove_unobserved_actions_from_partial_domain()
        self.handle_negative_preconditions_policy()
        self.end_measure_learning_time()
        report = self._construct_learning_report()

        return self.partial_domain, report

    def learn_action_model_with_conflicts(
        self,
        observations: List[Observation],
        fluent_patches: Set[FluentLevelPatch],
        model_patches: Set[ModelLevelPatch],
        **kwargs,
    ) -> Tuple[LearnerDomain, List[Conflict], Dict[str, str]]:
        self.set_patches(fluent_patches, model_patches)
        learned_domain, learning_report = self.learn_action_model(observations, **kwargs)
        return learned_domain, self.conflicts, learning_report
