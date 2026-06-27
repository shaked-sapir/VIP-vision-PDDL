"""
Mixin that adds conflict-driven noisy learning to any SAM-family learner.

Subclasses must implement three template methods:
  - _extract_discrete_effects(prev_preds, next_preds)
  - _extract_cannot_be_effects(prev_preds, next_preds)
  - _delegate_handle_effects(grounded_action, previous_state, next_state)
"""

from abc import abstractmethod
from collections import defaultdict
from pddl_plus_parser.models.observation import ObservedComponent
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from pddl_plus_parser.models import (
    Observation, ObservedComponent, State, ActionCall, Predicate, GroundedPredicate, Action,
)
from sam_learning.core import LearnerDomain

from src.pi_sam.noisy_pisam.typings import (
    ParameterBoundLiteral,
    ModelLevelPatch,
    FluentLevelPatch,
    PatchOperation,
    ModelPart,
    ConflictType,
    Conflict,
)
from src.utils.pddl import get_state_grounded_predicates, flip_fluent_in_state, copy_state


def _copy_state(state: State) -> State:
    """Delegate to the shared ``copy_state`` utility in ``pddl_state``."""
    return copy_state(state)


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
        # obs_idx -> component indices whose source state is GT. None = only state 0.
        self.gt_source_indices_by_obs: Optional[Dict[int, Set[int]]] = None
        # Pre-grouped fluent patches by (obs_idx, comp_idx). Built once in
        # set_patches() so that apply_fluent_patches() can iterate directly
        # without regrouping on every call — this enables the lazy deep-copy
        # optimization where only targeted components are copied.
        self._patches_by_comp: DefaultDict[Tuple[int, int], List[FluentLevelPatch]] = defaultdict(list)

    def _should_check_frame_axiom(self, obs_idx: int, comp_idx: int) -> bool:
        """True when the source state of this component is ground truth."""
        if self.gt_source_indices_by_obs is None:
            return comp_idx == 0
        return comp_idx in self.gt_source_indices_by_obs.get(obs_idx, set())

    # ------------------------------------------------------------------
    # Patch management
    # ------------------------------------------------------------------

    def set_patches(
        self,
        fluent_patches: Set[FluentLevelPatch],
        model_patches: Set[ModelLevelPatch],
        gt_source_indices_by_obs: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
        self.fluent_patches = fluent_patches
        self.model_patches = model_patches
        self.conflicts = []
        self.gt_source_indices_by_obs = gt_source_indices_by_obs

        self.forbidden_effects.clear()
        self.required_effects.clear()

        for patch in model_patches:
            if patch.model_part == ModelPart.EFFECT:
                if patch.operation == PatchOperation.FORBID:
                    self.forbidden_effects.setdefault(patch.action_name, set()).add(patch.pbl)
                else:  # REQUIRE
                    self.required_effects.setdefault(patch.action_name, set()).add(patch.pbl)

        self._patches_by_comp.clear()
        for patch in fluent_patches:
            self._patches_by_comp[(patch.observation_index, patch.component_index)].append(patch)

    # ------------------------------------------------------------------
    # Fluent-level patches
    # ------------------------------------------------------------------

    def apply_fluent_patches(self, observations: List[Observation]) -> List[Observation]:
        """Return observations with fluent flips applied, deep-copying only the flipped state(s).

        Correctness invariant: after mask_observation(), comp[i].next_state IS
        comp[i+1].previous_state. We preserve this via re-linking (co-patched) or
        lightweight ObservedComponent wrappers (unpatched neighbor).
        """
        if not self._patches_by_comp:
            return observations

        patched = {(o, c) for o, c in self._patches_by_comp
                   if 0 <= o < len(observations) and 0 <= c < len(observations[o].components)}
        if not patched:
            return observations

        result = list(observations)

        # Shallow-copy affected observations; selectively deep-copy only flipped state(s).
        for obs_idx, comp_idx in patched:
            if result[obs_idx] is observations[obs_idx]:
                obs_copy = Observation()
                obs_copy.grounded_objects = observations[obs_idx].grounded_objects
                obs_copy.components = list(observations[obs_idx].components)
                result[obs_idx] = obs_copy

            patches = self._patches_by_comp[(obs_idx, comp_idx)]
            old = observations[obs_idx].components[comp_idx]
            has_neighbor = (obs_idx, comp_idx + 1) in patched or (obs_idx, comp_idx - 1) in patched
            copy_next = has_neighbor or any(p.state_type == "next" for p in patches)
            copy_prev = has_neighbor or any(p.state_type == "prev" for p in patches)

            result[obs_idx].components[comp_idx] = ObservedComponent(
                previous_state=_copy_state(old.previous_state) if copy_prev else old.previous_state,
                call=old.grounded_action_call,
                next_state=_copy_state(old.next_state) if copy_next else old.next_state,
                is_successful=old.is_successful,
            )

        # Re-link co-patched neighbors that shared a boundary before copying (O(P)).
        for obs_idx, comp_idx in patched:
            adj = comp_idx + 1
            if (obs_idx, adj) in patched and adj < len(result[obs_idx].components):
                if observations[obs_idx].components[comp_idx].next_state is \
                   observations[obs_idx].components[adj].previous_state:
                    result[obs_idx].components[adj].previous_state = \
                        result[obs_idx].components[comp_idx].next_state

        # Apply flips, then wrap unpatched neighbors whose boundary was mutated.
        for (obs_idx, comp_idx), patches in self._patches_by_comp.items():
            if (obs_idx, comp_idx) not in patched:
                continue
            comp = result[obs_idx].components[comp_idx]
            obs, orig = result[obs_idx], observations[obs_idx]

            for patch in patches:
                target = comp.next_state if patch.state_type == "next" else comp.previous_state
                if target is not None:
                    self._flip_fluent_in_state(target, patch.fluent)

            if comp_idx + 1 < len(obs.components) and (obs_idx, comp_idx + 1) not in patched:
                if orig.components[comp_idx].next_state is orig.components[comp_idx + 1].previous_state:
                    neighbor = obs.components[comp_idx + 1]
                    obs.components[comp_idx + 1] = ObservedComponent(
                        previous_state=comp.next_state, call=neighbor.grounded_action_call,
                        next_state=neighbor.next_state, is_successful=neighbor.is_successful)
            if comp_idx - 1 >= 0 and (obs_idx, comp_idx - 1) not in patched:
                if orig.components[comp_idx].previous_state is orig.components[comp_idx - 1].next_state:
                    neighbor = obs.components[comp_idx - 1]
                    obs.components[comp_idx - 1] = ObservedComponent(
                        previous_state=neighbor.previous_state, call=neighbor.grounded_action_call,
                        next_state=comp.previous_state, is_successful=neighbor.is_successful)

        return result

    def _flip_fluent_in_state(self, state: State, fluent_str: str) -> None:
        """Flip fluent_str in the given state.

        Subclasses may override to handle missing fluents differently
        (e.g. closed-world creation).
        """
        flip_fluent_in_state(state, fluent_str)

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
        source_is_gt: bool = True,
    ) -> List[Conflict]:
        """Detect frame-axiom violations for a single transition.

        Checks only fluents that PI-SAM already identified as add/del effects,
        so violations are grounded in the same effect extraction logic used
        for the rest of learning.

        A violation arises when a fluent changes truth value but none of its
        objects appear among the action's parameters.

        Args:
            grounded_action: The grounded action at this transition.
            grounded_add_effects: Add effects extracted by PI-SAM for this transition.
            grounded_del_effects: Delete effects extracted by PI-SAM for this transition.
            source_is_gt: Whether the source (prev) state of this transition
                is ground truth. Carried on each Conflict so the search can
                decide its branching strategy (single-child for GT, two-child
                for non-GT).
        """
        action_name = grounded_action.name
        action_objs = set(grounded_action.parameters)
        local_conflicts: List[Conflict] = []

        for gp, frame_is_add in (
            [(g, True) for g in grounded_add_effects] +
            [(g, False) for g in grounded_del_effects]
        ):
            self._maybe_add_frame_conflict(
                gp, action_name, action_objs, frame_is_add=frame_is_add,
                source_is_gt=source_is_gt, local_conflicts=local_conflicts,
            )

        return local_conflicts

    def _maybe_add_frame_conflict(
        self,
        gp: GroundedPredicate,
        action_name: str,
        action_objs: Set[str],
        frame_is_add: bool,
        source_is_gt: bool,
        local_conflicts: List[Conflict],
    ) -> None:
        """Add a frame-axiom conflict if the fluent's objects are not a subset of the action's."""
        gp_objs = set(gp.object_mapping.values())
        if len(gp_objs) == 0:
            return
        if gp_objs <= action_objs:
            return

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
            source_is_gt=source_is_gt,
        )
        local_conflicts.append(conflict)

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

        # --- Must-be effects according to SAM rules within the current transition ---
        grounded_add_effects, grounded_del_effects = self._extract_discrete_effects(
            prev_preds, next_preds,
        )
        all_grounded_must = list(grounded_add_effects) + list(grounded_del_effects)

        # Frame-axiom conflicts (at every transition; source_is_gt tag drives
        # the search's branching strategy: single-child for GT, two-child for non-GT)
        source_is_gt = self._should_check_frame_axiom(
            self.current_observation_index, self.current_component_index
        )
        local_conflicts.extend(
            self._collect_frame_axiom_conflicts(
                grounded_action, grounded_add_effects, grounded_del_effects,
                source_is_gt=source_is_gt,
            )
        )

        # History BEFORE this transition
        prior_must_effects: Set[Predicate] = set(observed_action.discrete_effects)
        prior_cannot_effects: Set[Predicate] = set(
            self.cannot_be_effect.get(action_name, set())
        )

        # Model-patch sets (used to suppress data-driven conflicts)
        require_set = self.required_effects.get(action_name, set())
        forbid_set = self.forbidden_effects.get(action_name, set())

        # (2a) DATA-ONLY: prior cannot_be_effect + new must-be-effect
        for gp in all_grounded_must:
            possible_lifted = self.matcher.get_possible_literal_matches(grounded_action, [gp])

            for lifted_gp in possible_lifted:
                if lifted_gp in prior_cannot_effects:
                    pbl = ParameterBoundLiteral.from_lifted_predicate(lifted_gp)
                    if pbl not in require_set:
                        local_conflicts.append(Conflict(
                            action_name=action_name,
                            pbl=pbl,
                            conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,
                            observation_index=self.current_observation_index,
                            component_index=self.current_component_index,
                            grounded_fluent=gp.untyped_representation,
                        ))

        # (1a) PATCH-BASED: FORBID_EFFECT_VS_MUST
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

        # (2b) DATA-ONLY: prior must-be-effect + new cannot-be-effect
        for gp in cannot_be_effects:
            possible_lifted = self.matcher.get_possible_literal_matches(grounded_action, [gp])

            for lifted_gp in possible_lifted:
                if lifted_gp in prior_must_effects:
                    pbl = ParameterBoundLiteral.from_lifted_predicate(lifted_gp)
                    if pbl not in forbid_set:
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

        self.conflicts.extend(local_conflicts)
        self._delegate_handle_effects(grounded_action, previous_state, next_state)

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
        gt_source_indices_by_obs: Optional[Dict[int, Set[int]]] = None,
        **kwargs,
    ) -> Tuple[LearnerDomain, List[Conflict], Dict[str, str]]:
        self.set_patches(fluent_patches, model_patches, gt_source_indices_by_obs)
        learned_domain, learning_report = self.learn_action_model(observations, **kwargs)
        return learned_domain, self.conflicts, learning_report
