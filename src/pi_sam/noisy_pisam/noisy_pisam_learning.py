"""
DenoisePisamLearner: A specialized PISAM learner for conflict-based learning.

This learner extends PISAMLearner to support:
1. Fluent-level patches (flipping fluent values in observations)
2. Model-level patches (constraining what can be learned)
3. Conflict detection during learning

Key design: Reuses SAMLearner/PISAMLearner methods instead of reimplementing them.
Overrides only the specific methods needed for conflict detection.
"""

from copy import deepcopy
from typing import Dict, List, Tuple, Set

from pddl_plus_parser.models import Domain, Observation, State, ActionCall, GroundedPredicate, Predicate
from sam_learning.core import (
    LearnerDomain, extract_discrete_effects_partial_observability, extract_not_effects_partial_observability
)
from utilities import NegativePreconditionPolicy

from src.action_model.gym2SAM_parser import parse_grounded_predicate
from src.pi_sam.noisy_pisam.typings import (
    ConflictType, ModelPart, PatchOperation,
    ParameterBoundLiteral, FluentLevelPatch, ModelLevelPatch, Conflict
)
from src.pi_sam.pi_sam_learning import PISAMLearner
from src.utils.pddl import get_state_grounded_predicates, get_state_unmasked_predicates, get_state_masked_predicates


class NoisyPisamLearner(PISAMLearner):
    """
    A PISAM learner that supports patches and conflict detection.

    Key design: In each precondition/effect update, check the updates against the required/forbidden
    preconditions and effectss specified in the model-level patches, in order to detect conflicts.
    """

    def __init__(
        self,
        partial_domain: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        seed: int = 42
    ):
        super().__init__(partial_domain, negative_preconditions_policy, seed)

        # Patches and conflicts
        self.fluent_patches: Set[FluentLevelPatch] = set()
        self.model_patches: Set[ModelLevelPatch] = set()
        self.conflicts: List[Conflict] = []

        # Track current position for conflict reporting
        self.current_observation_index: int = 0
        self.current_component_index: int = 0

        # Index model patches for quick lookup
        self.forbidden_preconditions: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.required_preconditions: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.forbidden_effects: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.required_effects: Dict[str, Set[ParameterBoundLiteral]] = {}

    def set_patches(
        self,
        fluent_patches: Set[FluentLevelPatch],
        model_patches: Set[ModelLevelPatch]
    ):
        """
        Set the patches to apply during learning.

        :param fluent_patches: Fluent-level patches to apply to observations
        :param model_patches: Model-level patches to constrain learning
        """
        self.fluent_patches = fluent_patches
        self.model_patches = model_patches
        self.conflicts = []

        # Build index of model patches for quick lookup
        self.forbidden_preconditions = {}
        self.required_preconditions = {}
        self.forbidden_effects = {}
        self.required_effects = {}

        section_map = {
            ModelPart.PRECONDITION: {
                PatchOperation.FORBID: self.forbidden_preconditions,
                PatchOperation.REQUIRE: self.required_preconditions
            },
            ModelPart.EFFECT: {
                PatchOperation.FORBID: self.forbidden_effects,
                PatchOperation.REQUIRE: self.required_effects
            }
        }

        for patch in model_patches:
            d = section_map[patch.model_part][patch.operation]
            d.setdefault(patch.action_name, set()).add(patch.pbl)
    # ==================== Conflict Detection Helper Methods ====================
    # ---------- Fluent patches ----------

    def apply_fluent_patches(self, observations: List[Observation]) -> List[Observation]:
        """
        Apply fluent-level patches to observations.

        Creates a deep copy and flips the specified fluents.
        Maintains state consistency by updating adjacent component states:
        - If patching next_state, also update following component's previous_state
        - If patching previous_state, also update previous component's next_state

        :param observations: Original observations
        :return: Patched observations (new copies)
        """
        patched_obs = deepcopy(observations)

        for patch in self.fluent_patches:
            obs_idx = patch.observation_index
            comp_idx = patch.component_index

            obs = patched_obs[obs_idx]
            component = obs.components[comp_idx]

            patch_grounded_predicate = parse_grounded_predicate(patch.fluent, self.partial_domain)

            if patch.state_type == "next":
                # Flip in this component's next_state
                if component.next_state is not None:
                    self._flip_fluent_in_state(component.next_state, patch_grounded_predicate)

                # Also flip in the following component's previous_state (if exists)
                if comp_idx + 1 < len(obs.components):
                    next_comp = obs.components[comp_idx + 1]
                    if next_comp.previous_state is not None:
                        self._flip_fluent_in_state(next_comp.previous_state, patch_grounded_predicate)

            else:  # "prev"
                # Flip in this component's previous_state
                if component.previous_state is not None:
                    self._flip_fluent_in_state(component.previous_state, patch_grounded_predicate)

                # Also flip in the previous component's next_state (if exists)
                if comp_idx - 1 >= 0:
                    prev_comp = obs.components[comp_idx - 1]
                    if prev_comp.next_state is not None:
                        self._flip_fluent_in_state(prev_comp.next_state, patch_grounded_predicate)

        return patched_obs

    @staticmethod
    def _flip_fluent_in_state(state: State, fluent: GroundedPredicate) -> None:
        """
        Flip a fluent in a state (add if absent, remove if present).

        :param state: The state to modify
        :param fluent: The fluent to flip
        """

        if fluent in state.state_predicates[fluent.lifted_untyped_representation]:
            # Fluent is present - remove it
            state.state_predicates[fluent.lifted_untyped_representation].remove(fluent)
        else:
            # Fluent is absent - add it
            state.state_predicates[fluent.lifted_untyped_representation].add(fluent)

    # ---------- Model patch helpers ----------
    def _get_forbidden_set_and_type(
            self, action_name: str, model_part: ModelPart
    ) -> Tuple[Set[ParameterBoundLiteral], ConflictType]:
        if model_part == ModelPart.PRECONDITION:
            return (
                self.forbidden_preconditions.get(action_name, set()),
                ConflictType.FORBIDDEN_PRECONDITION,
            )
        return (
            self.forbidden_effects.get(action_name, set()),
            ConflictType.FORBIDDEN_EFFECT,
        )

    def _get_required_set_and_type(
            self, action_name: str, model_part: ModelPart
    ) -> Tuple[Set[ParameterBoundLiteral], ConflictType]:
        if model_part == ModelPart.PRECONDITION:
            return (
                self.required_preconditions.get(action_name, set()),
                ConflictType.REQUIRED_PRECONDITION,
            )
        return (
            self.required_effects.get(action_name, set()),
            ConflictType.REQUIRED_EFFECT,
        )

    def _detect_forbidden_added(
            self,
            action_name: str,
            model_part: ModelPart,
            added_literals: List[Predicate],
            grounded_candidates: Set[GroundedPredicate]=None,

    ) -> List[Conflict]:
        """
        Detect conflicts where a literal that is FORBIDDEN by a model-level patch
        is being *added* to the action model.

        - For PRECONDITION: a forbidden precondition appears in the precondition set.
        - For EFFECT: a forbidden effect appears in the effect set (add/delete).

        `grounded_candidates` should be the set/list of grounded predicates from
        the current transition (e.g., previous state or add/del effects), so that
        we can report a grounded fluent string that caused the conflict.

        :param action_name: Name of the action being updated
        :param model_part: Part of the model being updated (PRECONDITION or EFFECT)
        :param added_literals: Literals being added to the action model
        :param grounded_candidates: Grounded predicates from the current transition to match forbidden pbl
        :return: List of detected conflicts
        """
        conflicts: List[Conflict] = []
        forbidden_set, conflict_type = self._get_forbidden_set_and_type(action_name, model_part)

        for lit in added_literals:
            if not isinstance(lit, Predicate):
                continue
            for forbidden_pbl in forbidden_set:
                if not forbidden_pbl.matches(lit):
                    continue
                gf = forbidden_pbl.get_grounded_candidate(
                    grounded_candidates
                )
                conflict = Conflict(
                    action_name=action_name,
                    pbl=forbidden_pbl,
                    conflict_type=conflict_type,
                    observation_index=self.current_observation_index,
                    component_index=self.current_component_index,
                    grounded_fluent=gf.untyped_representation,
                )
                conflicts.append(conflict)
                self.logger.warning(f"Detected conflict: {conflict}")
        return conflicts

    def _detect_required_missing(
            self,
            action_name: str,
            model_part: ModelPart,
            current_literals: Set[Predicate],
    ) -> List[Conflict]:
        """
        Detect conflicts where a literal that is REQUIRED by a model-level patch
        is *missing* from the current action model.

        - For PRECONDITION: a required precondition does not appear in the
          current precondition set.
        - For EFFECT: a required effect does not appear in the current effect set.

        There is no single “causing” grounded fluent here (the issue is precisely
        the *absence*), so we use the PBL string as grounded_fluent.

        :param action_name: Name of the action being updated
        :param model_part: Part of the model being updated (PRECONDITION or EFFECT)
        :param current_literals: Current literals in the action model part
        :return: List of detected conflicts
        """
        conflicts: List[Conflict] = []
        required_set, conflict_type = self._get_required_set_and_type(action_name, model_part)

        for required_pbl in required_set:
            found = any(
                isinstance(lit, Predicate) and required_pbl.matches(lit)
                for lit in current_literals
            )
            if found:
                continue

            conflict = Conflict(
                action_name=action_name,
                pbl=required_pbl,
                conflict_type=conflict_type,
                observation_index=self.current_observation_index,
                component_index=self.current_component_index,
                grounded_fluent=str(required_pbl),
            )
            conflicts.append(conflict)
            self.logger.warning(f"Detected conflict: {conflict}")
        return conflicts

    def _detect_required_removed(
            self,
            action_name: str,
            model_part: ModelPart,
            removed_literals: Set[Predicate],
    ) -> List[Conflict]:
        """
        Detect conflicts where a literal that is REQUIRED by a model-level patch
        has been *removed* from the action model (e.g., by an update step).

        - For PRECONDITION: a required precondition was in the precondition set
          before update but is not thereafter.
        - For EFFECT: similarly for effects (if you ever remove them).

        Since the conflict is about a model literal being removed, we do not
        have a specific grounded fluent from the trajectory to point to; we use
        the PBL string as grounded_fluent.

        :param action_name: Name of the action being updated
        :param model_part: Part of the model being updated (PRECONDITION or EFFECT)
        :param removed_literals: Literals being removed from the action model
        :return: List of detected conflicts
        """
        conflicts: List[Conflict] = []
        required_set, conflict_type = self._get_required_set_and_type(action_name, model_part)

        for lit in removed_literals:
            if not isinstance(lit, Predicate):
                continue
            for required_pbl in required_set:
                if not required_pbl.matches(lit):
                    continue
                gf = getattr(lit, "untyped_representation", str(lit))
                conflict = Conflict(
                    action_name=action_name,
                    pbl=required_pbl,
                    conflict_type=conflict_type,
                    observation_index=self.current_observation_index,
                    component_index=self.current_component_index,
                    grounded_fluent=gf,
                )
                conflicts.append(conflict)
                self.logger.warning(f"Detected conflict: {conflict}")
        return conflicts

    def _detect_required_conflicting_effects(
            self,
            action_name: str,
            cannot_be_effects,
            lifted_cannot_be: List[Predicate],
    ) -> List[Conflict]:
        """
        Detect conflicts where a REQUIRED effect (from a model-level patch)
        also appears in the PI-SAM 'cannot_be_effects' set for this transition.

        Semantically: the model says 'must be effect', while the PI-SAM rule
        for this (s, a, s') transition says 'cannot be effect' – a contradiction.

        We report a grounded fluent by searching in `cannot_be_effects` for a
        predicate whose name is compatible with the required PBL.

        :param action_name: Name of the action being updated
        :param cannot_be_effects: Grounded predicates in the cannot_be_effects set
        :param lifted_cannot_be: Lifted predicates in the cannot_be_effects set
        :return: List of detected conflicts
        """

        conflicts: List[Conflict] = []
        required_set, conflict_type = self._get_required_set_and_type(action_name, ModelPart.EFFECT)
        if not required_set:
            return conflicts

        for required_pbl in required_set:
            for lit in lifted_cannot_be:
                if not isinstance(lit, Predicate) or not required_pbl.matches(lit):
                    continue

                matching_grounded = next(
                    (gp for gp in cannot_be_effects if gp.name == required_pbl.predicate_name),
                    None,
                )
                grounded_fluent = (
                    matching_grounded.untyped_representation
                    if matching_grounded is not None
                    else str(required_pbl)
                )

                conflict = Conflict(
                    action_name=action_name,
                    pbl=required_pbl,
                    conflict_type=conflict_type,
                    observation_index=self.current_observation_index,
                    component_index=self.current_component_index,
                    grounded_fluent=grounded_fluent,
                )
                conflicts.append(conflict)
                self.logger.warning(f"Detected conflict: {conflict}")
        return conflicts

    # ========= Overrides on top of PISAMLearner =========

    def _add_new_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        """
        Add new action's preconditions with conflict detection for model patches.

        Checks the previous_state for:
        1. Forbidden predicates that would become preconditions
        2. Required predicates that are missing

        :param grounded_action: The action being executed
        :param previous_state: The state before the action
        """
        super()._add_new_action_preconditions(grounded_action, previous_state)

        action_name = grounded_action.name
        current_action = self.partial_domain.actions[action_name]
        preconds = list(current_action.preconditions.root.operands)

        forbidden_conflicts = self._detect_forbidden_added(
            action_name, ModelPart.PRECONDITION, preconds, get_state_grounded_predicates(previous_state)
        )
        self.conflicts.extend(forbidden_conflicts)

        missing_conflicts = self._detect_required_missing(
            action_name, ModelPart.PRECONDITION, set(preconds)
        )
        self.conflicts.extend(missing_conflicts)

    def _update_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        """
        Update action preconditions with conflict detection.

        After updating, checks for required predicates missing from previous_state.

        :param grounded_action: The action being executed
        :param previous_state: The state before the action
        """
        action_name = grounded_action.name
        current_action = self.partial_domain.actions[action_name]
        preconditions_before = set(current_action.preconditions.root.operands)

        # original PI-SAM update logic from PISAMLearner
        self.logger.debug(f"Updating the preconditions of {grounded_action.name} in the action model.")
        previous_state_unmasked_predicates = set(
            self.matcher.get_possible_literal_matches(
                grounded_action, list(get_state_unmasked_predicates(previous_state))
            )
        )
        previous_state_masked_predicates = set(
            self.matcher.get_possible_literal_matches(
                grounded_action, list(get_state_masked_predicates(previous_state))
            )
        )

        conditions_to_remove = []
        for current_precondition in current_action.preconditions.root.operands:
            if (
                    isinstance(current_precondition, Predicate)
                    and current_precondition not in previous_state_masked_predicates
                    and current_precondition not in previous_state_unmasked_predicates
            ):
                conditions_to_remove.append(current_precondition)

        for condition in conditions_to_remove:
            current_action.preconditions.remove_condition(condition)

        preconditions_after = set(current_action.preconditions.root.operands)
        removed_preconditions = preconditions_before - preconditions_after

        removed_conflicts = self._detect_required_removed(
            action_name, ModelPart.PRECONDITION, removed_preconditions
        )
        self.conflicts.extend(removed_conflicts)

    def handle_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        action_name = grounded_action.name

        prev_preds = get_state_grounded_predicates(previous_state)
        next_preds = get_state_grounded_predicates(next_state)

        grounded_add_effects, grounded_del_effects = extract_discrete_effects_partial_observability(
            prev_preds, next_preds
        )
        lifted_add_effects = self.matcher.get_possible_literal_matches(
            grounded_action, list(grounded_add_effects)
        )
        lifted_delete_effects = self.matcher.get_possible_literal_matches(
            grounded_action, list(grounded_del_effects)
        )
        all_lifted_effects = list(lifted_add_effects) + list(lifted_delete_effects)

        # 1) forbidden effects
        forbidden_conflicts = self._detect_forbidden_added(
            action_name, ModelPart.EFFECT, all_lifted_effects, set(grounded_add_effects).union(grounded_del_effects)
        )
        if forbidden_conflicts:
            self.conflicts.extend(forbidden_conflicts)
            return

        # 2) required effects vs cannot_be_effects
        cannot_be_effects = extract_not_effects_partial_observability(prev_preds, next_preds)
        lifted_cannot_be = list(
            self.matcher.get_possible_literal_matches(grounded_action, list(cannot_be_effects))
        )

        required_conflicts = self._detect_required_conflicting_effects(
            action_name, cannot_be_effects, lifted_cannot_be
        )
        if required_conflicts:
            self.conflicts.extend(required_conflicts)
            return

        # no conflicts -> delegate to normal PI-SAM logic
        super().handle_effects(grounded_action, previous_state, next_state)

    def learn_action_model(self, observations: List[Observation], **kwargs) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Learn action model from observations, tracking indices for conflict reporting.

        Overrides PISAMLearner to track observation/component indices.

        :param observations: A list of grounded and masked observations
        :return: A tuple of (learned domain, learning report dictionary)
        """
        # Start learning but track indices
        self.logger.info("Starting to learn the action model with conflict detection.")

        # Track indices manually before processing
        # We'll override the observation loop to track indices
        self.start_measure_learning_time()
        self.deduce_initial_inequality_preconditions()
        self._complete_possibly_missing_actions()

        for obs_idx, observation in enumerate(observations):
            self.current_observation_index = obs_idx
            self.current_trajectory_objects = observation.grounded_objects

            for comp_idx, component in enumerate(observation.components):
                self.current_component_index = comp_idx

                if not component.is_successful:
                    self.logger.warning("Skipping the transition because it was not successful.")
                    continue

                self.handle_single_trajectory_component(component)

        self.construct_safe_actions()
        self._remove_unobserved_actions_from_partial_domain()
        self.handle_negative_preconditions_policy()
        self.end_measure_learning_time()
        learning_report = self._construct_learning_report()

        return self.partial_domain, learning_report

    def learn_action_model_with_conflicts(
        self,
        observations: List[Observation],
        fluent_patches: Set[FluentLevelPatch],
        model_patches: Set[ModelLevelPatch]
    ) -> Tuple[LearnerDomain, List[Conflict]]:
        """
        Learn action model with patches and return detected conflicts.

        This is the main entry point for denoising-aware learning.

        :param observations: Training observations (should be grounded and masked)
        :param fluent_patches: Fluent-level patches to apply
        :param model_patches: Model-level constraints
        :return: Tuple of (learned domain, list of conflicts)
        """
        # Set patches
        self.set_patches(fluent_patches, model_patches)

        # Apply fluent patches to observations
        patched_observations = self.apply_fluent_patches(observations)

        # Learn using parent methods - conflicts will be detected in overridden methods
        learned_domain, _ = self.learn_action_model(patched_observations)

        return learned_domain, self.conflicts
