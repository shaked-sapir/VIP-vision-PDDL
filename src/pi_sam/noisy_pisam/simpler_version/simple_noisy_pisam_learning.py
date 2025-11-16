from copy import deepcopy
from typing import List, Set, Dict, Tuple

from pddl_plus_parser.models import Domain, Observation, State, ActionCall, Predicate, GroundedPredicate, Action
from sam_learning.core import LearnerDomain, extract_discrete_effects_partial_observability, \
    extract_not_effects_partial_observability
from utilities import NegativePreconditionPolicy

from src.action_model.pddl2gym_parser import negate_str_predicate
from src.utils.pddl import (
    get_state_grounded_predicates,
    get_state_unmasked_predicates,
    get_state_masked_predicates,
)
from src.pi_sam.pi_sam_learning import PISAMLearner

from src.pi_sam.noisy_pisam.simpler_version.typings import (
    ParameterBoundLiteral,
    ModelLevelPatch,
    FluentLevelPatch,
    PatchOperation,
    ModelPart,
    ConflictType,
    Conflict,
)


class SimpleNoisyPisamLearner(PISAMLearner):
    """
    A simplified noisy PI-SAM learner that:

    - Applies fluent-level patches to trajectories (flipping specific fluents in
      prev/next states of specific transitions).

    - Applies model-level patches to PRECONDITIONS and EFFECTS:

        * EFFECT patches:
            - FORBID  : do not allow a certain PBL as effect.
            - REQUIRE : a certain PBL must be an effect.

        * PRECONDITION patches:
            - REQUIRE : a certain PBL must be a precondition.
                        We use this against the 'cannot_be_precondition' rule
                        (PI-SAM's precondition removal logic).

    - Detects conflicts directly at PI-SAM rule applications:

        Effects:
          - Must be effect (discrete add/del):
                if FORBID patch matches -> FORBID_VS_MUST conflict (skip update).
                else regular PI-SAM.

          - Not effect (cannot_be_effects):
                if REQUIRE patch matches -> REQUIRE_VS_CANNOT conflict (skip update).
                else regular PI-SAM.

        Preconditions:
          - When updating preconditions (cannot_be_precondition rule):
                if a literal is being removed AND matches a REQUIRED precondition PBL
                -> PRE_REQUIRE_VS_CANNOT conflict (do NOT remove it).

    All conflicts include:
      - observation_index / component_index (trajectory & transition indices)
      - grounded_fluent: always a grounded predicate string.
    """

    def __init__(
        self,
        partial_domain: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        seed: int = 42,
    ):
        super().__init__(partial_domain, negative_preconditions_policy, seed)

        # patches
        self.fluent_patches: Set[FluentLevelPatch] = set()
        self.model_patches: Set[ModelLevelPatch] = set()

        # indexed model patches
        self.forbidden_effects: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.required_effects: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.forbidden_preconditions: Dict[str, Set[ParameterBoundLiteral]] = {}

        # conflicts
        self.conflicts: List[Conflict] = []

        # indices for conflict localization
        self.current_observation_index: int = 0
        self.current_component_index: int = 0

    # -------------------------------------------------------------------------
    # Patch management
    # -------------------------------------------------------------------------

    def set_patches(
        self,
        fluent_patches: Set[FluentLevelPatch],
        model_patches: Set[ModelLevelPatch],
    ) -> None:
        """
        Set patch sets and build lookup dictionaries.

        Supported patch types:

          - (EFFECT, FORBID)   : cannot be effect
          - (EFFECT, REQUIRE)  : must be effect
          - (PRECONDITION, FORBID): cannot be precondition
        """
        self.fluent_patches = fluent_patches
        self.model_patches = model_patches
        self.conflicts = []

        self.forbidden_effects.clear()
        self.required_effects.clear()
        self.forbidden_preconditions.clear()

        for patch in model_patches:
            if patch.model_part == ModelPart.EFFECT:
                if patch.operation == PatchOperation.FORBID:
                    self.forbidden_effects.setdefault(patch.action_name, set()).add(patch.pbl)
                else:  # REQUIRE
                    self.required_effects.setdefault(patch.action_name, set()).add(patch.pbl)

            else:  # PRECONDITION
                if patch.operation == PatchOperation.FORBID:
                    self.forbidden_preconditions.setdefault(patch.action_name, set()).add(patch.pbl)
                else:
                    # FORBID preconditions are not handled in this simple version
                    self.logger.warning(
                        f"REQUIRE precondition patch not supported in SimpleNoisyPisamLearner: {patch}"
                    )

    # -------------------------------------------------------------------------
    # Fluent-level patches
    # -------------------------------------------------------------------------

    def apply_fluent_patches(self, observations: List[Observation]) -> List[Observation]:
        """
        Deep-copy observations and apply fluent-level patches.

        If state_type == "next": flip in this component's next_state and in the
        next component's previous_state.

        If state_type == "prev": flip in this component's previous_state and in
        the previous component's next_state.

        (Flipping itself is left as a hook; it depends on your State impl.)
        """
        patched = deepcopy(observations)

        for patch in self.fluent_patches:
            obs_idx = patch.observation_index
            comp_idx = patch.component_index

            if not (0 <= obs_idx < len(patched)):
                self.logger.warning(f"Fluent patch with invalid observation index: {patch}")
                continue

            obs = patched[obs_idx]
            if not (0 <= comp_idx < len(obs.components)):
                self.logger.warning(f"Fluent patch with invalid component index: {patch}")
                continue

            comp = obs.components[comp_idx]

            # IMPORTANT! Only flip in one state to avoid double-flipping(it applies automatically also to next_comp.prev
            if patch.state_type == "next":
                if comp.next_state is not None:
                    self._flip_fluent_in_state(comp.next_state, patch.fluent)

            # IMPORTANT! Only flip in one state to avoid double-flipping(it applies automatically also to prev_comp.next
            else:  # "prev"
                if comp.previous_state is not None:
                    self._flip_fluent_in_state(comp.previous_state, patch.fluent)

        return patched

    def _flip_fluent_in_state(self, state: State, fluent_str: str) -> None:
        """
        Flip fluent_str in the given state.
        """
        # Simple implementation: look for matching grounded predicate strings
        for gp in get_state_grounded_predicates(state):
            if gp.untyped_representation == fluent_str:
                gp_lifted_base_form = gp.lifted_untyped_representation if gp.is_positive else \
                    negate_str_predicate(gp.lifted_untyped_representation)
                gp_in_state = next((p for p in state.state_predicates[gp_lifted_base_form]
                                 if p.untyped_representation == fluent_str), None)
                gp_in_state.is_positive = not gp_in_state.is_positive
                self.logger.debug(f"Flipped fluent {fluent_str} in state.")
                return

        self.logger.warning(f"Could not find fluent {fluent_str} to flip in state.")

    # -------------------------------------------------------------------------
    # Helper: lift and match
    # -------------------------------------------------------------------------

    def _lift_and_match(
        self,
        grounded_action: ActionCall,
        grounded_pred: GroundedPredicate,
        pbl: ParameterBoundLiteral,
    ) -> bool:
        """
        Lift grounded_pred w.r.t. grounded_action and check if any lifted
        candidate matches the given PBL.
        """
        lifted_candidates = self.matcher.get_possible_literal_matches(
            grounded_action, [grounded_pred]
        )
        return any(
            isinstance(candidate, Predicate) and pbl.matches(candidate)
            for candidate in lifted_candidates
        )

    def _ground_pbl_with_action(self, pbl: ParameterBoundLiteral, grounded_action: ActionCall) -> str:
        """
        Ground a PBL with respect to a grounded action.

        Returns a GroundedPredicate.
        """
        # Get the lifted action schema to recover parameter names
        action_schema: Action = self.partial_domain.actions[grounded_action.name]

        # Map parameter name (e.g. '?x') -> object name (e.g. 'b1')
        # We assume order of action_schema.parameters matches the order
        # of grounded_action.grounded_parameters.
        signature_param_to_object: Dict[str, str] = {
            param: obj for param, obj in zip(action_schema.signature.keys(), grounded_action.parameters)
        }

        # Ground the PBL's parameters using that mapping
        grounded_args = []
        for p_name in pbl.parameters:
            grounded_args.append(signature_param_to_object[p_name])

        base = f"{pbl.predicate_name}({', '.join(grounded_args)})"
        return base if pbl.is_positive else f"not {base}"
    # -------------------------------------------------------------------------
    # Preconditions: REQUIRE vs cannot_be_precondition
    # -------------------------------------------------------------------------

    def _collect_forbidden_precondition_conflicts(
            self,
            grounded_action: ActionCall,
            previous_state: State,
    ) -> List[Conflict]:
        """
        Detect 'cannot be precondition' violations for a single transition.

        A conflict arises when:
          - There is a PRECONDITION+FORBID patch (cannot-be-precondition) for this action,
          - AND the current transition's previous_state contains a fluent that
            PI-SAM would treat as a candidate precondition,
          - AND that fluent matches the forbidden PBL.

        Returns ALL such conflicts for this (obs_idx, comp_idx, action).
        """
        action_name = grounded_action.name
        forbidden_set = self.forbidden_preconditions.get(action_name, set())
        if not forbidden_set:
            return []

        prev_grounded = list(get_state_grounded_predicates(previous_state))
        prev_lifted = set(
            self.matcher.get_possible_literal_matches(grounded_action, prev_grounded)
        )

        local_conflicts: List[Conflict] = []

        for lifted in prev_lifted:
            if not isinstance(lifted, Predicate):
                continue
            for pbl in forbidden_set:
                if pbl.matches(lifted):
                    grounded_fluent = self._ground_pbl_with_action(pbl, grounded_action)
                    conflict = Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.FORBID_PRECOND_VS_IS,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=grounded_fluent,
                    )
                    local_conflicts.append(conflict)
                    self.logger.warning(f"Detected forbidden-precondition conflict: {conflict}")

        return local_conflicts

    def _add_new_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        """
        New action case (PI-SAM rule):

        - If this transition violates a 'cannot be precondition' patch,
          record all conflicts and DO NOT add preconditions.

        - Otherwise, delegate to the base PI-SAM implementation, which:
            * adds all candidate preconditions from previous_state.
        """
        self.logger.debug(
            f"Adding preconditions of {grounded_action.name} with precondition patches."
        )

        conflicts = self._collect_forbidden_precondition_conflicts(
            grounded_action,
            previous_state,
        )
        if conflicts:
            self.conflicts.extend(conflicts)
            return

        # No conflicts: normal PI-SAM behavior
        super()._add_new_action_preconditions(grounded_action, previous_state)

    def _update_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        """
        Existing action case (PI-SAM cannot_be_precondition rule):

        - If this transition violates a 'cannot be precondition' patch,
          record all conflicts and DO NOT update preconditions for this
          transition.

        - Otherwise, delegate to the base PI-SAM implementation to apply
          cannot_be_precondition logic.
        """
        self.logger.debug(
            f"Updating preconditions of {grounded_action.name} with precondition patches."
        )

        conflicts = self._collect_forbidden_precondition_conflicts(
            grounded_action,
            previous_state,
        )
        if conflicts:
            self.conflicts.extend(conflicts)
            return

        # No conflicts: normal PI-SAM update (cannot_be_precondition)
        super()._update_action_preconditions(grounded_action, previous_state)
    # -------------------------------------------------------------------------
    # Effects: FORBID/REQUIRE vs must/cannot
    # -------------------------------------------------------------------------

    def handle_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """
        PI-SAM effect handling with conflict detection, aligned with the pseudo-code:

        Must be effect (discrete add/del):
            if FORBID patch matches -> FORBID_VS_MUST conflict, skip updates.
            else regular PI-SAM.

        Not effect (cannot_be_effects):
            if REQUIRE patch matches -> REQUIRE_VS_CANNOT conflict, skip updates.
            else regular PI-SAM.
        """
        action_name = grounded_action.name

        prev_preds = get_state_grounded_predicates(previous_state)
        next_preds = get_state_grounded_predicates(next_state)

        local_conflicts: List[Conflict] = []

        # === Must-be effects: discrete add/delete effects ===
        grounded_add_effects, grounded_del_effects = extract_discrete_effects_partial_observability(
            prev_preds, next_preds
        )

        forbid_set = self.forbidden_effects.get(action_name, set())

        for gp in list(grounded_add_effects) + list(grounded_del_effects):
            for pbl in forbid_set:
                if self._lift_and_match(grounded_action, gp, pbl):
                    conflict = Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.untyped_representation,
                    )
                    local_conflicts.append(conflict)
                    self.logger.warning(f"Detected conflict: {conflict}")

        # === Not effects: cannot_be_effects (PI-SAM rule) ===
        cannot_be_effects = extract_not_effects_partial_observability(prev_preds, next_preds)
        require_set = self.required_effects.get(action_name, set())

        for gp in cannot_be_effects:
            for pbl in require_set:
                if self._lift_and_match(grounded_action, gp, pbl):
                    conflict = Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.untyped_representation,
                    )
                    local_conflicts.append(conflict)
                    self.logger.warning(f"Detected conflict: {conflict}")

        if local_conflicts:
            # do not update effects if any conflicts detected
            self.conflicts.extend(local_conflicts)
            return

        # No conflicts -> regular PI-SAM effect handling
        super().handle_effects(grounded_action, previous_state, next_state)

    # -------------------------------------------------------------------------
    # Learning loop with index tracking
    # -------------------------------------------------------------------------

    def learn_action_model(
        self,
        observations: List[Observation],
        **kwargs,
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Same learning loop as PISAMLearner, but:

          - apply fluent-level patches on a deep copy of observations,
          - track (observation_index, component_index) for every transition,
          - conflicts are detected in _update_action_preconditions and handle_effects.
        """
        self.logger.info("Starting SimpleNoisyPisamLearner with conflict detection.")

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
    ) -> Tuple[LearnerDomain, List[Conflict]]:
        """
        High-level API:

        LearnWithConflicts(T, P):
            Apply fluent-level patches
            Apply model-level patches
            Run PI-SAM with conflict detection
            Return (M, Conflicts)
        """
        self.set_patches(fluent_patches, model_patches)
        learned_domain, _ = self.learn_action_model(observations, **kwargs)
        return learned_domain, self.conflicts
