from copy import deepcopy
from typing import List, Set, Dict, Tuple

from pddl_plus_parser.models import Domain, Observation, State, ActionCall, Predicate, GroundedPredicate, Action
from sam_learning.core import LearnerDomain, extract_discrete_effects_partial_observability, \
    extract_not_effects_partial_observability
from utilities import NegativePreconditionPolicy

from src.action_model.pddl2gym_parser import negate_str_predicate
from src.pi_sam.noisy_pisam.simpler_version.typings import (
    ParameterBoundLiteral,
    ModelLevelPatch,
    FluentLevelPatch,
    PatchOperation,
    ModelPart,
    ConflictType,
    Conflict,
)
from src.pi_sam.pi_sam_learning import PISAMLearner
from src.utils.pddl import (
    get_state_grounded_predicates, get_state_unmasked_predicates, get_state_masked_predicates,
)


class NoisyPisamLearner(PISAMLearner):
    """
    A PI-SAM learner with:
      - Fluent-level patches (flip specific grounded fluents in trajectories).
      - Model-level patches:
            EFFECT + FORBID  -> cannot be effect
            EFFECT + REQUIRE -> must be effect
            PRE + FORBID     -> cannot be precondition
      - Conflict detection:
            * Patch vs PI-SAM:
                - FORBID_EFFECT_VS_MUST
                - REQUIRE_EFFECT_VS_CANNOT
                - FORBID_PRECOND_VS_IS
            * Data-only PI-SAM effect inconsistencies, using the same types:
                - PRIOR cannot_be_effect + new must-effect -> FORBID_EFFECT_VS_MUST
                - PRIOR must-effect + new cannot_be_effect -> REQUIRE_EFFECT_VS_CANNOT

    If *no* patches are given, conflicts may still arise from PI-SAM itself
    whenever an effect literal is classified as both "must-be-effect" and
    "cannot-be-effect" across different transitions of the SAME action schema.
    """

    def __init__(
        self,
        partial_domain: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        seed: int = 42,
    ):
        super().__init__(partial_domain, negative_preconditions_policy, seed)

        # fluent-level patches (data repairs)
        self.fluent_patches: Set[FluentLevelPatch] = set()

        # model-level patches (global constraints)
        self.model_patches: Set[ModelLevelPatch] = set()

        self.forbidden_effects: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.required_effects: Dict[str, Set[ParameterBoundLiteral]] = {}
        self.forbidden_preconditions: Dict[str, Set[ParameterBoundLiteral]] = {}

        # conflicts (collected during learning)
        self.conflicts: List[Conflict] = []

        # indices to localize conflicts
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

            elif patch.model_part == ModelPart.PRECONDITION:
                if patch.operation == PatchOperation.FORBID:
                    self.forbidden_preconditions.setdefault(patch.action_name, set()).add(patch.pbl)
                else:
                    # FORBID preconditions are not handled in this version
                    self.logger.warning(
                        f"REQUIRE precondition patch not supported in NoisyPisamLearner: {patch}"
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

                # IMPORTANT! Only flip in one state to avoid double-flipping(it applies automatically also to next_comp.prev
                if patch.state_type == "next":
                    if comp.next_state is not None:
                        self._flip_fluent_in_state(comp.next_state, patch.fluent)

                # IMPORTANT! Only flip in one state to avoid double-flipping(it applies automatically also to prev_comp.next
                else:  # "prev"
                    if comp.previous_state is not None:
                        self._flip_fluent_in_state(comp.previous_state, patch.fluent)
            except ValueError as ve:
                self.logger.warning(f"{ve} [{obs_idx}][{comp_idx}], Full Patch: {patch}")
                continue
        return patched_observations

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

        # self.logger.warning(f"Could not find fluent {fluent_str} to flip in state: [{self.current_observation_index}][{self.current_component_index}].")
        raise ValueError(f"Could not find fluent {fluent_str} to flip in state")
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

        base = f"({pbl.predicate_name} {' '.join(grounded_args)})"
        return base if pbl.is_positive else f"(not {base})"
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

    # def _update_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
    #     """
    #     Existing action case (PI-SAM cannot_be_precondition rule):
    #
    #     - If this transition violates a 'cannot be precondition' patch,
    #       record all conflicts and DO NOT update preconditions for this
    #       transition.
    #
    #     - Otherwise, delegate to the base PI-SAM implementation to apply
    #       cannot_be_precondition logic.
    #     """
    #     self.logger.debug(
    #         f"Updating preconditions of {grounded_action.name} with precondition patches."
    #     )
    #
    #     conflicts = self._collect_forbidden_precondition_conflicts(
    #         grounded_action,
    #         previous_state,
    #     )
    #     if conflicts:
    #         self.conflicts.extend(conflicts)
    #         return
    #
    #     # No conflicts: normal PI-SAM update (cannot_be_precondition)
    #     super()._update_action_preconditions(grounded_action, previous_state)

    def _update_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        """
        Existing action case (PI-SAM cannot_be_precondition rule), with:

        1) Patch-based conflicts:
           - PRECONDITION+FORBID patches vs preconditions implied by this transition
             (handled via _collect_forbidden_precondition_conflicts).

        2) Data-only PI-SAM conflicts:
           - If PI-SAM would remove a precondition literal L because it is
             neither masked nor true in the previous_state (cannot_be_precondition),
             we treat that as a 'cannot be precondition' conflict instead of
             silently removing it.

           Semantically: prior transitions made L a precondition of this action,
           this transition says L cannot be a precondition. We want the search
           to decide if we should:
             * repair the data (flip fluent in prev_state via FluentLevelPatch), or
             * keep PI-SAM's constraints (in practice, the model-branch is a no-op
               for data-only conflicts and gets pruned).
        """
        self.logger.debug(
            f"Updating preconditions of {grounded_action.name} with precondition patches and data-only conflicts."
        )

        action_name = grounded_action.name
        current_action = self.partial_domain.actions[action_name]

        # --- 1) Patch-based 'cannot be precondition' conflicts ---
        patch_conflicts = self._collect_forbidden_precondition_conflicts(
            grounded_action,
            previous_state,
        )

        # --- 2) PI-SAM cannot_be_precondition logic (data-only conflicts) ---
        prev_unmasked = set(
            self.matcher.get_possible_literal_matches(
                grounded_action,
                list(get_state_unmasked_predicates(previous_state)),
            )
        )
        prev_masked = set(
            self.matcher.get_possible_literal_matches(
                grounded_action,
                list(get_state_masked_predicates(previous_state)),
            )
        )

        conditions_to_remove: List[Predicate] = []
        data_conflicts: List[Conflict] = []

        for current_precondition in current_action.preconditions.root.operands:
            if not isinstance(current_precondition, Predicate):
                continue

            # This is exactly PI-SAM's cannot_be_precondition condition:
            if (
                current_precondition not in prev_masked
                and current_precondition not in prev_unmasked
            ):
                # Base PI-SAM would remove this precondition.
                # We instead create a data-only 'cannot be precondition' conflict.
                pbl = ParameterBoundLiteral(
                    predicate_name=current_precondition.name,
                    parameters=tuple(p for p in current_precondition.signature.keys()),
                    is_positive=getattr(current_precondition, "is_positive", True),
                )
                grounded_fluent_str = self._ground_pbl_with_action(pbl, grounded_action)

                conflict = Conflict(
                    action_name=action_name,
                    pbl=pbl,
                    conflict_type=ConflictType.FORBID_PRECOND_VS_IS,
                    observation_index=self.current_observation_index,
                    component_index=self.current_component_index,
                    grounded_fluent=negate_str_predicate(grounded_fluent_str)
                )
                data_conflicts.append(conflict)
                self.logger.warning(
                    f"Detected data-only cannot_be_precondition conflict: {conflict}"
                )

                # Track for potential removal if we ever choose to apply pure PI-SAM
                conditions_to_remove.append(current_precondition)

        all_conflicts = patch_conflicts + data_conflicts

        if all_conflicts:
            # Record all conflicts and DO NOT remove any preconditions here.
            # Conflict search will later:
            #   - create a FluentLevelPatch (flip in prev_state) for data conflicts
            #   - for patch-based conflicts: also has the (no-op) model branch
            self.conflicts.extend(all_conflicts)
            return

        # No conflicts at all -> perform standard PI-SAM cannot_be_precondition removal
        for condition in conditions_to_remove:
            current_action.preconditions.remove_condition(condition)
    # -------------------------------------------------------------------------
    # Effects: FORBID/REQUIRE vs must/cannot (including data-only [non-patched] conflicts)
    # -------------------------------------------------------------------------

    def handle_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """
        PI-SAM effect handling with conflict detection for:

        1) Patch-based conflicts:

            - FORBID_EFFECT_VS_MUST:
                  cannot-be-effect patch vs must-be-effect (discrete add/del).

            - REQUIRE_EFFECT_VS_CANNOT:
                  must-be-effect patch vs cannot-be-effect (cannot_be_effects).

        2) PI-SAM data-only inconsistencies:

            - FORBID_EFFECT_VS_MUST:
                  PRIOR cannot_be_effect contains literal L, but this transition
                  treats L as must-be-effect (discrete_effect).

            - REQUIRE_EFFECT_VS_CANNOT:
                  PRIOR discrete_effects contains literal L, but this transition
                  treats L as cannot-be-effect.

        If ANY conflicts are found in this transition, we record them and
        SKIP the regular PI-SAM effect update for this transition.
        """
        action_name = grounded_action.name
        observed_action = self.partial_domain.actions[action_name]

        prev_preds = get_state_grounded_predicates(previous_state)
        next_preds = get_state_grounded_predicates(next_state)

        local_conflicts: List[Conflict] = []

        # --- Must-be effects: discrete add/delete for this transition ---
        grounded_add_effects, grounded_del_effects = extract_discrete_effects_partial_observability(
            prev_preds, next_preds
        )
        all_grounded_must = list(grounded_add_effects) + list(grounded_del_effects)

        # History BEFORE this transition:
        prior_must_effects: Set[Predicate] = set(observed_action.discrete_effects)  # lifted Predicates
        prior_cannot_effects: Set[Predicate] = set(self.cannot_be_effect.get(action_name, set()))  # lifted Predicates

        # ------------------------------------------------------------------
        # (2a) DATA-ONLY: PRIOR cannot_be_effect + new must-be-effect
        #       -> treat as FORBID_EFFECT_VS_MUST
        # ------------------------------------------------------------------
        for gp in all_grounded_must:
            possible_lifted_gp = self.matcher.get_possible_literal_matches(grounded_action, [gp])
            lifted_gp = possible_lifted_gp[0] if possible_lifted_gp else None
            if lifted_gp in prior_cannot_effects:
                pbl = ParameterBoundLiteral(
                    predicate_name=lifted_gp.name,
                    parameters=tuple(p for p in lifted_gp.signature.keys()),
                    is_positive=getattr(lifted_gp, "is_positive", True),
                )
                conflict = Conflict(
                    action_name=action_name,
                    pbl=pbl,
                    conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,
                    observation_index=self.current_observation_index,
                    component_index=self.current_component_index,
                    grounded_fluent=gp.untyped_representation,
                )
                local_conflicts.append(conflict)
                self.logger.warning(f"Detected data effect conflict (cannot vs must): {conflict}")

        # ------------------------------------------------------------------
        # (1a) PATCH-BASED: FORBID_EFFECT_VS_MUST
        # ------------------------------------------------------------------
        forbid_set = self.forbidden_effects.get(action_name, set())
        for gp in all_grounded_must:
            for pbl in forbid_set:
                if self._lift_and_match(grounded_action, gp, pbl):
                    conflict = Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.untyped_representation
                    )
                    local_conflicts.append(conflict)
                    self.logger.warning(f"Detected patch-based effect conflict (FORBID vs must): {conflict}")

        # --- cannot-be-effects for this transition ---
        cannot_be_effects: Set[GroundedPredicate] = extract_not_effects_partial_observability(prev_preds, next_preds)

        # ------------------------------------------------------------------
        # (2b) DATA-ONLY: PRIOR must-be-effect + new cannot-be-effect
        #       -> treat as REQUIRE_EFFECT_VS_CANNOT
        # ------------------------------------------------------------------
        for gp in cannot_be_effects:
            possible_lifted_gp = self.matcher.get_possible_literal_matches(grounded_action, [gp])
            lifted_gp = possible_lifted_gp[0] if possible_lifted_gp else None
            if lifted_gp in prior_must_effects:
                pbl = ParameterBoundLiteral(
                    predicate_name=lifted_gp.name,
                    parameters=tuple(p for p in lifted_gp.signature.keys()),
                    is_positive=getattr(lifted_gp, "is_positive", True),
                )
                conflict = Conflict(
                    action_name=action_name,
                    pbl=pbl,
                    conflict_type=ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                    observation_index=self.current_observation_index,
                    component_index=self.current_component_index,
                    # it was in cannot_be_effects => it is negated in the state => we have to report the negation form
                    grounded_fluent=gp.copy(is_negated=True).untyped_representation,
                )
                local_conflicts.append(conflict)
                self.logger.warning(f"Detected data effect conflict (must vs cannot): {conflict}")

        # ------------------------------------------------------------------
        # (1b) PATCH-BASED: REQUIRE_EFFECT_VS_CANNOT
        # ------------------------------------------------------------------
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
                        grounded_fluent=gp.copy(is_negated=True).untyped_representation,
                    )
                    local_conflicts.append(conflict)
                    self.logger.warning(f"Detected patch-based effect conflict (REQUIRE vs cannot): {conflict}")

        # ------------------------------------------------------------------
        # Decide whether to apply PI-SAM effect update
        # ------------------------------------------------------------------
        if local_conflicts:
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
    ) -> Tuple[LearnerDomain, List[Conflict], Dict[str, str]]:
        """
        High-level API:

        LearnWithConflicts(T, P):
            Apply fluent-level patches
            Apply model-level patches
            Run PI-SAM with conflict detection
            Return (M, Conflicts)
        """
        self.set_patches(fluent_patches, model_patches)
        learned_domain, learning_report = self.learn_action_model(observations, **kwargs)
        return learned_domain, self.conflicts, learning_report
