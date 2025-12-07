from copy import deepcopy
from typing import List, Set, Dict, Tuple

from pddl_plus_parser.models import Domain, Observation, State, ActionCall, Predicate, GroundedPredicate, Action
from sam_learning.core import LearnerDomain
from sam_learning.core.matching_utils import extract_discrete_effects_partial_observability, \
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
    get_state_grounded_predicates, )


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
            * Frame axiom violations:
                - FRAME_AXIOM: predicate changes without sharing objects with the action.

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
        candidates = {fluent_str, negate_str_predicate(fluent_str)}

        for gp in get_state_grounded_predicates(state):
            if gp.untyped_representation not in candidates:
                continue

            # Figure out which lifted key this gp lives under in state.state_predicates
            base_key = (
                gp.lifted_untyped_representation
                if gp.is_positive
                else negate_str_predicate(gp.lifted_untyped_representation)
            )

            # Find the exact instance in state.state_predicates and flip it
            for p in state.state_predicates[base_key]:
                if p.untyped_representation in candidates:
                    p.is_positive = not p.is_positive
                    # self.logger.debug(f"Flipped fluent {p.untyped_representation} in state.")
                    return

        # If we get here, neither the fluent nor its negation was found.
        raise ValueError(
            f"Could not find fluent {fluent_str} or its negation to flip in state"
        )

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
        grounded_args = [signature_param_to_object[p_name] for p_name in pbl.parameters]

        base = f"({pbl.predicate_name} {' '.join(grounded_args)})"
        return base if pbl.is_positive else f"(not {base})"
    # -------------------------------------------------------------------------
    # Preconditions handling - no conflicts possible
    # -------------------------------------------------------------------------

    def _add_new_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        super()._add_new_action_preconditions(grounded_action, previous_state)

    def _update_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        super()._update_action_preconditions(grounded_action, previous_state)

    # -------------------------------------------------------------------------
    # Frame-axiom conflicts
    # -------------------------------------------------------------------------

    def _collect_frame_axiom_conflicts(
            self,
            grounded_action: ActionCall,
            grounded_add_effects: Set[GroundedPredicate],
            grounded_del_effects: Set[GroundedPredicate],
    ) -> List[Conflict]:
        """
        Detect frame-axiom violations for a single transition.

        A frame-axiom violation arises when:
          - A grounded fluent changes truth value between prev/next
            (i.e., it's in add or delete effects),
          - AND none of its objects appear among the action's parameters.

        We encode direction via frame_is_add:
          True  -> came from ADD effects  (¬p -> p)
          False -> came from DEL effects  (p  -> ¬p)
        """
        action_name = grounded_action.name
        action_objs = set(grounded_action.parameters)
        local_conflicts: List[Conflict] = []

        for gp, frame_is_add in (
                [(g, False) for g in grounded_del_effects] +  # p -> ¬p
                [(g, True) for g in grounded_add_effects]  # ¬p -> p
        ):
            gp_objs = set(gp.object_mapping.values())
            # 1) Skip nullary preds: frame axiom not applicable
            if len(gp_objs) == 0:
                continue

            # 2) If shares an object with the action → normal effect, no frame violation
            if set(gp_objs) <= action_objs:
                continue

            pbl = ParameterBoundLiteral(
                predicate_name=gp.name,
                parameters=tuple(),
                is_positive=gp.is_positive
            )
            to_negate = self._should_negate_grounded_effect(
                gp,
                self.current_observation_index,
                self.current_component_index,
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
            kind = "ADD" if frame_is_add else "DEL"
            # self.logger.warning(f"Detected frame-axiom {kind} conflict: {conflict}")

        return local_conflicts

    # -------------------------------------------------------------------------
    # Effect handling with conflicts (patch + data-only + frame axioms)
    # -------------------------------------------------------------------------

    def _should_negate_grounded_effect(
            self,
            grounded_predicate,
            conflict_observation_index: int,
            conflict_component_index: int,
    ) -> bool:
        """
        Decide whether the grounded_fluent of a REQUIRE_EFFECT_VS_CANNOT conflict
        should be reported as negated.

        Negation is needed **only** if:
          - There exists a fluent patch that would flip this same grounded predicate,
          - That patch is at:
                observation_index == conflict_observation_index
                component_index == conflict_component_index + 1
                state_type == "prev"
          - And its fluent name matches grounded_predicate.untyped_representation (ignoring negation).

        This corresponds exactly to the PI-SAM semantics:
            cannot_be_effect at comp=i  ↔  negated in next state of comp=i
            but a "prev" patch at comp=i+1 would flip it back.

        Parameters
        ----------
        grounded_predicate : Predicate
            The *grounded* predicate gp (not lifted) part of the conflict.

        conflict_observation_index : int
            Original conflict obs index.

        conflict_component_index : int
            Original conflict component idx (the transition where cannot_be_effect was detected).

        Returns
        -------
        bool
            True  -> use negated form (gp.copy(is_negated=True))
            False -> use positive form   (gp.untyped_representation)
        """

        gp_repr = grounded_predicate.untyped_representation
        obs_idx = conflict_observation_index
        target_comp = conflict_component_index + 1  # The (i+1) component whose prev_state would be flipped

        for patch in self.fluent_patches:
            if (
                    patch.observation_index == obs_idx
                    and patch.component_index == target_comp
                    and patch.state_type == "prev"
                    and patch.fluent == gp_repr
            ):
                # Patch flips that very fluent → conflict should NOT add an extra negation
                return False

        # No patch overrides the negation → report negated form
        return True

    def handle_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """
        PI-SAM effect handling with conflict detection for:

        1) Patch-based conflicts:

            - FORBID_EFFECT_VS_MUST:
                  cannot-be-effect patch vs must-be-effect (discrete add/del).

            - REQUIRE_EFFECT_VS_CANNOT:
                  must-be-effect patch vs cannot-be-effect (cannot_be_effects).

        2) data-only inconsistencies:

            - PISAM FORBID_EFFECT_VS_MUST:
                  PRIOR cannot_be_effect contains literal L, but this transition
                  treats L as must-be-effect (discrete_effect).

            - PISAM REQUIRE_EFFECT_VS_CANNOT:
                  PRIOR discrete_effects contains literal L, but this transition
                  treats L as cannot-be-effect.

            - FRAME_AXIOM violations:
                    grounded effect literal changes truth value but none of its objects
                    appear among the action's parameters.

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

        # Frame-axiom conflicts
        local_conflicts.extend(
            self._collect_frame_axiom_conflicts(
                grounded_action,
                grounded_add_effects,
                grounded_del_effects,
            )
        )

        # # Early exit if any frame-axiom conflicts found
        # if local_conflicts:
        #     self.conflicts.extend(local_conflicts)
        #     return

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
                # self.logger.warning(f"Detected data effect conflict (cannot vs must): {conflict}")

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
                    # self.logger.warning(f"Detected patch-based effect conflict (FORBID vs must): {conflict}")

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
                to_negate = self._should_negate_grounded_effect(
                    gp,
                    self.current_observation_index,
                    self.current_component_index,
                )
                conflict = Conflict(
                    action_name=action_name,
                    pbl=pbl,
                    conflict_type=ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                    observation_index=self.current_observation_index,
                    component_index=self.current_component_index,
                    # it was in cannot_be_effects => it is negated in the state => we have to report the negation form
                    # grounded_fluent=gp.copy(is_negated=True).untyped_representation,
                    grounded_fluent=gp.copy(is_negated=to_negate).untyped_representation,
                )
                local_conflicts.append(conflict)
                # self.logger.warning(f"Detected data effect conflict (must vs cannot): {conflict}")

        # ------------------------------------------------------------------
        # (1b) PATCH-BASED: REQUIRE_EFFECT_VS_CANNOT
        # ------------------------------------------------------------------
        require_set = self.required_effects.get(action_name, set())
        for gp in cannot_be_effects:
            for pbl in require_set:
                if self._lift_and_match(grounded_action, gp, pbl):
                    to_negate = self._should_negate_grounded_effect(
                        gp,
                        self.current_observation_index,
                        self.current_component_index,
                    )
                    conflict = Conflict(
                        action_name=action_name,
                        pbl=pbl,
                        conflict_type=ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                        observation_index=self.current_observation_index,
                        component_index=self.current_component_index,
                        grounded_fluent=gp.copy(is_negated=to_negate).untyped_representation,
                    )
                    local_conflicts.append(conflict)
                    # self.logger.warning(f"Detected patch-based effect conflict (REQUIRE vs cannot): {conflict}")
        # ------------------------------------------------------------------
        # Early exit if any Effect-conflicts found
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
                # --- EARLY EXIT: stop learning at the first conflicting transition ---
            #     if self.conflicts:
            #         self.logger.info(
            #             f"Stopping learning early after first conflict at "
            #             f"obs={obs_idx}, comp={comp_idx}, #conflicts={len(self.conflicts)}"
            #         )
            #         break  # break inner loop over components
            #
            # if self.conflicts:
            #     break  # break outer loop over observations

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
