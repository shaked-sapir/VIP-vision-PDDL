"""Binds conflict-driven patch search to the PI-SAM learner."""

from pddl_plus_parser.models import Domain

from src.plan_denoising.conflict_search import ConflictDrivenPatchSearchBase
from src.plan_denoising.noisy_pisam_learning import NoisyPisamLearner


class ConflictDrivenPatchSearchPISAM(ConflictDrivenPatchSearchBase):
    """Conflict-driven patch search using PI-SAM (partial observability)."""

    def _create_learner(self, domain_copy: Domain) -> NoisyPisamLearner:
        return NoisyPisamLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy,
            seed=self.seed,
        )
