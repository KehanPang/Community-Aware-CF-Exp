from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SearchSettings:
    """Online search budgets. Symbols follow ComCE.pdf Section 5.2."""
    rollouts: int = 32                       # N_roll
    explanation_budget: int = 10             # B
    proposals_per_step: int = 4              # proposals scored by S before each sampled action
    guidance_scale: float = 0.5              # weight of S in the proposal choice
    guidance_batch_size: int = 4
    failure_threshold: float = 0.5           # tau_low
    failures_before_expansion: int = 4       # f_max
    expansion_budget: int = 8                # B_ext
    expansion_communities_per_trigger: int | str = 'all'
    certification_queries: int = 9           # Q
    lookahead: int = 2                       # l
    cleanup_queries: int = 2                 # Q_clean
    dedup_threshold: float = 0.9             # tau_dup
    temperature: float = 1.0

    def __post_init__(self):
        for name in ('rollouts', 'explanation_budget', 'proposals_per_step', 'failures_before_expansion', 'guidance_batch_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('certification_queries', 'cleanup_queries', 'lookahead', 'expansion_budget'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.explanation_budget > 10:
            raise ValueError('The WorldModel and policy accept at most 10 deleted edges')
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0.0:
            raise ValueError('guidance_scale must be finite and >= 0')
        if not math.isfinite(self.failure_threshold) or not 0 <= self.failure_threshold <= 1:
            raise ValueError('failure_threshold must be in [0,1]')
        if isinstance(self.dedup_threshold, bool) or not isinstance(self.dedup_threshold, (int, float)) \
                or not 0 <= self.dedup_threshold <= 1:
            raise ValueError('dedup_threshold must be in [0,1]')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be positive')
        per_trigger = self.expansion_communities_per_trigger
        if per_trigger != 'all' and (type(per_trigger) is not int or per_trigger < 1):
            raise ValueError("expansion_communities_per_trigger must be 'all' or a positive integer")

    @property
    def query_cap(self):
        return 1 + self.certification_queries + self.cleanup_queries

    def payload(self):
        return {**asdict(self), 'query_cap': self.query_cap}
