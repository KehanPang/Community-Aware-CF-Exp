import numpy as np
import torch

from comce.stage3.world_model import STATES


class RequestScorer:
    """Edge encodings of one search domain under both WorldModel parameter states."""

    def __init__(self, world_model, ranking_residual, sets, signatures, chunk_size=4096):
        self.world_model = world_model
        self.ranking_residual = ranking_residual
        self.sets = sets
        self.signatures = signatures
        self.chunk_size = chunk_size
        self.keys = list(sets.keys)
        self.index = {key: i for i, key in enumerate(self.keys)}
        if len(self.index) != len(self.keys):
            raise ValueError('Duplicate WM canonical edge ID')
        if signatures is not None:
            signatures.validate(self.keys)
        device = next(world_model.parameters()).device
        if sets.edge_rows.shape[-1] != world_model['omega'].edge_in:
            raise ValueError('WorldModel input contract mismatch')
        raw = sets.edge_rows.to(device)
        self.device = device
        self.embeddings = {state: world_model[state].encode_edges(raw) for state in STATES}

    def index_sets(self, candidates):
        result = []
        for candidate in candidates:
            if not 0 < len(candidate) <= 10 or len(candidate) != len(set(candidate)):
                raise ValueError('WM requires distinct nonempty edge sets of size <= 10')
            try:
                result.append(sorted(self.index[k] for k in candidate))
            except KeyError as exc:
                raise ValueError(f'Candidate edge outside the WM graph: {exc.args[0]}') from exc
        return result

    def state_logits(self, state, indices, mask):
        embeddings = self.embeddings[state][indices, torch.arange(indices.shape[1], device=indices.device)[None]]
        return self.world_model[state](embeddings, self.sets.set_features(indices, mask), mask)

    def residual_logits(self, indices, mask):
        rows = self.sets.edge_rows[:, 0, :].to(self.device)[indices]
        return self.ranking_residual(rows, mask)

    @staticmethod
    def as_array(values, count):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, np.float64)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError('Invalid World Model scores')
        return values.copy()

    @torch.no_grad()
    def world_model_scores(self, candidates):
        """z_omega for rollout prefixes (failure detection and expansion)."""
        sets = self.index_sets(candidates)
        if not sets:
            return np.empty(0, np.float64)
        if self.signatures is not None:
            self.signatures.validate([self.keys[i] for indices in sets for i in indices])
        values = torch.cat([self.state_logits('omega', *self.sets.pack(sets[begin:begin + self.chunk_size]))
                            for begin in range(0, len(sets), self.chunk_size)])
        return self.as_array(values, len(sets))

    def contract(self):
        omega = self.world_model['omega']
        return dict(canonical_edge_ids=list(self.keys), world_model='WorldModel', parameter_states=list(STATES),
                    edge_input_dim=omega.edge_in, set_input_dim=omega.set_dim, rollout_score='z_omega(u,D)',
                    ranking_score='z_omega_bar(u,D)+w_r^T psi(u,D)+b_r',
                    residual_descriptors='mean and max pooled center-relative edge rows plus |D|/B',
                    signature_context=self.signatures.statistics(self.keys))
