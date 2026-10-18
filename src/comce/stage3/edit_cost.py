from collections.abc import Mapping
import math


class EdgeCost:
    def __init__(self, *, mode='uniform', default_weight=1., weights=None, function=None, identity=None):
        if mode not in {'uniform', 'mapping', 'callable'}:
            raise ValueError('Unknown edge cost mode')
        self.mode = mode
        self.default_weight = self._weight(default_weight)
        self.weights = None
        self.function = function
        self.identity = identity
        if mode == 'mapping':
            if not isinstance(weights, Mapping) or any(not isinstance(k, str) for k in weights):
                raise ValueError('Edge weights require canonical string edge IDs')
            self.weights = {k: self._weight(v) for k, v in weights.items()}
        if mode == 'callable' and (not callable(function) or not isinstance(identity, str) or not identity.strip()):
            raise ValueError('Callable edge cost requires function(edge) and an explicit identity')
        if mode != 'callable' and function is not None:
            raise ValueError('function requires callable edge cost mode')

    @staticmethod
    def _weight(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Edge costs must be finite nonnegative numbers')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Edge costs must be finite nonnegative numbers')
        return value

    def bind(self, record):
        keys = list(map(str, record['edge_keys']))
        edges = {str(e['canonical_key']): e for e in record['edges']}
        if len(keys) != len(set(keys)) or set(edges) != set(keys):
            raise ValueError('Edge cost requires a complete canonical edge identity mapping')
        if self.mode == 'mapping':
            missing = set(keys) - self.weights.keys()
            if missing:
                raise ValueError(f'Missing edge costs for {len(missing)} legal edges: {min(missing)}')
            values = {k: self.weights[k] for k in keys}
        elif self.mode == 'callable':
            values = {k: self._weight(self.function(dict(edges[k]))) for k in keys}
        else:
            values = dict.fromkeys(keys, self.default_weight)
        return BoundEdgeCost(values, mode=self.mode, identity=self.identity)


class BoundEdgeCost:
    def __init__(self, values, *, mode, identity=None):
        self.values = values
        self.mode = mode
        self.identity = identity

    def __call__(self, keys):
        keys = tuple(keys)
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate edge ID in edit cost')
        try:
            value = math.fsum(self.values[k] for k in sorted(keys))
        except KeyError as exc:
            raise ValueError(f'Unknown canonical edge ID: {exc.args[0]}') from exc
        if not math.isfinite(value):
            raise ValueError('Non-finite total edit cost')
        return value

    def payload(self):
        return dict(mode=self.mode, identity=self.identity,
                    optimization='minimum_edit_cost_among_exact_certificates',
                    candidate_priority='descending_S_then_edit_cost',
                    cleanup_priority='ascending_singleton_S',
                    cardinality_budgets='edge_count', policy_objective='frozen_checkpoint_unit_cost_training')
