import ast
import hashlib
import json
import operator
from collections import Counter, defaultdict

import numpy as np


def row(*values, dtype=None):
    out = np.r_[tuple(values)]
    return out if dtype is None else out.astype(dtype)


def columns(*values, n, dtype=None):
    out = np.column_stack([np.full(n, v) if np.ndim(v) == 0 else v for v in values])
    return out if dtype is None else out.astype(dtype)


def mode_signature(values, *, tie='lowest', buckets=16):
    values = np.asarray(values, np.int64)
    modes = np.zeros(values.shape[1], np.int64)
    purity = np.zeros(values.shape[1], np.float32)
    if tie not in {'lowest', 'first'}:
        raise ValueError('Unknown signature mode tie rule')
    for j in range(values.shape[1]):
        if len(values):
            if tie == 'lowest':
                counts = np.bincount(values[:, j], minlength=buckets)
                modes[j] = int(np.argmax(counts))
                purity[j] = counts[modes[j]] / len(values)
            else:
                modes[j] = Counter(values[:, j].tolist()).most_common(1)[0][0]
                purity[j] = float(np.mean(values[:, j] == modes[j]))
    return modes, purity


def plain(value):
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def json_keys(fields):
    return np.asarray([json.dumps(plain(row), sort_keys=True, ensure_ascii=True) for row in zip(*fields)])


def nearest_signature_keys(center, tokens, match, *, divisor=1, dimensions=2, cut=.5, include_center_rel=True):
    dims = np.argsort(np.abs(center[None, :] - tokens), axis=1, kind='stable')[:, :min(dimensions, tokens.shape[1])]
    vals = np.take_along_axis(tokens, dims, axis=1) // divisor
    return np.asarray([json.dumps([[int(d), int(v)] for d, v in zip(dims[i], vals[i])]
                                  + ([['center_rel', (0 if float(match[i]) < cut else 1)]] if include_center_rel else []),
                                  sort_keys=True, ensure_ascii=True) for i in range(len(tokens))])


def quantize(values, cuts):
    return np.asarray([(values[:, i, None] > cuts[i]).sum(1) for i in range(len(cuts))]).T.astype(np.int64)


def endpoint_degree(edges, side):
    counts = Counter()
    for edge in edges:
        for p in ('src', 'dst'):
            counts[(str(edge[p + '_type']), str(edge[p + '_id']))] += 1
    return np.asarray([counts[(str(e[side + '_type']), str(e[side + '_id']))] for e in edges], np.float64)


class FeatureProgram:
    def __init__(self, inputs):
        self.values = dict(inputs)
        self.operations = {
            'array': np.asarray, 'row': row, 'columns': columns,
            'concat': np.concatenate, 'stack': np.stack,
            'zeros': np.zeros, 'ones': np.ones, 'eye': np.eye,
            'mean': np.mean, 'sum': np.sum, 'max': np.max, 'min': np.min,
            'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log, 'log1p': np.log1p,
            'clip': np.clip, 'sort': np.sort, 'argsort': np.argsort,
            'where': np.where, 'bincount': np.bincount, 'maximum': np.maximum, 'minimum': np.minimum,
            'cast': lambda v, dtype: np.asarray(v, dtype=dtype),
            'mode': mode_signature, 'json_keys': json_keys, 'nearest_signature_keys': nearest_signature_keys,
            'quantize': quantize, 'endpoint_degree': endpoint_degree,
            'endpoint_pairs': lambda edges: [(e['src_type'], e['dst_type']) for e in edges],
            'field': lambda rows, key: np.asarray([r[key] for r in rows]),
            'int': int, 'float': float, 'str': str, 'len': len,
            'list': lambda v: [str(x) for x in v],
        }

    def evaluate(self, source):
        return self._node(ast.parse(source, mode='eval').body)

    def _node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.values[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self._node(v) for v in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            return {self._node(k): self._node(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            return self._node(node.value)[self._node(node.slice)]
        if isinstance(node, ast.Slice):
            return slice(*(self._node(v) if v is not None else None for v in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.UnaryOp):
            operation = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}[type(node.op)]
            return operation(self._node(node.operand))
        if isinstance(node, ast.BinOp):
            operation = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
                         ast.FloorDiv: operator.floordiv, ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
                         ast.Pow: operator.pow}[type(node.op)]
            return operation(self._node(node.left), self._node(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            operation = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                         ast.Gt: operator.gt, ast.GtE: operator.ge}[type(node.ops[0])]
            return operation(self._node(node.left), self._node(node.comparators[0]))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.operations
                and all(k.arg for k in node.keywords)):
            return self.operations[node.func.id](*[self._node(a) for a in node.args],
                                                 **{k.arg: self._node(k.value) for k in node.keywords})
        raise ValueError(f'Unsupported declarative feature expression: {type(node).__name__}')

    def execute(self, program):
        for name, expression in program.items():
            self.values[name] = self.evaluate(expression)
        return self.values


def partition_keyed_groups(keyed_views, hops, center_id):
    communities = []
    for view, codes in keyed_views:
        if len(codes) != len(hops):
            raise ValueError('Region key/edge alignment mismatch')
        center = center_id
        try:
            center = int(center)
        except (TypeError, ValueError):
            center = str(center)
        groups = defaultdict(list)
        raw_codes = {}
        for i, code in enumerate(codes):
            key = str(code)
            groups[key].append(i)
            raw_codes[key] = code
        for key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            identity = view['id_format'].format(center=center, family=view['name'], code=raw_codes[key],
                                                digest=hashlib.md5(key.encode('utf-8')).hexdigest()[:12])
            communities.append(dict(community_id=identity, relation_family=view['name'], relation_key=key,
                                    member_indices=members, size=len(members),
                                    mean_hop=float(np.mean(hops[members])) if members else 0.))
    return communities
