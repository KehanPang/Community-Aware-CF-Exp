"""Typed-graph adapter: node features per type and typed edge arrays; M runs on the incoming typed subgraph."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from comce.datasets.oracle import CanonicalEdgeDomain, CertificationContext, ExplainedModel, GraphBatch
from comce.datasets.schema import GraphSchema

HOP_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.2}


def hop_weight_for(hop):
    hop = int(hop)
    return HOP_WEIGHTS[1] if hop <= 1 else HOP_WEIGHTS[2] if hop == 2 else HOP_WEIGHTS[3]


class TypedIncomingIndex:
    def __init__(self, edges, node_counts):
        self.edges = {tuple(k): np.asarray(v, np.int64) for k, v in edges.items()}
        self.counts = node_counts
        self.csr = {}
        for rel, arr in self.edges.items():
            order = np.argsort(arr[1], kind='stable')
            degree = np.bincount(arr[1], minlength=node_counts[rel[2]])
            self.csr[rel] = (np.r_[0, degree.cumsum()], order)

    def extract(self, root_type, root, layers):
        if layers < 1:
            raise ValueError('Positive dependency radius required')
        seen = {t: set() for t in self.counts}
        seen[root_type].add(int(root))
        frontier = {t: set(v) for t, v in seen.items()}
        selected = {rel: [] for rel in self.edges}
        for _ in range(layers):
            nxt = {t: set() for t in seen}
            for rel, arr in self.edges.items():
                src, _, dst = rel
                ptr, order = self.csr[rel]
                cols = [order[ptr[v]:ptr[v + 1]] for v in sorted(frontier[dst])]
                if not cols:
                    continue
                columns = np.concatenate(cols)
                selected[rel].append(columns)
                for value in arr[0, columns]:
                    v = int(value)
                    if v not in seen[src]:
                        nxt[src].add(v)
            for t in seen:
                seen[t].update(nxt[t])
            frontier = nxt
        nodes = {t: np.asarray(sorted(v), np.int64) for t, v in seen.items()}
        out, columns = {}, {}
        for rel, parts in selected.items():
            cols = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
            columns[rel] = cols
            a = self.edges[rel][:, cols]
            out[rel] = np.stack([np.searchsorted(nodes[rel[0]], a[0]), np.searchsorted(nodes[rel[2]], a[1])])
        return dict(nodes=nodes, edges=out, columns=columns, center_index=int(np.searchsorted(nodes[root_type], root)))


class TypedReceptiveField:
    """Canonical (undirected) edges within three hops of a target; each maps to one or more message columns."""

    def __init__(self, base_edge_arrays, canonical, type_order, endpoints, target_type):
        import scipy.sparse as sp
        self.canonical = canonical
        self.endpoints = endpoints
        self.target_type = target_type
        self.rels = list(base_edge_arrays)
        counts = {}
        for rel, arr in base_edge_arrays.items():
            src_type, _, dst_type = rel
            if arr.size:
                counts[src_type] = max(counts.get(src_type, 0), int(arr[0].max()) + 1)
                counts[dst_type] = max(counts.get(dst_type, 0), int(arr[1].max()) + 1)
        self.offsets = {}
        off = 0
        for ntype in type_order:
            self.offsets[ntype] = off
            off += int(counts.get(ntype, 0))
        self.n_nodes = off
        src_all = np.concatenate([arr[0].astype(np.int64) + self.offsets[rel[0]] for rel, arr in base_edge_arrays.items()])
        dst_all = np.concatenate([arr[1].astype(np.int64) + self.offsets[rel[2]] for rel, arr in base_edge_arrays.items()])
        data = np.ones(src_all.shape[0], dtype=np.int8)
        sym = sp.csr_matrix((data, (src_all, dst_all)), shape=(self.n_nodes, self.n_nodes))
        self.adj = (sym + sym.T).tocsr()
        self.adj.data[:] = 1
        self.integer_adj = self.adj.astype(np.int32)
        self.src_g = {rel: arr[0].astype(np.int64) + self.offsets[rel[0]] for rel, arr in base_edge_arrays.items()}
        self.dst_g = {rel: arr[1].astype(np.int64) + self.offsets[rel[2]] for rel, arr in base_edge_arrays.items()}

    def dist_from(self, center):
        dist = np.full(self.n_nodes, 127, dtype=np.int8)
        start = int(center) + self.offsets[self.target_type]
        frontier = np.zeros(self.n_nodes, dtype=np.int32)
        frontier[start] = 1
        dist[start] = 0
        for step in range(1, 4):
            reached = np.asarray(self.integer_adj @ frontier) > 0
            reached[dist < 127] = False
            dist[reached] = step
            frontier = reached.astype(np.int32)
            if not frontier.any():
                break
        return dist

    def hop_by_pair(self, center):
        dist = self.dist_from(center)
        hop_by_pair = {}
        for rel in self.rels:
            edge_type, swap = self.canonical[rel]
            s, d = self.src_g[rel], self.dst_g[rel]
            hop = np.minimum(dist[s].astype(np.int16), dist[d].astype(np.int16)) + 1
            mask = hop <= 3
            if not mask.any():
                continue
            s_loc = s[mask] - self.offsets[rel[0]]
            d_loc = d[mask] - self.offsets[rel[2]]
            a_id = d_loc if swap else s_loc
            b_id = s_loc if swap else d_loc
            for a, b, h in zip(a_id.tolist(), b_id.tolist(), hop[mask].tolist()):
                key = (edge_type, int(a), int(b))
                prev = hop_by_pair.get(key)
                if prev is None or h < prev:
                    hop_by_pair[key] = int(h)
        return hop_by_pair

    def pair_to_edge(self, pair, hop):
        edge_type, a, b = pair
        src_type, dst_type = self.endpoints[edge_type]
        return {'edge_type': edge_type, 'src_type': src_type, 'src_id': int(a), 'dst_type': dst_type, 'dst_id': int(b),
                'hop': int(hop), 'hop_weight': hop_weight_for(hop), 'also_delete_reverse': True,
                'canonical_key': f'{edge_type}|{src_type}|{int(a)}|{dst_type}|{int(b)}'}

    def domains(self, center):
        hops = self.hop_by_pair(center)
        edges = [self.pair_to_edge(pair, hops[pair])
                 for pair in sorted(hops, key=lambda p: self.pair_to_edge(p, hops[p])['canonical_key'])]
        if len({str(e['canonical_key']) for e in edges}) != len(edges):
            raise ValueError('duplicate canonical keys')
        u12 = [e for e in edges if int(e['hop']) <= 2]
        return {'center_id': int(center), 'U12': u12, 'U123': edges, 'U12_size': len(u12), 'U123_size': len(edges),
                'U12_edge_type_counts': dict(Counter(e['edge_type'] for e in u12))}


class TypedGraphIO:
    def __init__(self, resources, options, *, device, allowed_ids):
        self.device = device
        self.options = options
        self.allowed_ids = {str(c): str(s) for c, s in allowed_ids.items()}
        self.target_type = options['target_type']
        raw = torch.load(Path(resources['data']) / options['graph'], weights_only=True)
        self.features = raw['features']
        self.edges = {r: e.numpy() for r, e in raw['edges'].items()}
        self.incoming = TypedIncomingIndex(self.edges, {t: len(x) for t, x in self.features.items()})
        canonical = {tuple(rel): (name, bool(swap)) for rel, name, swap in options['canonical_relations']}
        endpoints = {name: tuple(pair) for name, pair in options['edge_endpoints'].items()}
        self.canonical = canonical
        self.rf = TypedReceptiveField(self.edges, canonical, options['type_order'], endpoints, self.target_type)
        self.observation = None

    def load_model(self, checkpoint_path, decision, *, device):
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        return ExplainedModel.from_state(checkpoint, device=device).seal()

    def prepare_context(self, center, *, split):
        if self.allowed_ids.get(str(center)) != split:
            raise ValueError('Center/split absent from the explicit experiment manifest')
        self.center = str(center)
        self.observation = None
        local = self.incoming.extract(self.target_type, int(center), int(self.options['subgraph_layers']))
        self.domains = self.rf.domains(int(center))
        keys = {e['canonical_key'] for e in self.domains['U123']}
        columns = {}
        for r, cols in local['columns'].items():
            name, swap = self.canonical[r]
            for i, j in enumerate(cols):
                s, d = self.edges[r][:, j]
                a, b = (d, s) if swap else (s, d)
                k = self.rf.pair_to_edge((name, int(a), int(b)), 1)['canonical_key']
                if k in keys:
                    columns.setdefault(k, []).append((r, i))
        batch = GraphBatch({t: x[local['nodes'][t]] for t, x in self.features.items()},
                           {r: torch.from_numpy(a) for r, a in local['edges'].items()}, self.target_type,
                           torch.tensor([local['center_index']]))
        if keys != set(columns):
            raise ValueError('Canonical legal unit without all message columns')
        self.current = CertificationContext(batch.to(self.device), center, CanonicalEdgeDomain(tuple(sorted(keys))),
                                            {k: tuple(v) for k, v in columns.items()})
        return self.current

    def accept_observation(self, observation):
        if observation.deleted:
            raise ValueError('Only this request original observation supplies Region inputs')
        self.observation = observation
        self.q0 = observation.response()

    def region_edges(self):
        return ([dict(e, src_row=int(e['src_id']), dst_row=int(e['dst_id'])) for e in self.domains['U123']],
                self.target_type, int(self.center))


def graph_schema(resources, options):
    raw = torch.load(Path(resources['data']) / options['graph'], map_location='cpu', weights_only=True,
                     mmap=True)
    counts = {t: int(x.shape[0]) for t, x in raw['features'].items()}
    edges = {}
    for relation, name, swap in options['canonical_relations']:
        if not swap:
            source, destination = options['edge_endpoints'][name]
            array = raw['edges'][tuple(relation)].numpy()
            edges[(source, name, destination)] = (array[0], array[1])
    target = options['target_type']
    return GraphSchema(counts, edges, target, tuple(str(i) for i in range(counts[target])))
