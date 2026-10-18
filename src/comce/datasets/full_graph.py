"""Full-graph adapter: one typed graph whose target vertices connect to attribute vertices.

Every query runs M on the whole graph. The legal domain of a target is the 3-hop receptive field through the
shared attribute vertices.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from comce.datasets.oracle import CanonicalEdgeDomain, CertificationContext, ExplainedModel, GraphBatch
from comce.datasets.schema import GraphSchema


class FullGraphIO:
    def __init__(self, resources, options, *, device, allowed_ids, identity=None):
        self.options = options
        self.device = device
        self.allowed_ids = {str(c): str(s) for c, s in allowed_ids.items()}
        self.types = tuple(options['node_types'])
        self.edge_types = tuple(options['edge_types'])
        self.relations = tuple(tuple(r) for r in options['message_relations'])
        count = len(self.edge_types)
        if len(self.relations) != 2 * count or len(self.types) != count + 1:
            raise ValueError('Full-graph schema needs one forward and one reverse message relation per edge type')
        graph = torch.load(Path(resources['data']) / options['graph'], map_location='cpu', weights_only=False)
        self.identity = options['graph_identity']
        self.feature_schema = options['feature_schema_sha']
        self.node_ids = graph['node_ids']
        self.offsets = graph['offsets']
        self.src, self.dst, self.rel = (np.asarray(graph[k], np.int64) for k in ('src', 'dst', 'rel'))
        self.attributes = np.asarray(graph[options['attribute_index']], np.int64)
        self.n = len(graph['y'])
        target = self.types[0]
        self.ids = list(map(int, self.node_ids[target]))
        self.local = {v: i for i, v in enumerate(self.ids)}
        self.input_dims = {t: int(np.asarray(graph['x'][t]).shape[1]) for t in self.types}
        self.x = {t: torch.as_tensor(np.asarray(graph['x'][t]), dtype=torch.float32, device=device) for t in self.types}
        target_types = [self.types[int(r) + 1] for r in self.rel]
        self.dst_rows = np.asarray([d - self.offsets[t] for d, t in zip(self.dst, target_types)], np.int64)
        self.keys = [f'{self.edge_types[int(r)]}:{self.ids[int(c)]}:{self.node_ids[t][int(d)]}'
                     for c, d, r, t in zip(self.src, self.dst_rows, self.rel, target_types)]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError('Duplicate canonical edge')
        self.edges, self.columns = {}, {}
        for r in range(count):
            positions = np.flatnonzero(self.rel == r)
            pair = torch.as_tensor(np.stack([self.src[positions], self.dst_rows[positions]]), dtype=torch.long, device=device)
            self.edges[self.relations[r]] = pair
            self.edges[self.relations[r + count]] = pair.flip(0)
            for column, position in enumerate(positions):
                self.columns[self.keys[position]] = ((self.relations[r], column), (self.relations[r + count], column))
        self.edges = {relation: self.edges[relation] for relation in self.relations}
        order = np.argsort(self.attributes.ravel(), kind='stable')
        attrs = self.attributes.ravel()[order]
        targets = np.repeat(np.arange(self.n), self.attributes.shape[1])[order]
        starts = np.flatnonzero(np.r_[True, attrs[1:] != attrs[:-1]])
        self.reverse = {int(attrs[s]): targets[s:e] for s, e in zip(starts, np.r_[starts[1:], len(attrs)])}
        self.observation = None

    def load_model(self, checkpoint_path, decision, *, device):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['graph'] != self.identity or checkpoint['feature_schema'] != self.feature_schema:
            raise ValueError('GNN checkpoint graph/feature binding mismatch')
        gnn = self.options['gnn']
        architecture = dict(input_dims=dict(self.input_dims), relations=[list(r) for r in self.relations],
                            target_type=self.types[0], num_classes=int(decision['num_classes']), gnn_type=checkpoint['kind'],
                            hidden=int(gnn['hidden']), layers=int(gnn['layers']), dropout=float(gnn['dropout']),
                            heads=int(gnn['heads']))
        model = ExplainedModel(architecture, dict(kind=decision['kind'], num_classes=int(decision['num_classes']),
                                                  threshold=float(decision['threshold']),
                                                  positive_class=int(decision['positive_class'])),
                               decision['class_vocabulary'], deployment_temperature=float(decision['deployment_temperature']))
        model.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
        return model.to(device).seal()

    def domain(self, center):
        near = {int(center)}
        for attribute in self.attributes[center]:
            near.update(self.reverse[int(attribute)].tolist())
        targets = np.asarray(sorted(near), np.int64)
        positions = np.concatenate([targets + r * self.n for r in range(len(self.edge_types))])
        hops = np.where(self.src[positions] == center, 1,
                        np.where(np.isin(self.dst[positions], self.attributes[center]), 2, 3)).astype(np.int64)
        return positions, hops

    def prepare_context(self, center, *, split):
        if self.allowed_ids.get(str(center)) != split:
            raise ValueError('Center/split absent from the explicit experiment manifest')
        self.center = str(center)
        self.center_local = int(self.local[int(center)])
        self.positions, self.hops = self.domain(self.center_local)
        keys = [self.keys[p] for p in self.positions]
        batch = GraphBatch(self.x, self.edges, self.types[0], torch.tensor([self.center_local], device=self.device))
        fingerprint = hashlib.sha256(json.dumps(dict(graph=self.identity, center=self.center, split=split,
                                                     domain=hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()),
                                                sort_keys=True).encode()).hexdigest()
        self.observation = None
        self.current = CertificationContext(batch, self.center, fingerprint, CanonicalEdgeDomain(tuple(sorted(keys))),
                                            {k: self.columns[k] for k in keys})
        return self.current

    def accept_observation(self, observation):
        if observation.context_fingerprint != self.current.graph_fingerprint or observation.deleted:
            raise ValueError('Only this request original observation supplies Region inputs')
        self.observation = observation
        self.q0 = observation.response()

    def region_edges(self):
        edges = []
        for position, hop in zip(self.positions, self.hops):
            r = int(self.rel[position])
            target = self.types[r + 1]
            row = int(self.dst_rows[position])
            edges.append(dict(canonical_key=self.keys[position], edge_type=self.edge_types[r], src_type=self.types[0],
                              src_id=str(self.ids[int(self.src[position])]), dst_type=target,
                              dst_id=str(self.node_ids[target][row]), hop=int(hop),
                              src_row=int(self.src[position]), dst_row=row))
        return edges, self.types[0], self.center_local


def graph_schema(resources, options):
    graph = torch.load(Path(resources['data']) / options['graph'], map_location='cpu', weights_only=False)
    types, edge_types = tuple(options['node_types']), tuple(options['edge_types'])
    src, dst, rel = (np.asarray(graph[k], np.int64) for k in ('src', 'dst', 'rel'))
    edges = {}
    for r, name in enumerate(edge_types):
        mask = rel == r
        edges[(types[0], name, types[r + 1])] = (src[mask] - graph['offsets'][types[0]], dst[mask] - graph['offsets'][types[r + 1]])
    return GraphSchema({t: len(graph['node_ids'][t]) for t in types}, edges, types[0],
                       tuple(map(str, graph['node_ids'][types[0]])))
