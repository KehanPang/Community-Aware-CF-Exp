"""Citation-graph adapter: papers cite papers and are written by authors; M runs on the 3-hop citation context."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from comce.datasets.oracle import CanonicalEdgeDomain, CertificationContext, ExplainedModel, GraphBatch
from comce.datasets.schema import GraphSchema

CITES, WRITTEN_BY = 'paper_cites_paper', 'paper_written_by_author'


class CitationGraph:
    def __init__(self, root, options):
        self.root = Path(root)
        self.meta = pd.read_parquet(self.root / options['metadata'])
        self.ids = self.meta.paper_id.astype(str).tolist()
        self.index = {c: i for i, c in enumerate(self.ids)}
        self.authors = torch.load(self.root / options['author_ids'], map_location='cpu', weights_only=True)
        arrays = np.load(self.root / options['graph_arrays'])
        self.cp, self.ci, self.ap, self.ai = arrays['cite_ptr'], arrays['cite_idx'], arrays['author_ptr'], arrays['author_idx']
        self.paper_inputs = np.load(self.root / options['paper_inputs'], mmap_mode='r')
        self.author_inputs = np.load(self.root / options['author_inputs'], mmap_mode='r')

    def context(self, center):
        c = self.index[str(center)]
        depth = {c: 0}
        frontier = [c]
        edges = {}
        for hop in range(1, 4):
            nxt = []
            for paper in frontier:
                for ref in self.ci[self.cp[paper]:self.cp[paper + 1]]:
                    ref = int(ref)
                    edges.setdefault((0, paper, ref), hop)
                    if ref not in depth:
                        depth[ref] = hop
                        nxt.append(ref)
                for author in self.ai[self.ap[paper]:self.ap[paper + 1]]:
                    edges.setdefault((1, paper, int(author)), hop)
            frontier = nxt
        papers = sorted(depth)
        authors = sorted({v for t, p, v in edges if t == 1})
        pi = {v: i for i, v in enumerate(papers)}
        ai = {v: i for i, v in enumerate(authors)}
        rows = []
        for (kind, paper, other), hop in edges.items():
            key = (f'{CITES}:{self.ids[paper]}:{self.ids[other]}' if kind == 0
                   else f'{WRITTEN_BY}:{self.ids[paper]}:{self.authors[other]}')
            rows.append((key, kind, pi[other] if kind == 0 else ai[other], pi[paper], hop))
        rows.sort()
        return dict(center_id=str(center), center_index=pi[c], paper_indices=np.asarray(papers, np.int64),
                    author_indices=np.asarray(authors, np.int64), edge_keys=[r[0] for r in rows],
                    kind=np.asarray([r[1] for r in rows], np.int64), src=np.asarray([r[2] for r in rows], np.int64),
                    dst=np.asarray([r[3] for r in rows], np.int64), hops=np.asarray([r[4] for r in rows], np.int64))

    def collate(self, context):
        arrays = dict(paper_x=np.asarray(self.paper_inputs[context['paper_indices']], np.float32),
                      author_x=np.asarray(self.author_inputs[context['author_indices']], np.float32),
                      kind=context['kind'], src=context['src'], dst=context['dst'],
                      centers=np.asarray([context['center_index']], np.int64))
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}


class CitationGraphIO:
    def __init__(self, resources, options, *, device, allowed_ids):
        self.device = device
        self.allowed_ids = {str(c): str(s) for c, s in allowed_ids.items()}
        self.graph = CitationGraph(Path(resources['data']), options)
        self.relations = None
        self.observation = None

    def load_model(self, checkpoint_path, decision, *, device):
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        model = ExplainedModel.from_state(checkpoint, device=device).seal()
        self.relations = [tuple(r) for r in checkpoint['architecture']['relations']]
        return model

    def prepare_context(self, center, *, split):
        if self.allowed_ids.get(str(center)) != split:
            raise ValueError('Center/split absent from the explicit experiment manifest')
        self.center = str(center)
        self.observation = None
        if str(self.graph.meta.iloc[self.graph.index[center]]['split']) != split:
            raise ValueError('Native split mismatch')
        self.context = self.graph.context(center)
        b = self.graph.collate(self.context)
        batch = GraphBatch({'paper': b['paper_x'], 'author': b['author_x']},
                           {r: torch.stack([b['src'][b['kind'] == i], b['dst'][b['kind'] == i]])
                            for i, r in enumerate(self.relations)}, 'paper', b['centers'])
        columns = {}
        for i, r in enumerate(self.relations):
            positions = np.flatnonzero(self.context['kind'] == i)
            for col, j in enumerate(positions):
                columns[self.context['edge_keys'][j]] = [(r, col)]
        keys = set(self.context['edge_keys'])
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
        ctx = self.context
        edges = []
        for key, kind, s, d, hop in zip(ctx['edge_keys'], ctx['kind'], ctx['src'], ctx['dst'], ctx['hops']):
            typ, a, b = key.split(':')
            cited = int(kind) == 0
            edges.append(dict(canonical_key=key, edge_type=typ, src_type='paper', src_id=a,
                              dst_type='paper' if cited else 'author', dst_id=b, hop=int(hop),
                              src_row=int(ctx['paper_indices'][d]),
                              dst_row=int((ctx['paper_indices'] if cited else ctx['author_indices'])[s])))
        return edges, 'paper', int(ctx['paper_indices'][ctx['center_index']])


def graph_schema(resources, options):
    root = Path(resources['data'])
    ids = pd.read_parquet(root / options['metadata'], columns=['paper_id']).paper_id.astype(str).tolist()
    authors = torch.load(root / options['author_ids'], map_location='cpu', weights_only=True)
    arrays = np.load(root / options['graph_arrays'])

    def csr(pointer, index):
        return np.repeat(np.arange(len(pointer) - 1), np.diff(pointer)), np.asarray(index, np.int64)
    edges = {('paper', CITES, 'paper'): csr(arrays['cite_ptr'], arrays['cite_idx']),
             ('paper', WRITTEN_BY, 'author'): csr(arrays['author_ptr'], arrays['author_idx'])}
    return GraphSchema({'paper': len(ids), 'author': len(authors)}, edges, 'paper', tuple(ids))
