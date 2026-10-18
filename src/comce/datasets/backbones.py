import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv, HGTConv

STD_TYPES = frozenset({'std_gcn', 'std_graphsage', 'std_hgt'})


class StandardBackbone(nn.Module):
    def __init__(self, *, gnn_type, input_dims, relations, target_type,
                 num_classes, hidden=128, layers=3, dropout=0.2, heads=4):
        super().__init__()
        if gnn_type not in STD_TYPES or layers != 3:
            raise ValueError('Exactly three layers of a declared standard backbone required')
        self.gnn_type = gnn_type
        self.node_types = tuple(input_dims)
        self.relations = tuple(tuple(r) for r in relations)
        self.target_type = target_type
        self.dropout = float(dropout)
        self.encoders = nn.ModuleDict({t: nn.Sequential(
            nn.Linear(input_dims[t], hidden), nn.LayerNorm(hidden), nn.GELU())
            for t in self.node_types})
        convs = []
        for _ in range(layers):
            if gnn_type == 'std_gcn':
                conv = GCNConv(hidden, hidden, cached=False, add_self_loops=True)
            elif gnn_type == 'std_graphsage':
                conv = SAGEConv(hidden, hidden, aggr='mean', root_weight=True)
            else:
                conv = HGTConv(hidden, hidden, (list(self.node_types), list(self.relations)),
                               heads=heads)
            convs.append(conv)
        self.layers = nn.ModuleList(convs)
        self.norms = nn.ModuleList(nn.ModuleDict({t: nn.LayerNorm(hidden)
                                                  for t in self.node_types})
                                   for _ in range(layers))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                        nn.Linear(hidden, num_classes))

    def forward(self, graph):
        if set(graph.x) != set(self.node_types):
            raise ValueError('Node schema mismatch')
        h = {t: self.encoders[t](graph.x[t]) for t in self.node_types}
        if self.gnn_type != 'std_hgt':
            offsets, total = {}, 0
            for t in self.node_types:
                offsets[t] = total
                total += len(h[t])
            parts = [graph.edges[r] + graph.edges[r].new_tensor([[offsets[r[0]]], [offsets[r[2]]]])
                     for r in self.relations]
            edges = torch.cat(parts, dim=1)
        for conv, norms in zip(self.layers, self.norms, strict=True):
            if self.gnn_type == 'std_hgt':
                out = conv(h, graph.edges)
                h = {t: F.dropout(norms[t](h[t] if t not in out else out[t]),
                                  p=self.dropout, training=self.training)
                     for t in self.node_types}
            else:
                flat = conv(torch.cat([h[t] for t in self.node_types]), edges)
                out = {t: flat[offsets[t]:offsets[t] + len(h[t])] for t in self.node_types}
                h = {t: F.dropout(F.gelu(norms[t](out[t])), p=self.dropout,
                                  training=self.training)
                     for t in self.node_types}
        return self.classifier(h[self.target_type]), h
