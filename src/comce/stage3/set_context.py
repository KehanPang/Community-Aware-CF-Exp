import numpy as np
import torch

SLOTS = 10


class SetContext:
    """Edge rows x_e and set context c_{u,Delta} of the current search domain."""

    def __init__(self, record, domain, device='cpu'):
        self.keys = sorted(domain['macro_domain_edge_keys'])
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.domain = domain
        self.device = device
        self.m = len(self.keys)
        keys = list(map(str, record['edge_keys']))
        rows = np.array(record['edge_rows'], dtype=np.float32, copy=True)
        center = np.array(record['center_state'], dtype=np.float32, copy=True)
        if len(keys) != len(set(keys)) or len(keys) != len(rows):
            raise ValueError('Invalid exact feature rows')
        position = dict(zip(keys, range(len(keys)), strict=True))
        self.edge_in = rows.shape[1]
        self.set_dim = len(center) + 3
        raw = np.asarray([rows[position[k]] for k in self.keys], np.float32).reshape(self.m, self.edge_in)
        # The (m, SLOTS, d) layout reproduces the execution the frozen checkpoints were evaluated with.
        self.edge_rows = torch.as_tensor(raw, device=device)[:, None, :].expand(-1, SLOTS, -1)
        self.center = torch.as_tensor(center, device=device)
        self.universe_size = len(position)

    def set_features(self, indices, mask):
        size = mask.sum(1).float()
        n = len(indices)
        return torch.cat([self.center[None].expand(n, -1),
                          torch.stack([size / 10, size / max(1, self.universe_size),
                                       torch.full_like(size, np.log1p(self.universe_size))], 1)], 1)

    def pack(self, sets):
        if not sets:
            return (torch.empty((0, 1), dtype=torch.long, device=self.device),
                    torch.empty((0, 1), dtype=torch.bool, device=self.device))
        width = max(map(len, sets))
        a = np.zeros((len(sets), width), np.int64)
        mask = np.zeros_like(a, bool)
        for i, s in enumerate(sets):
            if not 0 < len(s) <= SLOTS or len(s) != len(set(s)):
                raise ValueError('Exact nonempty set <=10 required')
            a[i, :len(s)] = s
            mask[i, :len(s)] = True
        if a.max(initial=0) >= self.m or a.min(initial=0) < 0:
            raise ValueError('Candidate outside domain')
        return (torch.as_tensor(a, device=self.device), torch.as_tensor(mask, device=self.device))
