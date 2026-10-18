from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DIMENSIONS = 8
BUCKETS = 16
SCHEMA = 'comce-signature-bank'


def random_directions(input_dim, seed, dimensions=DIMENSIONS):
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(0.0, 1.0, size=(int(input_dim), int(dimensions)))
    return directions / np.linalg.norm(directions, axis=0, keepdims=True)


def project_rows(matrix, directions, chunk=65536):
    out = np.empty((matrix.shape[0], directions.shape[1]), np.float64)
    for begin in range(0, matrix.shape[0], chunk):
        out[begin:begin + chunk] = np.asarray(matrix[begin:begin + chunk], np.float64) @ directions
    return out


def quantile_thresholds(coordinates, buckets=BUCKETS):
    levels = np.linspace(0.0, 1.0, int(buckets) + 1)[1:-1]
    return np.stack([np.quantile(coordinates[:, j], levels) for j in range(coordinates.shape[1])])


class SignatureBank:
    def __init__(self, coordinates, thresholds, provenance):
        self.coordinates = {str(t): np.asarray(v, np.float32) for t, v in coordinates.items()}
        self.thresholds = np.asarray(thresholds, np.float64)
        self.provenance = dict(provenance)
        if self.thresholds.shape != (DIMENSIONS, BUCKETS - 1):
            raise ValueError('Signature thresholds must be one quantile row per projected coordinate')
        if any(v.ndim != 2 or v.shape[1] != DIMENSIONS or not np.isfinite(v).all() for v in self.coordinates.values()):
            raise ValueError('Signature coordinates must be finite projected node coordinates')

    def coordinates_for(self, node_types, rows):
        rows = np.asarray(rows, np.int64).reshape(-1)
        node_types = np.broadcast_to(np.asarray(node_types).astype(str), rows.shape)
        out = np.empty((len(rows), DIMENSIONS), np.float64)
        for node_type in np.unique(node_types):
            mask = node_types == node_type
            out[mask] = self.coordinates[str(node_type)][rows[mask]]
        return out

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(schema=SCHEMA, coordinates={t: torch.from_numpy(v) for t, v in self.coordinates.items()},
                        thresholds=torch.from_numpy(self.thresholds), provenance=self.provenance), path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('schema') != SCHEMA:
            raise ValueError('Not a ComCE signature bank')
        return cls({t: v.numpy() for t, v in payload['coordinates'].items()}, payload['thresholds'].numpy(),
                   payload['provenance'])


@dataclass(frozen=True)
class SignatureContext:
    edge_keys: tuple[str, ...]
    center: np.ndarray
    edge_tokens: np.ndarray
    coordinate_spaces: dict[str, str]
    base_signatures: dict[str, np.ndarray] = field(default_factory=dict)
    region_members: dict[str, tuple[str, ...]] = field(default_factory=dict)
    base_coordinate_spaces: dict[str, str] = field(default_factory=dict)
    representations: dict[str, str] = field(default_factory=dict)

    def validate(self, requested_keys):
        legal = set(self.edge_keys)
        tokens = np.asarray(self.edge_tokens)
        center = np.asarray(self.center)
        if len(legal) != len(self.edge_keys) or tokens.ndim != 2 or len(tokens) != len(self.edge_keys):
            raise ValueError('SignatureContext edge/token alignment mismatch')
        if center.ndim != 1 or not self.coordinate_spaces.get('center') or not self.coordinate_spaces.get('token'):
            raise ValueError('SignatureContext requires explicit center and token coordinates')
        if set(self.base_signatures) != set(self.base_coordinate_spaces):
            raise ValueError('Base Signature coordinates must be explicit')
        arrays = [center, tokens, *self.base_signatures.values()]
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError('Nonfinite SignatureContext')
        if not set(requested_keys) <= legal or any(not set(members) <= legal for members in self.region_members.values()):
            raise ValueError('SignatureContext contains an unknown edge')

    def statistics(self, requested_keys):
        self.validate(requested_keys)
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(edges=self.edge_keys, spaces=self.coordinate_spaces,
                                      regions=self.region_members, base_spaces=self.base_coordinate_spaces,
                                      representations=self.representations), sort_keys=True).encode())
        for name, values in [('center', self.center), ('edge_tokens', self.edge_tokens), *sorted(self.base_signatures.items())]:
            array = np.ascontiguousarray(values)
            digest.update(name.encode())
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        return dict(identity=digest.hexdigest(), coordinate_spaces=dict(self.coordinate_spaces),
                    center_shape=list(np.shape(self.center)), edge_token_shape=list(np.shape(self.edge_tokens)),
                    regions=len(self.region_members), base_coordinate_spaces=dict(self.base_coordinate_spaces),
                    representations=dict(self.representations))
