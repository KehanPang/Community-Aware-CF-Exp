"""The explained GNN M behind its classification interface and the per-request exact oracle."""
from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

import torch
from torch import nn

from comce.datasets.backbones import STD_TYPES, StandardBackbone
from comce.datasets.decision import DecisionSpec


@dataclass(frozen=True)
class GraphBatch:
    x: Mapping[str, torch.Tensor]
    edges: Mapping[tuple, torch.Tensor]
    target_type: str
    targets: torch.Tensor
    lookup_ids: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device):
        return GraphBatch({k: v.to(device) for k, v in self.x.items()},
                          {k: v.to(device) for k, v in self.edges.items()}, self.target_type,
                          self.targets.to(device), {k: v.to(device) for k, v in self.lookup_ids.items()})


@dataclass(frozen=True)
class CanonicalEdgeDomain:
    keys: tuple[str, ...]

    def validate(self, deleted):
        result = tuple(sorted(map(str, deleted)))
        if len(set(result)) != len(result) or not set(result).issubset(self.keys):
            raise ValueError('Duplicate or out-of-domain canonical edge')
        return result


@dataclass(frozen=True)
class CertificationContext:
    graph: GraphBatch
    center_id: str
    graph_fingerprint: str
    domain: CanonicalEdgeDomain
    columns: Mapping[str, tuple[tuple[tuple, int], ...]]

    def delete(self, keys):
        keys = self.domain.validate(keys)
        remove = {r: [] for r in self.graph.edges}
        for key in keys:
            for relation, column in self.columns[key]:
                remove[relation].append(column)
        edges = {}
        for relation, edge in self.graph.edges.items():
            keep = torch.ones(edge.shape[1], dtype=torch.bool, device=edge.device)
            if remove[relation]:
                keep[remove[relation]] = False
            edges[relation] = edge[:, keep]
        return GraphBatch(self.graph.x, edges, self.graph.target_type, self.graph.targets, self.graph.lookup_ids)


@dataclass
class GNNOutput:
    raw_logits: torch.Tensor
    deployed_logits: torch.Tensor
    probabilities: torch.Tensor
    prediction: torch.Tensor


@dataclass
class Observation:
    output: GNNOutput
    model_fingerprint: str
    context_fingerprint: str
    deleted: tuple[str, ...]
    runtime_record: dict | None = None

    def response(self, original=None):
        if self.runtime_record is not None:
            return self.runtime_record
        p = self.output.probabilities[0].detach().cpu().tolist()
        pred = int(self.output.prediction[0])
        before = pred if original is None else int(original.output.prediction[0])
        return dict(probs=p, pred=pred, original_pred=before, score=p[before], flip=bool(self.deleted and pred != before))


def state_fingerprint(config, state):
    h = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode())
    for key, value in sorted(state.items()):
        v = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class ExplainedModel(nn.Module):
    """A frozen GCN, GraphSAGE or HGT classifier. Only logits, probabilities and the decision leave this class."""

    def __init__(self, architecture, decision_spec, class_vocabulary, *, deployment_temperature=1.0):
        super().__init__()
        if deployment_temperature <= 0:
            raise ValueError('Positive deployment temperature required')
        self.architecture = dict(architecture)
        self.decision_spec = DecisionSpec(**decision_spec)
        if architecture['num_classes'] != self.decision_spec.num_classes:
            raise ValueError('Architecture and deployment class dimensions differ')
        self.class_vocabulary = tuple(map(str, class_vocabulary))
        if len(set(self.class_vocabulary)) != self.decision_spec.num_classes:
            raise ValueError('One ordered, unique native class vocabulary required')
        if architecture['gnn_type'] not in STD_TYPES:
            raise ValueError('Only the frozen standard GNN backbones are supported')
        self.backbone = StandardBackbone(**architecture)
        self.register_buffer('deployment_temperature', torch.tensor(float(deployment_temperature)))
        self._frozen_fingerprint = None

    def forward(self, graph_batch):
        if graph_batch.target_type != self.architecture['target_type']:
            raise ValueError('Target type mismatch')
        logits, _ = self.backbone(graph_batch)
        raw = logits[graph_batch.targets]
        deployed = raw / self.deployment_temperature
        probs = deployed.softmax(-1)
        return GNNOutput(raw, deployed, probs, self.decision_spec.prediction(probs))

    def metadata(self):
        return dict(format='comce_gnn_state_dict', architecture=self.architecture,
                    decision_spec=vars(self.decision_spec), class_vocabulary=list(self.class_vocabulary))

    def fingerprint(self):
        return self._frozen_fingerprint or state_fingerprint(self.metadata(), self.state_dict())

    @classmethod
    def from_state(cls, payload, *, device='cpu'):
        model = cls(payload['architecture'], payload['decision_spec'], payload['class_vocabulary'])
        model.load_state_dict(payload['state_dict'], strict=True)
        if state_fingerprint(model.metadata(), model.state_dict()) != payload['fingerprint']:
            raise ValueError('Checkpoint metadata/state fingerprint mismatch')
        return model.to(device)

    def seal(self):
        self.eval().requires_grad_(False)
        self._frozen_fingerprint = state_fingerprint(self.metadata(), self.state_dict())
        return self

    def explanation_eligible(self, observation):
        return self.decision_spec.kind != 'positive_to_negative' or bool(observation.output.prediction[0] == 1)


class ExactOracle:
    """Per-request real-GNN oracle: q0 is observed once, every intervention is one counted forward of M."""

    def __init__(self, query_cap=12):
        self.query_cap = query_cap
        self.logical = self.physical = 0
        self.q0 = None
        self.identity = None

    @torch.no_grad()
    def query(self, model, context, deleted):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Only sealed evaluation models may be certified')
        if context.graph.targets.numel() != 1:
            raise ValueError('One target per counted oracle context')
        deleted = context.domain.validate(deleted)
        identity = (model.fingerprint(), context.graph_fingerprint, context.center_id)
        if self.identity is not None and self.identity != identity:
            raise ValueError('Request-local oracle cannot cross model/context/center')
        self.identity = identity
        if not deleted and self.q0 is not None:
            return self.q0
        if deleted and self.q0 is None:
            raise ValueError('Observe q0 first')
        if self.logical >= self.query_cap:
            raise RuntimeError('Real-GNN query cap exhausted')
        self.logical += 1
        self.physical += 1
        output = model(context.delete(deleted) if deleted else context.graph)
        result = Observation(output, identity[0], context.graph_fingerprint, deleted)
        before = self.q0.output if self.q0 is not None else output
        p = output.probabilities[0].detach().cpu().tolist()
        pred = int(output.prediction[0])
        original = int(before.prediction[0])
        score_class = model.decision_spec.positive_class if model.decision_spec.kind == 'positive_to_negative' else original
        result.runtime_record = dict(probs=p, pred=pred, original_pred=original, score=p[score_class],
                                     flip=bool(deleted and model.decision_spec.event(before.probabilities, output.probabilities)[0]))
        if not deleted:
            self.q0 = result
        return result

    def observe(self, model, context):
        return self.query(model, context, ())

    def intervene(self, model, context, deleted):
        return self.query(model, context, deleted)
