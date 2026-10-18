from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from comce.stage1.signatures import SignatureContext
from comce.stage2.operators import FeatureProgram, partition_keyed_groups


@dataclass
class RegionInputs:
    fields: dict


class RegionRecord(dict):
    signature_context: SignatureContext


@lru_cache(maxsize=1)
def region_program():
    return yaml.safe_load((Path(__file__).with_name('region_program.yaml')).read_text())


class RegionConstructor:
    def __init__(self):
        self.spec = region_program()

    def prepare(self, inputs):
        program = FeatureProgram({**self.spec['constants'], **inputs.fields})
        for name, expression in self.spec['program'].items():
            program.values[name] = program.evaluate(expression)
        return program

    def partition(self, program):
        keyed_views = ((view, program.evaluate(view['key'])) for view in self.spec['views'])
        return partition_keyed_groups(keyed_views, program.values['hops'], program.values['center_id'])

    def construct(self, inputs):
        program = self.prepare(inputs)
        communities = self.partition(program)
        signatures, features = [], []
        for community in communities:
            local = FeatureProgram({**program.values, 'mi': community['member_indices'], 'size': len(community['member_indices'])})
            local.execute(self.spec['community_program'])
            signatures.append(local.evaluate(self.spec['signature_output']))
            features.append(local.evaluate(self.spec['feature_output']))
        program.values.update(communities=communities, community_sigs=np.asarray(signatures, dtype=np.int64),
                              community_feats=np.asarray(features, dtype=np.float32))
        keys = [str(e['canonical_key']) for e in program.values['edges']]
        witness = set(map(str, inputs.fields.get('witness_keys', [])))
        indices = sorted(i for i, key in enumerate(keys) if key in witness)
        program.values.update(witness_keys=[keys[i] for i in indices], witness_indices=indices)
        record = RegionRecord({key: program.evaluate(expression) for key, expression in self.spec['outputs'].items()})
        if not np.isfinite(record['edge_rows']).all() or not np.isfinite(record['community_feats']).all():
            raise ValueError('Nonfinite Stage2 features')
        spaces = dict(self.spec['coordinates'])
        record.signature_context = SignatureContext(
            edge_keys=tuple(keys), center=program.values['cs'], edge_tokens=program.values['ts'],
            coordinate_spaces=spaces,
            base_signatures={k: program.values[k] for k in ('ss', 'ds')},
            region_members={str(c['community_id']): tuple(keys[i] for i in c['member_indices']) for c in communities},
            base_coordinate_spaces={k: spaces['center'] for k in ('ss', 'ds')},
            representations=self.spec.get('representations', {}))
        record.signature_context.validate(keys)
        return record


def region_inputs(*, center_id, center_type, center_row, edges, bank, relation_names, probabilities, prediction,
                  class_names=None, witness_keys=()):
    edges = sorted((dict(e) for e in edges), key=lambda e: str(e['canonical_key']))
    keys = [str(e['canonical_key']) for e in edges]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate canonical edge in Region inputs')
    relation_names = list(relation_names)
    probabilities = np.asarray(probabilities, np.float64).reshape(-1)
    class_names = [str(c) for c in (class_names or range(len(probabilities)))]
    if len(class_names) != len(probabilities):
        raise ValueError('Class vocabulary and probability dimensions disagree')
    source = bank.coordinates_for([e['src_type'] for e in edges], [int(e['src_row']) for e in edges])
    target = bank.coordinates_for([e['dst_type'] for e in edges], [int(e['dst_row']) for e in edges])
    fields = dict(center_id=center_id, center_type=str(center_type),
                  center_z=bank.coordinates_for([center_type], [int(center_row)])[0],
                  source_z=source.reshape(len(edges), -1), target_z=target.reshape(len(edges), -1), edges=edges,
                  hops=np.asarray([int(e['hop']) for e in edges], np.int64),
                  relations=np.asarray([relation_names.index(e['edge_type']) for e in edges], np.int64),
                  relation_names=relation_names, cutpoints=bank.thresholds, probabilities=probabilities,
                  prediction=int(prediction), prediction_label=class_names[int(prediction)], num_classes=len(class_names),
                  witness_keys=list(witness_keys))
    return RegionInputs(fields)


def build_record(io, bank, relation_names, witness_keys=()):
    """Region record for the request currently held by a graph adapter (after its q0 observation)."""
    edges, center_type, center_row = io.region_edges()
    inputs = region_inputs(center_id=io.center, center_type=center_type, center_row=center_row, edges=edges, bank=bank,
                           relation_names=relation_names, probabilities=io.q0['probs'], prediction=io.q0['pred'],
                           witness_keys=witness_keys)
    return RegionConstructor().construct(inputs)
