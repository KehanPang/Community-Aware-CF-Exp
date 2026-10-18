from comce.datasets import citation_graph, full_graph, typed_graph
from comce.datasets.citation_graph import CitationGraphIO
from comce.datasets.full_graph import FullGraphIO
from comce.datasets.typed_graph import TypedGraphIO

# Input adapters only translate a graph format into canonical edges, message columns and M's forward pass.
GRAPH_ADAPTERS = {'full_graph': FullGraphIO, 'citation_graph': CitationGraphIO, 'typed_graph': TypedGraphIO}
# The same formats read as a schema (node types, typed edges, target IDs) for Stage 1, without M.
GRAPH_SCHEMAS = {'full_graph': full_graph.graph_schema, 'citation_graph': citation_graph.graph_schema,
                 'typed_graph': typed_graph.graph_schema}
