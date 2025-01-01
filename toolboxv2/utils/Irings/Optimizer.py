import numpy as np
from collections import defaultdict

from .zero import IntelligenceRing, Concept
from typing import List

from sklearn.cluster import DBSCAN
import networkx as nx


class TopologyOptimizer:
    def __init__(self, network: 'NetworkManager'):
        self.network = network
        self.graph = nx.Graph()

    def optimize(self):
        self._build_graph()
        self._optimize_connections()
        self._update_network()

    def _build_graph(self):
        # Create graph from current connections
        for source, connections in self.network.connections.items():
            for target, weight in connections.items():
                self.graph.add_edge(source, target, weight=weight)

    def _optimize_connections(self):
        # Optimize using minimum spanning tree with additional edges
        mst = nx.minimum_spanning_tree(self.graph)

        # Add high-weight non-MST edges
        non_mst_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if not mst.has_edge(u, v)
        ]

        sorted_edges = sorted(
            non_mst_edges,
            key=lambda x: x[2]['weight'],
            reverse=True
        )

        for u, v, d in sorted_edges[:len(self.graph.nodes)]:
            mst.add_edge(u, v, weight=d['weight'])

        self.graph = mst

    def _update_network(self):
        # Update network connections based on optimized graph
        new_connections = defaultdict(dict)

        for u, v, d in self.graph.edges(data=True):
            new_connections[u][v] = d['weight']
            new_connections[v][u] = d['weight']

        self.network.connections = new_connections

        # Update ring adapters
        for ring_id, ring in self.network.rings.items():
            ring.adapter.connected_rings = {
                k: self.network.rings[k].adapter
                for k in new_connections[ring_id]
            }


class RingRestructurer:
    def __init__(self, ring: IntelligenceRing):
        self.ring = ring

    def restructure(self):
        concepts = list(self.ring.concept_graph.concepts.values())
        vectors = np.array([c.vector for c in concepts])

        # Cluster concepts
        clusters = self._cluster_concepts(vectors)

        # Reorganize based on clusters
        self._reorganize_concepts(concepts, clusters)

    def _cluster_concepts(self, vectors: np.ndarray) -> np.ndarray:
        clustering = DBSCAN(eps=0.3, min_samples=2)
        return clustering.fit_predict(vectors)

    def _reorganize_concepts(self, concepts: List[Concept], clusters: np.ndarray):
        for i, cluster_id in enumerate(clusters):
            if cluster_id == -1:
                continue

            concept = concepts[i]
            cluster_concepts = [
                concepts[j] for j, c in enumerate(clusters)
                if c == cluster_id
            ]

            # Update relations within cluster
            concept.relations = {
                c.id: np.dot(concept.vector, c.vector)
                for c in cluster_concepts
                if c.id != concept.id
            }
