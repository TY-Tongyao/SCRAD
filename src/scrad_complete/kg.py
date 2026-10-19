from collections import OrderedDict
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path

class SequenceKG:

    def __init__(self, sequences, centers):
        sequences, centers = (np.asarray(sequences), np.asarray(centers))
        if len(sequences) != len(centers):
            raise ValueError('One center is required for every sequence')
        self.entities, inverse = np.unique(centers, return_inverse=True)
        self.lookup = {int(node): i for i, node in enumerate(self.entities)}
        valid = (sequences >= 0) & (sequences != centers[:, None])
        row = np.broadcast_to(inverse[:, None], sequences.shape)[valid]
        col = sequences[valid]
        nattr = int(sequences.max(initial=-1)) + 1
        self.incidence = sparse.csr_matrix((np.ones(len(row), np.int32), (row, col)), shape=(len(self.entities), nattr))
        self.incidence.data[:] = 1
        self.by_attribute = self.incidence.tocsc()
        bipartite = sparse.bmat([[None, self.incidence], [self.incidence.T, None]], format='csr')
        self.bipartite = bipartite
        self.path_cache = OrderedDict()
        _, components = connected_components(bipartite, directed=False)
        self.components = components[:len(self.entities)]

    def shortest_paths(self, query_nodes, center, references):
        center = int(center)
        refs = list(dict.fromkeys((int(x) for x in references)))
        result = {ref: {'reachable': False, 'entities': [], 'attributes': []} for ref in refs}
        if center in result:
            result[center] = {'reachable': True, 'entities': [center], 'attributes': []}
        attributes = {int(x) for x in query_nodes if x >= 0 and x != center}
        origin = self.lookup.get(center)
        if origin is not None:
            lo, hi = self.incidence.indptr[origin:origin + 2]
            attributes.update(self.incidence.indices[lo:hi].tolist())
        bridges = {}
        for attribute in sorted(attributes):
            if attribute >= self.by_attribute.shape[1]:
                continue
            lo, hi = self.by_attribute.indptr[attribute:attribute + 2]
            for entity in self.by_attribute.indices[lo:hi]:
                entity = int(entity)
                if entity != origin:
                    bridges.setdefault(entity, attribute)
        neighbors = np.asarray(sorted(bridges), np.int32)
        reachable_components = {int(self.components[i]) for i in neighbors}
        if origin is not None:
            reachable_components.add(int(self.components[origin]))
        for ref in refs:
            root = self.lookup.get(ref)
            if ref == center or root is None or int(self.components[root]) not in reachable_components:
                continue
            if root not in self.path_cache:
                distance, predecessor = shortest_path(self.bipartite, directed=False, indices=root, return_predecessors=True, unweighted=True, method='D')
                self.path_cache[root] = (distance, predecessor)
                if len(self.path_cache) > 64:
                    self.path_cache.popitem(last=False)
            self.path_cache.move_to_end(root)
            distance, predecessor = self.path_cache[root]
            current = origin
            best = distance[origin] if origin is not None else np.inf
            extra_attribute = None
            if len(neighbors):
                chosen = int(neighbors[np.argmin(distance[neighbors])])
                if distance[chosen] + 2 < best:
                    current, best, extra_attribute = (chosen, distance[chosen] + 2, bridges[chosen])
            if not np.isfinite(best):
                continue
            vertices = [center]
            attributes_on_path = []
            if extra_attribute is not None:
                vertices.append(int(self.entities[current]))
                attributes_on_path.append(extra_attribute)
            while current != root:
                attribute_vertex = int(predecessor[current])
                current = int(predecessor[attribute_vertex])
                attributes_on_path.append(attribute_vertex - len(self.entities))
                vertices.append(int(self.entities[current]))
            result[ref] = {'reachable': True, 'entities': vertices, 'attributes': attributes_on_path}
        return result

    @staticmethod
    def verbalize(path):
        if not path['reachable']:
            return 'No connecting path in the sequence KG.'
        entities, attributes = (path['entities'], path['attributes'])
        if not attributes:
            return f'Target and reference denote the same entity {entities[0]}.'
        return '; '.join((f'Entity {a} shares attribute node {attribute} with entity {b}' for a, attribute, b in zip(entities[:-1], attributes, entities[1:]))) + '.'
