import abc
import array
from numbers import Number
from itertools import product
import random
import numpy as np
import bisect
from heapq import heappop, heappush
from functools import reduce
import networkx as nx
import sys
import os.path
from pathlib import Path
from enum import Enum
from multiprocessing import Process
import timeit

import shapely

from Anytime import AnytimeAlgorithm, PRINT

OUT_DIR = Path("./out")


class GraphChanges(abc.ABC):
    def __init__(self, orig_graph: nx.Graph):
        self.orig_graph = orig_graph

    def add_provider(self) -> Callable[[nx.Graph, list['DeltaType']], nx.Graph]:
        return self.add

    def remove_provider(self) -> Callable[[nx.Graph, list['DeltaType']], nx.Graph]:
        return self.remove

    def complete_graph(self):
        return nx.complete_graph(self.orig_graph.nodes)

    def empty_graph(self):
        return nx.empty_graph(n=0, create_using=self.orig_graph)

    @abc.abstractmethod
    def add(self, graph: nx.Graph, deltas: list['DeltaType']):
        pass

    def add_undo(self, graph: nx.Graph, deltas: list['DeltaType']):
        return self.remove(graph, deltas)

    @abc.abstractmethod
    def remove(self, graph: nx.Graph, deltas: list['DeltaType']):
        pass

    def remove_undo(self, graph: nx.Graph, deltas: list['DeltaType']):
        return self.add(graph, deltas)

    @abc.abstractmethod
    def get_deltas(self, big_graph=None, small_graph=None) -> list['DeltaType']:
        pass

def VertexSetConstraintChangesFactory(vertexSubsets):
    vertexSubsets = [frozenset(vertexSubset) for vertexSubset in vertexSubsets]
    from collections import Counter
    class VertexSetConstraintChanges(GraphChanges):
        origVertexSubsets = Counter(v for subset in vertexSubsets for v in subset)
        def add(self, input_graph: nx.Graph, deltas: list[int]):
            graph = input_graph.copy()
            # Removing a constraint - Adding vertices
            curVertexSubsets = Counter(v for subset in deltas for v in subset)
            removedVertices = set(self.origVertexSubsets - curVertexSubsets)
            remainingVertices = set(self.orig_graph.nodes) - removedVertices
            graph.add_nodes_from(remainingVertices)
            graph.add_edges_from(
                set(self.orig_graph.subgraph(remainingVertices).edges)
                - set(graph.edges)
            )
            return graph

        def remove(self, input_graph: nx.Graph, deltas: list[int]):
            graph = input_graph.copy()
            # Adding a Constraint - Removing vertices in constraint
            vertices = (vertex for subset in deltas for vertex in subset)
            graph.remove_nodes_from(vertices)
            return graph

        def empty_graph(self):
            remainingVertices = set(self.orig_graph.nodes) - set(self.origVertexSubsets)
            return nx.Graph(orig_graph.subgraph(remainingVertices))

        def get_deltas(self, big_graph=None, small_graph=None):
            if big_graph is None:
                big_graph = self.orig_graph
            if small_graph is None:
                small_graph = nx.Graph()

            small_graph_nodes = set(small_graph.nodes)
            big_graph_nodes = set(big_graph.nodes)
            constraints = list()
            for vertexSubset in vertexSubsets:
                if (not (vertexSubset & small_graph_nodes)) and (vertexSubset & big_graph_nodes):
                    constraints.append(vertexSubset)
            return constraints

    return VertexSetConstraintChanges

def randomPolygonModel(gridDim, n_obstacles):
    obstacles = {}
    gridDim = tuple(gridDim)
    total_points = np.prod(gridDim)
    obstacles = []
    for i in range(n_obstacles):
        n = random.randint(3, 7)
        flat_indices = np.random.choice(total_points, size=n, replace=False)
        points = np.array(np.unravel_index(flat_indices, gridDim)).T  # shape (k, d)
        polygon = shapely.Polygon(shapely.concave_hull([shapely.Point(*x) for x in points]))
        min_x, min_y, max_x, max_y  = [int(x) for x in polygon.bounds]
        obstacle = set()
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if polygon.contains(shapely.Point(x, y)):
                    obstacle.add((x, y))
        obstacles.append(obstacle)
    return obstacles


def randomRectangularHeuristicModel(gridDim, n_obstacles):
    k = n_obstacles
    if isinstance(gridDim, Number):
        gridDim = (gridDim,)

    gridDim = tuple(gridDim)
    total_points = np.prod(gridDim)

    if k > total_points:
        raise ValueError("Cannot sample more unique points than total grid points.")

    # Sample flat indices
    flat_indices = np.random.choice(total_points, size=k, replace=False)
    dimensions = np.random.randint(low=1, high=20, size=(k, len(gridDim)))

    # Convert flat indices to multi-dimensional indices
    points = np.array(np.unravel_index(flat_indices, gridDim)).T  # shape (k, d)

    vertexSubsets = list()
    for point, dim in zip(points, dimensions):
        axes = [np.arange(start, min(start+length, max_length-1)) for start, length, max_length in zip(point, dim, gridDim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        points = np.stack(mesh, axis=-1).reshape(-1, len(point))
        vertexSubsets.append(points)

    return vertexSubsets

def randomRectangularObstacleModel(gridDim, n_obstacles):
    k = n_obstacles
    if isinstance(gridDim, Number):
        gridDim = (gridDim,)

    gridDim = tuple(gridDim)
    total_points = np.prod(gridDim)

    # Sample flat indices
    flat_indices = np.random.choice(total_points, size=2 * k, replace=True)

    # Convert flat indices to multi-dimensional indices
    points = np.array(np.unravel_index(flat_indices, gridDim)).T  # shape (2*k, d)
    left_points, riht_points = np.split(points, 2)
    left_points, riht_points = np.minimum(left_points, riht_points), np.maximum(left_points, riht_points)

    vertexSubsets = list()
    for left, riht in zip(left_points, riht_points):
        axes = [np.arange(start, end+1, dtype=int) for start, end in zip(left, riht)]
        mesh = np.meshgrid(*axes, indexing='ij')
        points = np.stack(mesh, axis=-1).reshape(-1, len(left))
        vertexSubsets.append(points)

    return vertexSubsets


def independentObstacleModel(gridDim, m, n_obstacles):
    assert m > 0
    total_points = np.prod(gridDim)
    if m >= 1:
        coverage = np.random.randint(n_obstacles, size=(total_points, m))
    else:
        coverage = [list() for _ in range(total_points)]
        for i in range(total_points):
            if random.random() < m:
                coverage[i].append(random.randrange(n_obstacles))
    vertexSubsets = [list() for _ in range(n_obstacles)]
    for vertex in product(*[range(x) for x in gridDim]):
        point = np.prod(np.array(vertex) + 1)
        constraints = coverage[point - 1]
        for constraint in constraints:
            vertexSubsets[constraint].append(tuple(vertex))
    return [np.array(vertexSubset, dtype=int) for vertexSubset in vertexSubsets]


def render2DGrid(m, n, obstacles, soln=None, rem_obstacles = None):
    # flat_obstacles = set(tuple(x) for vertexSet in obstacles for x in vertexSet)
    if rem_obstacles is None:
        obstacles = [[tuple(vertex) for vertex in obstacle] for obstacle in obstacles]
    else:
        obstacles = [([tuple(vertex) for vertex in obstacle] if obstacle not in rem_obstacles else list()) for obstacle in obstacles]
    k = len(obstacles)
    width = len(str(k))
    # print(flat_obstacles)
    for i in range(m):
        for j in range(n):
            ob_id = None
            if soln is not None and (i, j) in soln:
                print('X'.center(width), end=' ')
                continue
            for x, obstacle in enumerate(obstacles):
                if any((i,j) == y for y in obstacle):
                    ob_id = x
                    break
            if ob_id is not None:
                print(str(ob_id).zfill(width), end=' ')
            else:
                print('-'.center(width), end=' ')
        print()

def convert_graph(gridDim, constraints):
    graph = nx.grid_graph(gridDim)
    print(list(graph.nodes))

class exact_mcr(AnytimeAlgorithm):
    def _run(self, graph, src, dst, obstacles):
        path_cover = {x:list() for x in graph.nodes}
        coverage = {x:set() for x in graph.nodes}
        for i, obstacle in enumerate(obstacles):
            for vertex in obstacle:
                coverage[vertex].add(i)
        queue = [(len(coverage[src]), src, coverage[src])]
        while queue:
            n_obstacles, vertex, cur_obstacles = heappop(queue)
            for i, subset in reversed(list(enumerate(path_cover[vertex]))):
                if subset > cur_obstacles:
                    del path_cover[vertex][i]
                if subset <= cur_obstacles:
                    break
            else:
                if vertex == dst:
                    indices = cur_obstacles
                    return {obstacles[i] for i in indices}
                path_cover[vertex].append(frozenset(cur_obstacles))
                for next_vertex in graph.neighbors(vertex):
                    next_obstacles = cur_obstacles | coverage[next_vertex]
                    if not any(subset <= next_obstacles for subset in path_cover[next_vertex]):
                        heappush(queue, (len(next_obstacles), next_vertex, next_obstacles))
        indices = min(path_cover[dst], key=len)
        return {obstacles[i] for i in indices}

def ids_ed_mcr2(graph, src, dst, obstacles):
    path_cover = {x:list() for x in graph.nodes}
    coverage = {x:set() for x in graph.nodes}
    for i, obstacle in enumerate(obstacles):
        for vertex in obstacle:
            coverage[vertex].add(i)
    queue = [(manhattan_dist(graph, src, dst), src, coverage[src])]
    next_queue = list()
    cur_obstacle_count = len(coverage[src])
    while queue:
        while queue:
            _, vertex, cur_obstacles = heappop(queue)
            for i, subset in reversed(list(enumerate(path_cover[vertex]))):
                if subset <= cur_obstacles:
                    break
            else:
                if vertex == dst:
                    indices = cur_obstacles
                    return {obstacles[i] for i in indices}
                path_cover[vertex].append(frozenset(cur_obstacles))
                for next_vertex in graph.neighbors(vertex):
                    next_obstacles = cur_obstacles | coverage[next_vertex]
                    if len(next_obstacles) == cur_obstacle_count:
                        if not any(subset <= next_obstacles for subset in path_cover[next_vertex]):
                            heappush(queue, (manhattan_dist(graph, next_vertex, dst), next_vertex, next_obstacles))
                    else:
                        if not any(subset <= next_obstacles for subset in path_cover[next_vertex]):
                            next_queue.append((manhattan_dist(graph, next_vertex, dst), next_vertex, next_obstacles))
        queue = sorted(next_queue, key=lambda x: (len(x[2]), x[0]))
        cur_obstacle_count = len(queue[0][2])
        x = bisect.bisect_right([len(y[2]) for y in queue], cur_obstacle_count)
        next_queue = queue[x:]
        queue = queue[:x]

def ids_mc_mcr2(graph, src, dst, obstacles):
    path_cover = {x:list() for x in graph.nodes}
    coverage = {x:set() for x in graph.nodes}
    for i, obstacle in enumerate(obstacles):
        for vertex in obstacle:
            coverage[vertex].add(i)
    queue = [(src, coverage[src])]
    next_queue = list()
    cur_obstacle_count = len(coverage[src])
    while queue:
        while queue:
            vertex, cur_obstacles = queue.pop()
            for i, subset in reversed(list(enumerate(path_cover[vertex]))):
                if subset <= cur_obstacles:
                    break
            else:
                if vertex == dst:
                    indices = cur_obstacles
                    return {obstacles[i] for i in indices}
                path_cover[vertex].append(frozenset(cur_obstacles))
                for next_vertex in graph.neighbors(vertex):
                    next_obstacles = cur_obstacles | coverage[next_vertex]
                    if len(next_obstacles) == cur_obstacle_count:
                        if not any(subset <= next_obstacles for subset in path_cover[next_vertex]):
                            queue.append((next_vertex, next_obstacles))
                    else:
                        if not any(subset <= next_obstacles for subset in path_cover[next_vertex]):
                            next_queue.append((next_vertex, next_obstacles))
        queue = sorted(next_queue, key=lambda x: len(x[1]))
        cur_obstacle_count = len(queue[0][1])
        x = bisect.bisect_right([len(y[1]) for y in queue], cur_obstacle_count)
        next_queue = queue[x:]
        queue = queue[:x]

class greedy_mcr1(AnytimeAlgorithm):
    def _run(self, graph, src, dst, obstacles):
        path_cover = {x:set(range(len(obstacles))) for x in graph.nodes}
        coverage = {x:set() for x in graph.nodes}
        for i, obstacle in enumerate(obstacles):
            for vertex in obstacle:
                coverage[vertex].add(i)
        queue = [[len(coverage[src]), src, coverage[src]]]
        while queue:
            n_obstacles, vertex, cur_obstacles = heappop(queue)
            if vertex == dst:
                indices = cur_obstacles
                return {obstacles[i] for i in indices}
            if len(cur_obstacles) < len(path_cover[vertex]):
                path_cover[vertex] = cur_obstacles
                for next_vertex in graph.neighbors(vertex):
                    next_obstacles = cur_obstacles | coverage[next_vertex]
                    if len(next_obstacles) <= len(path_cover[next_vertex]):
                        heappush(queue, (len(next_obstacles), next_vertex, next_obstacles))
        indices = path_cover[dst]
        return {obstacles[i] for i in indices}


def manhattan_dist(graph, src, dst):
    return sum(abs(x-y) for x, y in zip(graph.nodes[src]["pos"], graph.nodes[dst]["pos"]))


class greedy_mcr2(AnytimeAlgorithm):
    def _run(self, graph, src, dst, obstacles):
        path_cover = {x:set(range(len(obstacles))) for x in graph.nodes}
        coverage = {x:set() for x in graph.nodes}
        for i, obstacle in enumerate(obstacles):
            for vertex in obstacle:
                coverage[vertex].add(i)
        queue = [[manhattan_dist(graph, src, dst), src, coverage[src]]]
        while queue:
            n_obstacles, vertex, cur_obstacles = heappop(queue)
            if vertex == dst:
                indices = cur_obstacles
                return {obstacles[i] for i in indices}
            if len(cur_obstacles) < len(path_cover[vertex]):
                path_cover[vertex] = cur_obstacles
                for next_vertex in graph.neighbors(vertex):
                    next_obstacles = cur_obstacles | coverage[next_vertex]
                    if len(next_obstacles) <= len(path_cover[next_vertex]):
                        heappush(queue, (manhattan_dist(graph, next_vertex, dst), next_vertex, next_obstacles))
        indices = path_cover[dst]
        return {obstacles[i] for i in indices}



def ids_mc_mcr(graph, src, dst, obstacles):
    coverage = {x:set() for x in graph.nodes}
    for i, obstacle in enumerate(obstacles):
        for vertex in obstacle:
            coverage[vertex].add(i)
    queue = [(src, coverage[src])]
    next_queue = list()
    cur_obstacle_count = len(coverage[src])
    while queue:
        visited = dict()
        while queue:
            vertex, cur_obstacles = queue.pop()
            frozen_cur_obstacles = frozenset(cur_obstacles)
            if frozen_cur_obstacles in visited:
                if vertex in visited[frozen_cur_obstacles]:
                    continue
                else:
                    visited[frozen_cur_obstacles].add(vertex)
            else:
                visited[frozen_cur_obstacles] = {vertex}
            # print(vertex, cur_obstacles)
            if vertex == dst:
                indices = cur_obstacles
                return {obstacles[i] for i in indices}
            for next_vertex in graph.neighbors(vertex):
                next_obstacles = cur_obstacles | coverage[next_vertex]
                if len(next_obstacles) == cur_obstacle_count:
                    queue.append((next_vertex, next_obstacles))
                else:
                    next_queue.append((next_vertex, next_obstacles))
        queue = sorted(next_queue, key=lambda x: len(x[1]))
        cur_obstacle_count = len(queue[0][1])
        x = bisect.bisect_right([len(y[1]) for y in queue], cur_obstacle_count)
        next_queue = queue[x:]
        queue = queue[:x]


def ids_ed_mcr(graph, src, dst, obstacles):
    coverage = {x:set() for x in graph.nodes}
    for i, obstacle in enumerate(obstacles):
        for vertex in obstacle:
            coverage[vertex].add(i)
    queue = [(manhattan_dist(graph, src, dst), src, coverage[src])]
    next_queue = list()
    cur_obstacle_count = len(coverage[src])
    while queue:
        visited = dict()
        while queue:
            _, vertex, cur_obstacles = heappop(queue)
            frozen_cur_obstacles = frozenset(cur_obstacles)
            if frozen_cur_obstacles in visited:
                if vertex in visited[frozen_cur_obstacles]:
                    continue
                else:
                    visited[frozen_cur_obstacles].add(vertex)
            else:
                visited[frozen_cur_obstacles] = {vertex}
            # print(vertex, cur_obstacles)
            if vertex == dst:
                indices = cur_obstacles
                return {obstacles[i] for i in indices}
            for next_vertex in graph.neighbors(vertex):
                next_obstacles = cur_obstacles | coverage[next_vertex]
                if len(next_obstacles) == cur_obstacle_count:
                    heappush(queue, (manhattan_dist(graph, next_vertex, dst), next_vertex, next_obstacles))
                else:
                    next_queue.append((manhattan_dist(graph, next_vertex, dst), next_vertex, next_obstacles))
        queue = sorted(next_queue, key=lambda x: (len(x[2]), x[0]))
        cur_obstacle_count = len(queue[0][2])
        x = bisect.bisect_right([len(y[2]) for y in queue], cur_obstacle_count)
        next_queue = queue[x:]
        queue = queue[:x]

def reachability(graph, src, dst, obstacles):
    obstacles = set(obstacles)
    if not (graph.nodes[src]["new_coverage"] < obstacles):
        return False
    if not (graph.nodes[dst]["new_coverage"] < obstacles):
        return False
    visited = {src,}

    queue = array.array('L', [src])
    while queue:
        u = queue.pop()
        for v in graph.neighbors(u):
            if v == dst:
                return True
            if v not in visited:
                if not graph.nodes[v]["new_coverage"] < obstacles:
                    continue
                queue.append(v)
                visited.add(v)
    return False


class DD_VARIANT(Enum):
    VANILLA = 1
    TWOPHASE = 2
    OPDD = 4
    PROBDD = 5
    CDD = 6
    ONEMIN = 7

#TODO Remove use_indices - always false
class dd_mcr(AnytimeAlgorithm):
    def __init__(self, timeout=None, variant=DD_VARIANT.VANILLA, max_depth=2, repeat=1):
        super().__init__(timeout)
        self.max_depth = max_depth
        self.variant = variant
        self.max_repeat = repeat

    def opdd(self, graph, src, dst, obstacles, removed=list(), n=2):
        cur_obstacles = list(obstacles)
        cur_result = cur_obstacles + removed
        if self.result is None or len(cur_result) < len(self.result):
            self.result = cur_result

        granularity = len(cur_obstacles)+1
        while granularity > 1:
            granularity = min(granularity, len(cur_obstacles)+1) // 2

            for i in range(len(cur_obstacles), 0, -granularity):
                cbar = cur_obstacles[:max(i-granularity, 0)] + cur_obstacles[i:]
                t = reachability(graph, src, dst, cbar + removed)

                if t:
                    cur_obstacles = cbar
                    if len(cur_obstacles) + len(removed) < len(self.result):
                        self.result = cur_obstacles + removed
        return self.result

    # One Pass Delta Debugging
    def opdd2(self, graph, src, dst, obstacles, removed=list(), n=2):
        cur_obstacles = list(obstacles)
        cur_result = cur_obstacles + removed
        if self.result is None or len(cur_result) < len(self.result):
            self.result = cur_result

        # We replace the tail recursion from the paper by a loop
        while n < len(cur_obstacles):
            # tc = reachability(graph, src, dst, cur_obstacles + removed)
            # assert tc

            # if n >= len(cur_obstacles):
            #     # Ensure 1-minimality
            #     self.result = self.onemin(graph, src, dst, cur_obstacles, removed=removed)
            #     return self.result

            k, m = divmod(len(cur_obstacles), n)
            # cs = [cur_obstacles[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

            next_n = n

            #for i in range(n):
            for i in range(n-1, -1, -1):
                cbar = cur_obstacles[:i*k+min(i, m)] + cur_obstacles[(i+1)*k+min(i+1, m):]
                # cbar = list(set(cur_obstacles) - set(cs[i]))
                # k, m = divmod(len(cur_obstacles), n)
                # test = cur_obstacles[:i*k+min(i, m)] + cur_obstacles[(i+1)*k+min(i+1, m):]
                # assert sorted(test) == sorted(cbar)
                t = reachability(graph, src, dst, cbar + removed)

                if t:
                    cur_obstacles = cbar
                    if len(cur_obstacles) + len(removed) < len(self.result):
                        self.result = cur_obstacles + removed
                    next_n = next_n - 1


            # n = min(len(cur_obstacles), next_n * 2)
            n = 2 * next_n

        return self.onemin(graph, src, dst, cur_obstacles, removed=removed)

    def _run(self, graph, src, dst, obstacles, max_repeat = 1):
        mcr_func = self.opdd
        # if self.variant == DD_VARIANT.VANILLA:
        #     mcr_func = self.vanilla
        # elif self.variant == DD_VARIANT.TWOPHASE:
        #     mcr_func = self.twophase
        # elif self.variant == DD_VARIANT.OPDD:
        #     mcr_func = self.opdd
        # elif self.variant == DD_VARIANT.PROBDD:
        #     mcr_func = self.probdd
        # elif self.variant == DD_VARIANT.CDD:
        #     mcr_func = self.cdd
        # elif self.variant == DD_VARIANT.ONEMIN:
        #     mcr_func = self.onemin

        obstacles = list(obstacles)
        self.result = obstacles
        for i in range(self.max_repeat):
            _ = mcr_func(graph, src, dst, set(obstacles))
            random.shuffle(obstacles)
            self.set_marker(f"Repeat_finished {i}")

        return self.result

    def __str__(self):
        return str(self.variant) + "_" + str(self.max_repeat)

def load_map(filename):
    with open(filename, 'r') as f:
        map_type = f.readline().strip().split()[1]
        assert map_type == "octile"
        height = f.readline()
        height = int(height.split(' ')[1])
        width = f.readline()
        width = int(width.split(' ')[1])
        _ = f.readline()
        graph = nx.grid_graph((width, height))
        for i in range(height):
            row = f.readline()
            for j, char in enumerate(row):
                if j >= width:
                    break
                # https://web.archive.org/web/20160304051410id_/http://web.cs.du.edu/~sturtevant/papers/benchmarks.pdf
                if char in {'@', 'O', 'T', 'W'}:
                    graph.remove_node((i, j))
        return (height, width), graph

if __name__ == '__main__':
    OUT_DIR = Path(sys.argv[1])
    random.seed(42)
    np.random.seed(42)
    gridDim = (64, 64)
    n_obstacles_list = [10] + list(range(20, 210, 10))
    # n_obstacles_list = [500]
    n_obstacles = n_obstacles_list[-1]
    timeout = None
    mcr_funcs = [
                 exact_mcr(timeout),
                 greedy_mcr1(timeout),
                 dd_mcr(timeout, DD_VARIANT.OPDD),
                 # dd_mcr(timeout, DD_VARIANT.OPDD, repeat=3),
    ]
    mcr_funcs = [dd_mcr(timeout, DD_VARIANT.OPDD, repeat=1), 
                    greedy_mcr1(timeout), 
                    #exact_mcr(timeout)
                    ]
    for real_obstacles in [independentObstacleModel(gridDim, 50, n_obstacles), randomRectangularObstacleModel(gridDim, n_obstacles)]:
        real_obstacles = [frozenset(tuple(int(x) for x in vertex) for vertex in obstacle) for obstacle in real_obstacles]
        graph = nx.grid_graph(gridDim)
        src, dst = random.sample(sorted(graph.nodes), k=2)
        node_mapping = {node:tuple(node) for node in graph.nodes}
        graph = nx.relabel_nodes(graph, node_mapping)
        def evaluate(graph, gridDim, real_obstacles, n_obstacles, src, dst, mcr_funcs):
            obstacles = real_obstacles[:n_obstacles]

            new_graph = nx.Graph(graph)
            new_graph = VertexSetConstraintChangesFactory(obstacles)(nx.Graph(graph)).remove(new_graph, obstacles)
            try:
                path = nx.shortest_path(new_graph, source=src, target=dst)
                return
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                pass

            coverage = {node:set() for node in graph.nodes}
            for i, obstacle in enumerate(obstacles):
                for node in obstacle:
                    if node in coverage:
                        coverage[node].add(i)
            nx.set_node_attributes(graph, coverage, name="coverage")
            aux_graph = nx.convert_node_labels_to_integers(graph, label_attribute="pos")
            equivalences = nx.equivalence_classes(aux_graph.nodes, lambda u, v: aux_graph.nodes[u]["coverage"] == aux_graph.nodes[v]["coverage"])

            partitions = list()
            for eq_class in equivalences:
                subgraph = aux_graph.subgraph(eq_class)
                components = nx.connected_components(subgraph)
                partitions.extend(components)
            aux_graph = nx.quotient_graph(aux_graph, partitions, relabel=True)

            aux_src = None
            aux_dst = None
            for node in aux_graph.nodes:
                real_positions = nx.get_node_attributes(aux_graph.nodes[node]["graph"], "pos")
                positions = [np.array(pos, dtype=np.float64) for pos in real_positions.values()]
                new_pos = np.average(positions, axis=0)
                new_pos = tuple(new_pos)
                aux_graph.nodes[node]["pos"] = new_pos
                first_node = list(aux_graph.nodes[node]["graph"].nodes)[0]
                aux_graph.nodes[node]["coverage"] = aux_graph.nodes[node]["graph"].nodes[first_node]["coverage"]
                if src in real_positions.values():
                    aux_src = node
                if dst in real_positions.values():
                    aux_dst = node

            assert aux_src is not None
            assert aux_dst is not None

            new_obstacles = [set() for _ in range(len(obstacles))]
            for node in aux_graph.nodes:
                for obstacle in aux_graph.nodes[node]["coverage"]:
                    new_obstacles[obstacle].add(node)

            obstacles = [frozenset(obstacle) for obstacle in new_obstacles]

            new_coverage = dict()
            for node in aux_graph.nodes:
                node_coverage = set()
                for obstacle in aux_graph.nodes[node]["coverage"]:
                    node_coverage.add(obstacles[obstacle])
                new_coverage[node] = frozenset(node_coverage)
            nx.set_node_attributes(aux_graph, new_coverage, name="new_coverage")

            aux_graph = nx.freeze(aux_graph)

            report = []
            report.append(len(obstacles))
            print("Obstacles:", len(obstacles))
            print(20*'-')
            for i, mcr_func in enumerate(mcr_funcs):
                soln = mcr_func(aux_graph, aux_src, aux_dst, obstacles)
                if soln is None:
                    soln = set(obstacles)
                time = timeit.timeit
                if i==0 and False:
                    time =0
                else:
                     time = timeit.timeit("mcr_func(aux_graph, aux_src, aux_dst, obstacles)", number=11, globals=locals())/11
                remaining_obstacles = set(obstacles) - set(soln)
                new_graph = nx.Graph(aux_graph)
                vertices = [vertex for subset in remaining_obstacles for vertex in subset]
                new_graph.remove_nodes_from(vertices)
                assert aux_src in new_graph.nodes
                assert aux_dst in new_graph.nodes
                path = nx.shortest_path(new_graph, source=aux_src, target=aux_dst)
                report.extend([len(path), len(soln), time])
                print(mcr_func, "\tsoln:", len(soln), "\ttime:", time)
            print()
            report.extend([src, dst, gridDim])
            out = ', '.join(str(x) for x in report)
            # print(out + '\n')
            sys.stdout.flush()

        for n_obstacles in n_obstacles_list:
            evaluate(graph, gridDim, real_obstacles, n_obstacles, src, dst, mcr_funcs)