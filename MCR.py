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

from general_graph import Graph, ShortestPathSpecFactory, VertexSetConstraintChangesFactory
from Anytime import AnytimeAlgorithm, PRINT

OUT_DIR = Path("./out")


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


def dd_mcr_orig(graph, src, dst, obstacles):
    new_graph = Graph(graph) >> (ShortestPathSpecFactory(src, dst), VertexSetConstraintChangesFactory(obstacles), )
    existing_obstacles = VertexSetConstraintChangesFactory(obstacles)(graph).get_deltas(graph, new_graph.graph)
    remaining_obstacles = set(obstacles) - set(existing_obstacles)
    return remaining_obstacles
    # return existing_obstacles


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


    queue = [(manhattan_dist(graph, src, dst), src)]
    while queue:
        _, vertex = heappop(queue)
        for v in graph.neighbors(vertex):
            if v == dst:
                return True
            if v not in visited:
                if not graph.nodes[v]["new_coverage"] < obstacles:
                    continue
                heappush(queue, (manhattan_dist(graph, v, dst), v))
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

    def onemin(self, graph, src, dst, obstacles, removed=list()):
        cur_obstacles = obstacles + removed
        if len(cur_obstacles) < len(self.result):
            self.result = cur_obstacles
        cur_obstacles = set(cur_obstacles)
        result = set(obstacles)
        for obstacle in obstacles:
            cur_obstacles.remove(obstacle)
            t = reachability(graph, src, dst, cur_obstacles)
            # print("ITER:", min(len(cur_obstacles), len(self.result)), t, len(cur_obstacles) <= len(self.result))

            if t:
                result.remove(obstacle)
                if len(cur_obstacles) < len(self.result):
                    self.result = set(cur_obstacles)
            else:
                cur_obstacles.add(obstacle)
            # print("ITER:", len(self.result))
        # self.set_marker(f"1-minimal done {len(result)}")
        return result

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
                # print("ITER:", min(len(cur_obstacles), len(self.result)), t, len(cur_obstacles) <= len(self.result))

                if t:
                    cur_obstacles = cbar
                    if len(cur_obstacles) + len(removed) < len(self.result):
                        self.result = cur_obstacles + removed
                # print("ITER:", len(self.result))
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

    def probdd(self, graph, src, dst, obstacles, removed= list()):
        # Initialize the probability of each element with p0
        p0 = 0.05
        cur_obstacles = [(p0, obstacle) for obstacle in obstacles]

        all_obstacles = list(obstacle for _, obstacle in cur_obstacles) + removed
        if self.result is None or len(all_obstacles) < len(self.result):
            self.result = all_obstacles

        while any(prob < 1 for prob, _ in cur_obstacles):
            # tc = reachability(graph, src, dst, all_obstacles)
            # assert tc
            random.shuffle(cur_obstacles)
            cur_obstacles.sort(key=lambda x: x[0])

            end_index = 0
            currentMaxGain = 0
            den_factor = 1

            for i, l in enumerate(cur_obstacles):
                den_factor *= (1-l[0])
                gain = (i+1) * den_factor
                if gain > currentMaxGain:
                    currentMaxGain = gain
                    end_index = i
                else:
                    break

            #print("HI", end_index+1, den_factor)

            set_S = set(cur_obstacles[:end_index+1])

            cbar = cur_obstacles[end_index+1:]
            t = reachability(graph, src, dst, [x[1] for x in cbar] + removed)

            if t:
                cur_obstacles = cbar
                all_obstacles = list(obstacle for _, obstacle in cur_obstacles) + removed
                if len(all_obstacles) < len(self.result):
                    self.result = all_obstacles
            else:
                den_factor = reduce(lambda x, y: x * (1 - y[0]), set_S, 1)
                factor = 1 / (1 - den_factor)
                next_obstacles = list()
                for prob_obstacle in cur_obstacles:
                    if prob_obstacle in set_S:
                        prob, obstacle = prob_obstacle
                        next_obstacles.append((prob * factor, obstacle))
                    else:
                        next_obstacles.append(prob_obstacle)
                cur_obstacles = next_obstacles

        return set(x[1] for x in cur_obstacles)

    # Counter Based Delta Debugging
    def cdd(self, graph, src, dst, obstacles, removed=list()):
        cur_obstacles = list(obstacles)
        cur_result = cur_obstacles + removed
        if self.result is None or len(cur_result) < len(self.result):
            self.result = cur_result
        cbar_offset = 0
        # Round number, Initially 0
        r = 0
        pr = 0.05
        sr = 2

        # We replace the tail recursion from the paper by a loop
        while sr > 1:
            # tc = reachability(graph, src, dst, cur_obstacles+removed)
            # assert tc


            currentMaxGain = 0
            sr = 0
            for s in range(1, len(cur_obstacles)):
                gain = s*((1-pr)**s)
                if currentMaxGain < gain:
                    currentMaxGain = gain
                    sr = s

            k = int((len(cur_obstacles) + sr - 1)/sr)
            cs = [cur_obstacles[sr*i:min(sr*i+sr, len(cur_obstacles))] for i in range(k)]


            for j in range(k):
                i = int((j + cbar_offset) % k)
                cbar = list(set(cur_obstacles) - set(cs[i]))
                # k, m = divmod(len(cur_obstacles), sr)
                # test = cur_obstacles[:i*k+min(i, m)] + cur_obstacles[(i+1)*k+min(i+1, m):]
                # assert sorted(test) == sorted(cbar)
                t = reachability(graph, src, dst, cbar + removed)

                #TODO: Try changing as per OPDD (direct to 2*next_n-2)
                if t:
                    cur_obstacles = cbar

                    if len(cur_obstacles) + len(removed) < len(self.result):
                        self.result = cur_obstacles + removed

                    # In next run, start removing the following subset
                    cbar_offset = i
                    break

            r += 1
            pr *= 1.582

        self.result = self.onemin(graph, src, dst, cur_obstacles, removed=removed)
        return self.result

    def vanilla(self, graph, src, dst, obstacles, removed=list(), n=2):
        cur_obstacles = list(obstacles)
        cur_result = cur_obstacles + removed
        if self.result is None or len(cur_result) < len(self.result):
            self.result = cur_result
        cbar_offset = 0

        # We replace the tail recursion from the paper by a loop
        while 1:
            # tc = reachability(graph, src, dst, cur_obstacles+removed)
            # assert tc

            if len(cur_obstacles) + len(removed) < len(self.result):
                self.result = cur_obstacles + removed

            if n > len(cur_obstacles):
                # No further minimizing
                return self.result

            k, m = divmod(len(cur_obstacles), n)
            cs = [cur_obstacles[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

            cbar_failed = False

            next_obstacles = cur_obstacles[:]
            next_n = n

            for j in range(n):
                i = int((j + cbar_offset) % n)
                #cbar = list(set(cur_obstacles) - set(cs[i]))
                k, m = divmod(len(cur_obstacles), n)
                cbar = cur_obstacles[:i*k+min(i, m)] + cur_obstacles[(i+1)*k+min(i+1, m):]
                # assert sorted(test) == sorted(cbar)
                t = reachability(graph, src, dst, cbar + removed)

                #TODO: Try changing as per OPDD (direct to 2*next_n-2)
                if t:
                    cbar_failed = True
                    next_obstacles = cbar
                    next_n = next_n - 1

                    # In next run, start removing the following subset
                    cbar_offset = i
                    break

            if not cbar_failed:
                if n >= len(cur_obstacles):
                    # No further minimizing
                    if len(cur_obstacles) + len(removed) < len(self.result):
                        self.result = cur_obstacles + removed
                    return self.result

                next_n = min(len(cur_obstacles), n * 2)
                cbar_offset = (cbar_offset * next_n) / n

            cur_obstacles = next_obstacles
            n = next_n

    # def dd_mcr3(self, c, r=list(), *, _depth=0):
    #TODO Move to BFS based approach

    def twophase(self, graph, src, dst, cur_obstacles, removed=list(), _depth=0):
        if self.result is None or len(cur_obstacles) + len(removed) < len(self.result):
            self.result = set(cur_obstacles) | set(removed)
        for _ in range(self.max_repeat):
            solns = list()
            set_c = set(cur_obstacles)
            list_c = list(cur_obstacles)
            random.shuffle(list_c)
            while reachability(graph, src, dst, list_c + removed):
                soln = frozenset(self.vanilla(graph, src, dst, list_c, removed=removed))
                if len(soln) + len(removed) < len(self.result):
                    self.result = soln | set(removed)
                set_c -= soln
                list_c = list(set_c)
                if soln not in solns:
                    solns.append(soln)

            if _depth < self.max_depth:
                all_soln = set()
                all_soln = all_soln.union(*solns)
                set_c = set(cur_obstacles) - all_soln
                list_c = list(set_c)
                for i, soln in enumerate(list(solns)):
                    irreplaceable_deltas = self.vanilla(graph, src, dst, soln, removed=list_c+removed)
                    if len(irreplaceable_deltas) + len(removed) >= len(self.result):
                        continue
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Pre Opt Soln size: ", len(solns[i]) + len(removed))
                    verbose=0
                    if len(irreplaceable_deltas) == len(soln):
                        verbose=0
                        if verbose:
                            print(_depth*"  "+"Post Opt Soln size: ", len(solns[i]) + len(removed), " (Irreplaceable)")
                        verbose=0
                        continue
                    result = self.twophase(graph, src, dst, list_c, removed=irreplaceable_deltas+removed, _depth=_depth+1, use_indices=True)
                    if len(result) < len(soln) - len(irreplaceable_deltas):
                        solns[i] = result + irreplaceable_deltas
                        if len(solns[i]) + len(removed) < len(self.result):
                            self.result = solns[i] + removed
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Post Opt Soln size: ", len(solns[i]) + len(removed))
                    verbose=0
        return self.result

    def twophase_specific(self, graph, src, dst, cur_obstacles, removed=list(), _depth=0):
        if self.result is None or len(cur_obstacles) + len(removed) < len(self.result):
            self.result = set(cur_obstacles) | set(removed)
        for _ in range(self.max_repeat):
            solns = list()
            set_c = set(cur_obstacles)
            list_c = list(cur_obstacles)
            random.shuffle(list_c)
            while reachability(graph, src, dst, list_c + removed):
                soln = frozenset(self.vanilla(graph, src, dst, list_c, removed=removed))
                if len(soln) + len(removed) < len(self.result):
                    self.result = soln | set(removed)
                set_c -= soln
                list_c = list(set_c)
                if soln not in solns:
                    solns.append(soln)

            if _depth < self.max_depth:
                all_soln = set()
                all_soln = all_soln.union(*solns)
                set_c = set(cur_obstacles) - all_soln
                list_c = list(set_c)
                for i, soln in enumerate(list(solns)):
                    irreplaceable_deltas = self.vanilla(graph, src, dst, soln, removed=list_c+removed)
                    if len(irreplaceable_deltas) + len(removed) >= len(self.result):
                        continue
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Pre Opt Soln size: ", len(solns[i]) + len(removed))
                    verbose=0
                    if len(irreplaceable_deltas) == len(soln):
                        verbose=0
                        if verbose:
                            print(_depth*"  "+"Post Opt Soln size: ", len(solns[i]) + len(removed), " (Irreplaceable)")
                        verbose=0
                        continue
                    result = self.twophase(graph, src, dst, list_c, removed=irreplaceable_deltas+removed, _depth=_depth+1, use_indices=True)
                    if len(result) < len(soln) - len(irreplaceable_deltas):
                        solns[i] = result + irreplaceable_deltas
                        if len(solns[i]) + len(removed) < len(self.result):
                            self.result = solns[i] + removed
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Post Opt Soln size: ", len(solns[i]) + len(removed))
                    verbose=0

    # Combinatorial Blocking Algorithm
    def cba(self, graph, src, dst, cur_obstacles, removed=list()):
        if self.result is None or len(cur_obstacles) + len(removed) < len(self.result):
            self.result = set(cur_obstacles) | set(removed)
        for _ in range(self.max_repeat):
            solns = list()
            set_c = set(cur_obstacles)
            list_c = list(cur_obstacles)
            random.shuffle(list_c)
            while reachability(graph, src, dst, list_c + removed):
                soln = frozenset(self.vanilla(graph, src, dst, list_c, removed=removed))
                if len(soln) + len(removed) < len(self.result):
                    self.result = soln | set(removed)
                set_c -= soln
                list_c = list(set_c)
                if soln not in solns:
                    solns.append(soln)

    def _run(self, graph, src, dst, obstacles, max_repeat = 1):
        mcr_func = None
        if self.variant == DD_VARIANT.VANILLA:
            mcr_func = self.vanilla
        elif self.variant == DD_VARIANT.TWOPHASE:
            mcr_func = self.twophase
        elif self.variant == DD_VARIANT.OPDD:
            mcr_func = self.opdd
        elif self.variant == DD_VARIANT.PROBDD:
            mcr_func = self.probdd
        elif self.variant == DD_VARIANT.CDD:
            mcr_func = self.cdd
        elif self.variant == DD_VARIANT.ONEMIN:
            mcr_func = self.onemin

        obstacles = list(obstacles)
        self.result = obstacles
        # print("ITER:", len(self.result))
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

if __name__ == '__main__' and False:
    mcr_func = dd_mcr(None, DD_VARIANT.TWOPHASE, max_depth=2, repeat=1)
    mcr_func = dd_mcr(None, DD_VARIANT.ONEMIN, repeat=1)
    filename = "./maps/random-64-64-20.map"
    obstacleModel = randomRectangularObstacleModel
    obstacleModel = independentObstacleModel
    obstacleModel = randomPolygonModel
    n_obstacles = 200

    gridDim, graph = load_map(filename)
    real_obstacles = obstacleModel(gridDim, n_obstacles)
    # real_obstacles = obstacleModel(gridDim, 50, n_obstacles)
    obstacles = [frozenset(tuple(vertex) for vertex in obstacle) for obstacle in real_obstacles]
    node_mapping = {node: tuple(node) for node in graph.nodes}
    graph = nx.relabel_nodes(graph, node_mapping)

    while True:
        src, dst = random.sample(graph.nodes, k=2)
        new_graph = nx.Graph(graph)
        VertexSetConstraintChangesFactory(obstacles)(nx.Graph(graph)).remove(new_graph, obstacles)
        try:
            path = nx.shortest_path(new_graph, source=src, target=dst)
        except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
            break

    coverage = {node: set() for node in graph.nodes}
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

    obs_graph = nx.Graph()
    for obstacle in obstacles:
        obs_graph.add_node(obstacle)

    list_obstacles = list(obstacles)
    for i, u in enumerate(list_obstacles):
        for v in list_obstacles[i + 1:]:
            if len(u & v):
                obs_graph.add_edge(u, v)

    real_obstacles = obstacles
    set_obstacles = set(obstacles)
    print(len(obstacles))
    exact_soln = exact_mcr(None)(aux_graph, aux_src, aux_dst, obstacles)
    print(len(exact_soln))
    greedy_soln = greedy_mcr1(None)(aux_graph, aux_src, aux_dst, obstacles)
    print(len(greedy_soln))
    print()
    # for obstacles in itertools.permutations(real_obstacles):
    # for _ in range(10000):
    for _ in range(1):
        random.shuffle(obstacles)
        soln = mcr_func(aux_graph, aux_src, aux_dst, obstacles)
        remaining_obstacles = set_obstacles - set(soln)
        new_graph = nx.Graph(aux_graph)
        vertices = [vertex for subset in remaining_obstacles for vertex in subset]
        new_graph.remove_nodes_from(vertices)
        path = nx.shortest_path(new_graph, source=aux_src, target=aux_dst)
        print(len(soln))
        sys.stdout.flush()


# Convergence data
if __name__ == '__main__' and False:
    OUT_DIR = Path(sys.argv[1])
    # TODO: Graph compaction for nodes with same coverage
    random.seed(42)
    np.random.seed(42)
    gridDim = (64, 64)
    gridDim_list = [(x, x) for x in range(100, 169, 10)]
    n_obstacles_list = [100] + list(range(500, 1100, 100))
    n_obstacles_list = [100] + list(range(200, 2100, 200))
    n_obstacles_list = [10] + list(range(20, 210, 10))
    n_obstacles_list = [500]
    n_obstacles = n_obstacles_list[-1]
    # real_obstacles = randomRectangularHeuristicModel(gridDim, n_obstacles)
    # mcr_funcs = [dd_mcr2, dd_mcr3, exact_mcr, ids_mc_mcr2, ids_ed_mcr2, greedy_mcr1, greedy_mcr2, aco_mcr, saco_mcr]
    # mcr_funcs = [dd_mcr2, dd_mcr3, exact_mcr, greedy_mcr1, greedy_mcr2, aco_mcr]
    # mcr_funcs = [dd_mcr2, dd_mcr3, greedy_mcr1, greedy_mcr2, aco_mcr]
    timeout = None
    # mcr_funcs = [dd_mcr2(timeout), dd_mcr3(timeout), exact_mcr(timeout), greedy_mcr1(timeout), greedy_mcr2(timeout), aco_mcr(timeout)]
    mcr_funcs = [
                 # exact_mcr(timeout),
                 greedy_mcr1(timeout),
                 # dd_mcr(timeout, DD_VARIANT.ONEMIN),
                 # dd_mcr(timeout, DD_VARIANT.VANILLA),
                 dd_mcr(timeout, DD_VARIANT.OPDD),
                 # dd_mcr(timeout, DD_VARIANT.OPDD, repeat=3),
    ]  # independent 4
    mcr_funcs = [dd_mcr(timeout, DD_VARIANT.OPDD, repeat=1), 
                    greedy_mcr1(timeout), 
                    #exact_mcr(timeout)
                    ]
    for real_obstacles in [independentObstacleModel(gridDim, 50, n_obstacles), randomRectangularObstacleModel(gridDim, n_obstacles)]:
    # for real_obstacles in [randomRectangularObstacleModel(gridDim, n_obstacles)]:
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

            # src, dst = random.sample(aux_graph.nodes, k=2)
            aux_graph = nx.freeze(aux_graph)

            obs_graph = nx.Graph()
            for obstacle in obstacles:
                obs_graph.add_node(obstacle)

            list_obstacles = list(obstacles)
            for i, u in enumerate(list_obstacles):
                for v in list_obstacles[i + 1:]:
                    if len(u & v):
                        obs_graph.add_edge(u, v)
            communities = nx.community.louvain_communities(obs_graph)


            REPORT = True
            report = []
            report.append(len(obstacles))
            # if REPORT:
                # print(f"Src: {src}")
                # print(f"Dst: {dst}")
            for i, mcr_func in enumerate(mcr_funcs):
                # if i==2:
                #     obstacles = list(itertools.chain.from_iterable(communities))
                # elif i==3:
                #     obstacles = list(reverse_cuthill_mckee_ordering(obs_graph))
                # elif i==4:
                #     obstacles.sort(key=len)
                # print(mcr_func)
                # if not False:
                #     if hasattr(mcr_func, '__name__'):
                #         print(mcr_func.__name__)
                #         print(len(mcr_func.__name__)*"=")
                #     elif hasattr(mcr_func, '__class__'):
                #         print(mcr_func.__class__.__name__)
                #         print(len(mcr_func.__class__.__name__)*"=")

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
                if True:
                    report.extend([len(path), len(soln), time])
                    # print(str(len(path)) + ", "+str(len(soln)) + ", "+str(time), end=", ")
                    # print(f"Time: {time}")
                    # print(f"Soln size: {len(soln)}")
                    # print(f"Path size: {len(path)}")
                    # print()
                # render2DGrid(*gridDim, obstacles, new_graph.nodes, rem_obstacles=soln)
            if True:
                # print(src, ",", dst, ", ", gridDim)
                report.extend([src, dst, gridDim])
                out = ', '.join(str(x) for x in report)
                print(out + '\n')
            else:
                print()
                print()
            sys.stdout.flush()
        print()
        print()

        # for n_obstacles in n_obstacles_list:
        for n_obstacles in n_obstacles_list:
            evaluate(graph, gridDim, real_obstacles, n_obstacles, src, dst, mcr_funcs)


if __name__ == '__main__':
    OUT_DIR = Path(sys.argv[1])
    # TODO: Graph compaction for nodes with same coverage
    gridDim = (100, 100)
    gridDim_list = [(x, x) for x in range(100, 169, 10)]
    n_obstacles_list = [100] + list(range(500, 1100, 100))
    n_obstacles_list = [100] + list(range(200, 2100, 200))
    n_obstacles_list = [10]
    n_obstacles = n_obstacles_list[-1]
    # real_obstacles = randomRectangularHeuristicModel(gridDim, n_obstacles)
    # mcr_funcs = [dd_mcr2, dd_mcr3, exact_mcr, ids_mc_mcr2, ids_ed_mcr2, greedy_mcr1, greedy_mcr2, aco_mcr, saco_mcr]
    # mcr_funcs = [dd_mcr2, dd_mcr3, exact_mcr, greedy_mcr1, greedy_mcr2, aco_mcr]
    # mcr_funcs = [dd_mcr2, dd_mcr3, greedy_mcr1, greedy_mcr2, aco_mcr]
    timeout = None
    # mcr_funcs = [dd_mcr2(timeout), dd_mcr3(timeout), exact_mcr(timeout), greedy_mcr1(timeout), greedy_mcr2(timeout), aco_mcr(timeout)]
    mcr_funcs = [exact_mcr(timeout),
                 greedy_mcr1(timeout),
                 # dd_mcr(timeout, DD_VARIANT.ONEMIN),
                 # dd_mcr(timeout, DD_VARIANT.VANILLA),
                 dd_mcr(timeout, DD_VARIANT.OPDD),
                 # dd_mcr(timeout, DD_VARIANT.OPDD, repeat=3)]  # independent 4
                 # dd_mcr(timeout, DD_VARIANT.PROBDD),
                 # dd_mcr(timeout, DD_VARIANT.CDD)]
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, max_depth=2, repeat=1),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, max_depth=2, repeat=2),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, max_depth=2, repeat=4),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, max_depth=2, repeat=8),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, max_depth=2, repeat=16),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, max_depth=2, repeat=32)]
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, 0),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, 1),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, 2),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, 3),
                 # dd_mcr(timeout, DD_VARIANT.TWOPHASE, max_depth=4)]
    # mcr_funcs = [greedy_mcr1(timeout), exact_mcr(timeout)]
    # mcr_funcs = [dd_mcr2(timeout), dd_mcr3(timeout), aco_mcr(timeout)]
    # for gridDim in gridDim_list:
    # mcr_funcs = [dd_mcr(timeout, DD_VARIANT.OPDD, repeat=100)]
    processes = list()
    # for file in ["./maps/random-64-64-20.map"]:
    # for file in ["./maps/empty-16-16.map"]:
    for file in Path("./maps").iterdir():
        print(file)
        gridDim, graph = load_map(file)
        # real_obstacles = randomRectangularObstacleModel(gridDim, n_obstacles)
        #real_obstacles = independentObstacleModel(gridDim, 50, n_obstacles)
        obstacleModel = independentObstacleModel
        obstacleModel = randomRectangularObstacleModel
        #obstacleModel = randomPolygonModel
        real_obstacles = obstacleModel(gridDim, n_obstacles)
        # real_obstacles = obstacleModel(gridDim, 50, n_obstacles)
        real_obstacles = [frozenset(tuple(vertex) for vertex in obstacle) for obstacle in real_obstacles]
        # src = tuple(0 for _ in gridDim)
        # dst = tuple([x-1 for x in gridDim])
        # src = tuple(random.randrange(length) for length in gridDim)
        # dst = tuple(random.randrange(length) for length in gridDim)
        src, dst = random.sample(graph.nodes, k=2)

        # graph = nx.grid_graph(gridDim)
        node_mapping = {node:tuple(node) for node in graph.nodes}
        graph = nx.relabel_nodes(graph, node_mapping)
        def evaluate(graph, gridDim, real_obstacles, n_obstacles, src, dst, mcr_funcs):
            obstacles = real_obstacles[:n_obstacles]

            # render2DGrid(*gridDim, obstacles)

            new_graph = nx.Graph(graph)
            VertexSetConstraintChangesFactory(obstacles)(nx.Graph(graph)).remove(new_graph, obstacles)
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

            # src, dst = random.sample(aux_graph.nodes, k=2)
            aux_graph = nx.freeze(aux_graph)

            obs_graph = nx.Graph()
            for obstacle in obstacles:
                obs_graph.add_node(obstacle)

            list_obstacles = list(obstacles)
            for i, u in enumerate(list_obstacles):
                for v in list_obstacles[i + 1:]:
                    if len(u & v):
                        obs_graph.add_edge(u, v)
            communities = nx.community.louvain_communities(obs_graph)


            REPORT = True
            report = []
            report.append(str(file))
            report.append(str(obstacleModel.__name__))
            report.append(len(obstacles))
            if REPORT:
                print(f"Src: {src}")
                print(f"Dst: {dst}")
            for i, mcr_func in enumerate(mcr_funcs):
                # if i==2:
                #     obstacles = list(itertools.chain.from_iterable(communities))
                # elif i==3:
                #     obstacles = list(reverse_cuthill_mckee_ordering(obs_graph))
                # elif i==4:
                #     obstacles.sort(key=len)
                print(mcr_func)
                # if not False:
                #     if hasattr(mcr_func, '__name__'):
                #         print(mcr_func.__name__)
                #         print(len(mcr_func.__name__)*"=")
                #     elif hasattr(mcr_func, '__class__'):
                #         print(mcr_func.__class__.__name__)
                #         print(len(mcr_func.__class__.__name__)*"=")

                soln = mcr_func(aux_graph, aux_src, aux_dst, obstacles)
                if soln is None:
                    soln = set(obstacles)
                time = timeit.timeit
                if i == 0:
                    time = 0
                else:
                    time = timeit.timeit("mcr_func(aux_graph, aux_src, aux_dst, obstacles)", number=5, globals=locals())/5
                remaining_obstacles = set(obstacles) - set(soln)
                new_graph = nx.Graph(aux_graph)
                vertices = [vertex for subset in remaining_obstacles for vertex in subset]
                new_graph.remove_nodes_from(vertices)
                assert aux_src in new_graph.nodes
                assert aux_dst in new_graph.nodes
                path = nx.shortest_path(new_graph, source=aux_src, target=aux_dst)
                if True:
                    report.extend([len(path), len(soln), time])
                    # print(str(len(path)) + ", "+str(len(soln)) + ", "+str(time), end=", ")
                    print(f"Time: {time}")
                    print(f"Soln size: {len(soln)}")
                    print(f"Path size: {len(path)}")
                    print()
                # render2DGrid(*gridDim, obstacles, new_graph.nodes, rem_obstacles=soln)
            if True:
                # print(src, ",", dst, ", ", gridDim)
                report.extend([src, dst, gridDim])
                out = ', '.join(str(x) for x in report)
                with open(OUT_DIR/f"{os.path.basename(file)}_{n_obstacles}.txt", 'a') as f:
                    f.write(out + '\n')
            else:
                print()
                print()
            sys.stdout.flush()
        print()
        with open(OUT_DIR/f"{os.path.basename(file)}_{n_obstacles}.txt", 'a') as f:
            f.write('\n')

        # for n_obstacles in n_obstacles_list:
        for n_obstacles in n_obstacles_list:
            evaluate(graph, gridDim, real_obstacles, n_obstacles, src, dst, mcr_funcs)
            p = Process(target=evaluate, args=(graph, gridDim, real_obstacles, n_obstacles, src, dst, mcr_funcs))
            #processes.append(p)
            #p.start()

    for p in processes:
        p.join()
        pass