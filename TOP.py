from enum import Enum
import heapq
import itertools
from multiprocessing import Process
import os
from pathlib import Path
import random
import sys
import timeit
from functools import lru_cache

import networkx as nx

from Anytime import AnytimeAlgorithm

random.seed(42)


def top_preprocess(graph, src, dst, k, m):
    d1 = nx.single_source_dijkstra_path_length(graph, src, cutoff=m, weight='weight')
    d2 = nx.single_source_dijkstra_path_length(graph, dst, cutoff=m, weight='weight')
    nodes = d1.keys() & d2.keys()
    nodes = [node for node in nodes if d1[node] + d2[node] <= m]
    return graph.subgraph(nodes)

def read_top_instance(file_path):
    with open(file_path, 'r') as file:
        # Read the first three lines
        n_line = file.readline().strip()
        m_line = file.readline().strip()
        tmax_line = file.readline().strip()

        # Parse n N
        _, N = n_line.split(';')
        N = int(N)

        # Parse m P
        _, P = m_line.split(';')
        P = int(P)

        # Parse tmax Tmax
        _, Tmax = tmax_line.split(';')
        Tmax = float(Tmax)

        # Read the remaining lines for each node
        nodes = []
        for i in range(N):
            line = file.readline().strip()
            if line == '':
                continue  # Skip empty lines
            x_str, y_str, s_str = line.split(';')
            x = float(x_str)
            y = float(y_str)
            s = float(s_str)
            nodes.append({'id': i, 'x': x, 'y': y, 'score': s})

        return N, P, Tmax, nodes


# https://en.wikipedia.org/wiki/Multifit_algorithm
def multifit_partition(lst, scores, k):
    # items = sorted(lst, key=lambda x: scores[x], reverse=True)
    items = lst
    weights = [scores[x] for x in items]
    total_weight = sum(weights)
    max_weight = max(weights)

    def can_pack(capacity):
        bins = k * [0]
        for w in weights:
            for bin_i in bins:
                if bin_i + w <= capacity:
                    bin_i += w
                    break
            else:
                return False
        return True

    # Binary search for the minimum feasible capacity
    eps = 1e-5
    left, right = max_weight, total_weight
    best_cap = right

    while right - left > eps:
        mid = (left + right) / 2
        if can_pack(mid):
            best_cap = mid
            right = mid
        else:
            left = mid

    # Perform actual packing with best_cap
    partitions = [list() for _ in range(k)]
    bin_loads = k * [0]

    for item, w in zip(items, weights):
        # First-Fit Decreasing: place in first bin that can take it
        for i in range(k):
            if bin_loads[i] + w <= best_cap:
                partitions[i].append(item)
                bin_loads[i] += w
                break
        else:
            # If not placed (due to precision), place in emptiest bin (graceful degradation)
            idx = bin_loads.index(min(bin_loads))
            partitions[idx].append(item)
            bin_loads[idx] += w

    # return partitions, bin_loads, best_cap
    return [bin for bin in partitions if bin]


def greedy_partition(lst, scores, k):
    heap = [(0, list()) for _ in range(k)]
    for elem in lst:
        cur_score, cur_lst = heapq.heappop(heap)
        cur_lst.append(elem)
        heapq.heappush(heap, (cur_score + scores[elem], cur_lst))
    return [x[1] for x in heap if x[1]]


# Remove nodes that cannot be reached and returned in m
def load_map(filename):
    with open(filename, 'r') as f:
        _ = f.readline()
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
                if char in {'@', 'T'}:
                    graph.remove_node((i, j))
        return (height, width), graph


class DD_VARIANT(Enum):
    TWOPHASE = 2
    OPDD = 4
    ONEMIN = 7

class rand_top(AnytimeAlgorithm):
    def __init__(self, graph, src, dst, k, m, timeout=None, repeat=1):
        super().__init__(timeout)
        self.max_repeat = repeat
        self.nodes = {u: graph.nodes[u]["score"] for u in graph.nodes}
        self.graph = graph
        if src in self.nodes:
            del self.nodes[src]
        if dst in self.nodes:
            del self.nodes[dst]
        self.src = src
        self.dst = dst
        self.k = k
        self.m = m
        self.kpaths = None

    def _run(self):
        self.result = set()
        self.kpaths = None
        if self.graph.number_of_nodes() == 0:
            return
        if self.graph[self.src][self.dst]['weight'] > self.m:
            return
        vertices = list(self.nodes)

        if self.src in vertices:
            vertices.remove(self.src)
        if self.dst in vertices:
            vertices.remove(self.dst)

        self.kpaths = list()
        best_reward = 0

        for _ in range(self.max_repeat):
            cur_vertices = list(vertices)
            random.shuffle(cur_vertices)
            kpaths = list()
            total_reward = 0
            for k in range(self.k):
                c = 0
                j = self.src
                path = [self.src]
                for i in cur_vertices[:]:
                    if c + self.graph[j][i]['weight'] + self.graph[i][self.dst]['weight'] <= self.m:
                        c += self.graph[j][i]['weight']
                        cur_vertices.remove(i)
                        path.append(i)
                        total_reward += self.nodes[i]
                        j = i
                path.append(self.dst)
                # c += self.graph[j][self.dst]['weight']
                # kpaths.append((path, c))
                kpaths.append(path)

            if total_reward > best_reward:
                self.kpaths = kpaths
                best_reward = total_reward
                self.result = set(x for path  in kpaths for x in path)

        return self.result


class wdd_rand_top(AnytimeAlgorithm):
    def __init__(self, graph, src, dst, k, m, timeout=None,
                 variant=DD_VARIANT.OPDD, max_depth=0, repeat=1):
        super().__init__(timeout)
        self.max_depth = max_depth
        self.variant = variant
        self.max_repeat = repeat
        self.nodes = {u: graph.nodes[u]["score"] for u in graph.nodes}
        self.weight = nx.get_edge_attributes(graph, "weight")
        self.k = k
        self.m = m
        self.graph = graph
        if src in self.nodes:
            del self.nodes[src]
        if dst in self.nodes:
            del self.nodes[dst]
        self.src = src
        self.dst = dst
        self.kpaths = None

    def check(self, vertices):
        cur_vertices = list(vertices)
        kpaths = list()
        for k in range(self.k):
            c = 0
            j = self.src
            path = [self.src]
            for i in cur_vertices[:]:
                if c + self.graph[j][i]['weight'] + self.graph[i][self.dst]['weight'] <= self.m:
                    c += self.graph[j][i]['weight']
                    cur_vertices.remove(i)
                    path.append(i)
                    j = i
            path.append(self.dst)
            kpaths.append(path)

        return True, kpaths

    def _score(self, obstacles):
        score = sum(self.nodes[x] for x in obstacles if x not in [self.src, self.dst])
        return score

    def onemin(self, obstacles, removed=list()):
        removed = set(removed)
        obstacles = sorted(obstacles, key = lambda x: self.nodes[x], reverse=True)
        cur_obstacles = set(obstacles) | removed
        for obstacle in obstacles:
            cbar = set(cur_obstacles)
            cbar.remove(obstacle)
            t, kpaths = self.check(cbar | removed)
            # print("ITER:", min(self._score(self.result), self._score({x for path in kpaths for x in path})), t, self._score({x for path in kpaths for x in path}) <= self._score(self.result))
            if t:
                cur_vertices = {x for path in kpaths for x in path}
                if self._score(cur_vertices) > self._score(self.result):
                    self.result = cur_vertices
                    self.kpaths = kpaths
            # print("ITER:", self._score(self.result))
            assert self._score(self.result) != 0

        return list(cur_obstacles - removed)

    # One Pass Delta Debugging
    def opdd(self, obstacles, removed=list(), n=2):
        cur_obstacles = obstacles

        while True:

            if n >= len(cur_obstacles):
                # Ensure 1-minimality
                soln = self.onemin(cur_obstacles, removed=removed)
                return soln

            cs = multifit_partition(cur_obstacles, self.nodes, n)

            next_n = n

            for cs_i in cs:
                if len(cs_i) == 0:
                    continue
                cbar = [obstacle for obstacle in cur_obstacles if obstacle not in cs_i]
                t, kpaths = self.check(cbar + removed)
                # print("ITER:", min(self._score(self.result), self._score({x for path in kpaths for x in path})), t, self._score({x for path in kpaths for x in path}) <= self._score(self.result))

                if t:
                    cur_vertices = {x for path in kpaths for x in path}
                    next_obstacles = self.nodes.keys() - cur_vertices
                    if self._score(cur_vertices) > self._score(self.result):
                        self.result = cur_vertices
                        self.kpaths = kpaths
                    if next_obstacles >= set(cur_obstacles):
                        assert self._score(self.result) != 0
                        continue
                    next_obstacles = next_obstacles &  set(cur_obstacles)
                    next_n = next_n - 1
                    cur_obstacles = cbar
                assert self._score(self.result) != 0


            n = 2 * next_n


    def _run(self):
        self.result = set()
        # print("ITER:", self._score(self.result))
        # assert self._score(self.result) != 0
        self.kpaths = None

        if self.graph.number_of_nodes() == 0:
            return
        if self.graph[self.src][self.dst]['weight'] > self.m:
            self.result = set()
            return

        top_func = None
        if self.variant == DD_VARIANT.TWOPHASE:
            top_func = self.twophase
        elif self.variant == DD_VARIANT.OPDD:
            top_func = self.opdd
        elif self.variant == DD_VARIANT.ONEMIN:
            top_func = self.onemin

        obstacles = list(self.nodes)

        if self.src in obstacles:
            obstacles.remove(self.src)
        if self.dst in obstacles:
            obstacles.remove(self.dst)


        for _ in range(self.max_repeat):
            t, kpaths = self.check(obstacles)
            if not t:
                continue
            else:
                cur_vertices = {x for path in kpaths for x in path}
                if self._score(cur_vertices) > self._score(self.result):
                    self.result = cur_vertices
                    self.kpaths = kpaths

            _ = top_func(obstacles)
            random.shuffle(obstacles)
            if self.variant == DD_VARIANT.TWOPHASE:
                break

        return self.result

    def __str__(self):
        return f"WDD_{self.variant}_{self.max_repeat}_{self.max_depth}"



def load_graph(file_path):
    N, P, Tmax, nodes = read_top_instance(file_path)

    G = nx.Graph()

    for node_info in nodes:
        node_id = node_info['id']
        x = node_info['x']
        y = node_info['y']
        score = node_info['score']
        G.add_node(node_id, x=x, y=y, score=score)

    for i in range(N):
        for j in range(i + 1, N):
            dist = ((nodes[i]['x'] - nodes[j]['x'])**2
                    + (nodes[i]['y'] - nodes[j]['y'])**2) ** 0.5
            G.add_edge(i, j, weight=dist)

    return G, P, Tmax

if __name__ == '__main__':
    OUT_DIR = Path(sys.argv[1])
    random.seed(42)
    def evaluate(file_path):
        graph, P, Tmax = load_graph(file_path)
        src = 0
        dst = graph.number_of_nodes()-1
        graph = top_preprocess(graph, src, dst, P, Tmax)


        timeout = None
        top_funcs = []

        top_funcs = [
            rand_top(graph, src, dst, P, Tmax, timeout, repeat=10),
            wdd_rand_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.OPDD, repeat=3),
            ]
        print(file_path)
        print(20*"=")
        for top_func in top_funcs:
            soln = top_func()
            time = timeit.timeit("top_func()", number=5, globals=locals())/5
            soln_len = 0 if soln is None else sum(graph.nodes[x]['score'] for x in soln)
            print(top_func, "time:", time, "score:", soln_len)
            filename = os.path.basename(file_path)
            file_stem, _ = os.path.splitext(filename)
            with open(OUT_DIR/f"{file_stem}.txt", 'a') as f:
                f.write(f"{soln_len}, {time}, ")
        with open(OUT_DIR/f"{file_stem}.txt", 'a') as f:
            f.write("\n")
        print()

    processes = list()
    for file in itertools.chain(Path("./new_instances").iterdir(), Path("./Chao").iterdir()):
        print(file)
        evaluate(file)
        p = Process(target=evaluate, args=(file, ))
        # processes.append(p)
        # p.start()

    for p in processes:
        p.join()
