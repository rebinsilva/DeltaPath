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


def dijkstra_path(G, src, tgt, forbidden=set()):
    if src not in G.nodes or tgt not in G.nodes:
        raise nx.NodeNotFound

    dist = {node: float('inf') for node in G.nodes()}
    prev = {}
    dist[src] = 0
    visited = set(forbidden)
    if src in visited:
        visited.remove(src)
    heap = [(0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == tgt:
            break
        for v in G.neighbors(u):
            weight = G[u][v]['weight']
            alt = d + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    else:
        raise nx.NetworkXNoPath

    # Reconstruct path
    if tgt not in prev and tgt != src:
        return None  # no path

    path = []
    current = tgt
    while current != src:
        path.append(current)
        current = prev[current]
    path.append(src)
    path.reverse()
    return path

def dijkstra_multi_source_target_path(G, srcs, tgts, src_lens, forbidden=set()):
    if set(srcs) - set(G.nodes) or set(tgts) - set(G.nodes):
        raise nx.NodeNotFound

    dist = {node: float('inf') for node in G.nodes()}
    prev = {}
    heap = list()
    visited = set(forbidden)
    for i, src in enumerate(srcs):
        dist[src] = src_lens[i]
        heap.append((src_lens[i], src))
        if src in visited:
            visited.remove(src)

    tgt = None
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u in tgts:
            tgt = u
            break
        for v in G.neighbors(u):
            weight = G[u][v]['weight']
            alt = d + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    else:
        raise nx.NetworkXNoPath

    # Reconstruct path
    if tgt not in prev and tgt != src:
        return None  # no path

    path = []
    current = tgt
    while current not in srcs:
        path.append(current)
        current = prev[current]
    path.append(current)
    path.reverse()
    return path

def dijkstra_shortest_target_path(G, src, tgts, forbidden=set()):
    if src not in G.nodes or set(tgts) - set(G.nodes):
        raise nx.NodeNotFound('1')

    dist = {node: float('inf') for node in G.nodes()}
    prev = {}
    dist[src] = 0
    visited = set(forbidden)
    if src in visited:
        visited.remove(src)
    heap = [(0, src)]

    tgt = None
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u in tgts:
            tgt = u
            break
        for v in G.neighbors(u):
            weight = G[u][v]['weight']
            alt = d + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    else:
        raise nx.NetworkXNoPath

    # Reconstruct path
    if tgt not in prev and tgt != src:
        return None  # no path

    path = []
    current = tgt
    while current != src:
        path.append(current)
        current = prev[current]
    path.append(src)
    path.reverse()
    return path

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


# Ideas
# False
# Simulated Aneealing with crystofieds

# True
# Initial based on dijkstra on vertices in set
# Build on top of initial set

# sel_nodes should not have src and dst
def has_atmost_kpaths_provider(graph, src, dst, k, m):
    aux_graph = graph.copy()
    pre_lst = [src, dst]

    if graph.number_of_nodes() == 0:
        return
    # for i in range(1, 1 + k//2):
    #     new_src = (src, i)
    #     aux_graph.add_node(new_src)
    #     pre_lst.append(new_src)
    #     for v, edge_attributes in graph[src].items():
    #         weight = edge_attributes['weight']
    #         aux_graph.add_edge(new_src, v, weight=weight)
    for i in range(1, (1 + k) // 2):
        new_src = (src, i)
        new_dst = (dst, i)
        aux_graph.add_node(new_src)
        aux_graph.add_node(new_dst)
        pre_lst.append(new_src)
        pre_lst.append(new_dst)
        for v, edge_attributes in graph[src].items():
            weight = edge_attributes['weight']
            aux_graph.add_edge(new_src, v, weight=weight)
        for v, edge_attributes in graph[dst].items():
            weight = edge_attributes['weight']
            aux_graph.add_edge(new_dst, v, weight=weight)

    disjoint_paths = list(nx.node_disjoint_paths(graph, src, dst, cutoff=k))

    for i, path in enumerate(disjoint_paths[:]):
        if nx.path_weight(graph, path, 'weight') > m:
            del disjoint_paths[i]

    @lru_cache(256)
    def has_atmost_kpaths(sel_nodes):
        # src_steiner = nx.steiner_tree(graph, nodes | {src,})
        # src_steiner_weight = sum(data["weight"] for node, data in src_steiner.nodes(data=True))
        # dst_steiner = nx.steiner_tree(graph, nodes | {dst,})
        # dst_steiner_weight = sum(data["weight"] for node, data in dst_steiner.nodes(data=True))
        # nxt_src_list = set(graph.neighbors(src)) & nodes
        # A hybrid adaptive large neighborhood search heuristic for the team orienteering problem

        chosen_paths = list()


        tsp_nodes = list(sel_nodes) + pre_lst

        # TODO Add Simulated Annealing with crystofides as initial seed
        tsp = nx.approximation.traveling_salesman_problem
        tsp_path = tsp(aux_graph, nodes=tsp_nodes, cycle=(k % 2 == 0))
        tsp_path_weight = nx.path_weight(aux_graph, tsp_path, 'weight')
        if k * m < 2 * tsp_path_weight / 3:
            return (False, chosen_paths)

        uncovered = set(sel_nodes)
        visited = list()

        # for _ in range(k):
        #     if not uncovered:
        #         return (True, chosen_paths)
        #     best_path = None
        #     best_cover_len = 0

        #     for u in list(uncovered):
        #         try:
        #             path1 = dijkstra_path(graph, src, u, forbidden=visited)
        #             cost1 = nx.path_weight(graph, path1, 'weight')
        #         except (nx.NetworkXNoPath, nx.NodeNotFound):
        #             continue

        #         del path1[-1]

        #         try:
        #             path2 = dijkstra_path(graph, u, dst, forbidden=visited + path1)
        #             cost2 = nx.path_weight(graph, path2, 'weight')
        #         except (nx.NetworkXNoPath, nx.NodeNotFound):
        #             continue

        #         total_cost = cost1 + cost2
        #         if total_cost > m:
        #             continue

        #         full_path = path1 + path2

        #         new_covered = len(set(full_path) & uncovered)
        #         if new_covered > best_cover_len:
        #             best_cover_len = new_covered
        #             best_path = full_path

        #     if best_path is None:
        #         return (False, chosen_paths)

        #     chosen_paths.append(best_path)
        #     visited += best_path[1:-1]
        #     best_path = set(best_path)
        #     uncovered -= best_path

        if not sel_nodes:
            return (len(disjoint_paths) == k, disjoint_paths)

        kpaths = k*[None]
        visited = set()
        uncovered = set(sel_nodes)
        for i in range(k):
            try:
                path1 = dijkstra_shortest_target_path(graph, src, uncovered, forbidden=visited)
                cost1 = nx.path_weight(graph, path1, 'weight')
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return (False, chosen_paths)

            visited.update(path1[1:])
            uncovered.remove(path1[-1])

            try:
                path2 = dijkstra_path(graph, path1[-1], dst, forbidden = visited)
                cost2 = nx.path_weight(graph, path2, 'weight')
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return (False, chosen_paths)
            if cost1 + cost2 > m:
                return (False, chosen_paths)
            kpaths[i] = (path1, cost1)

        while uncovered:
            srcs = [path[0][-1] for path in kpaths]
            src_lens = [path[1] for path in kpaths]
            try:
                path1 = dijkstra_multi_source_target_path(graph, srcs, uncovered, src_lens, forbidden=visited)
                cost1 = nx.path_weight(graph, path1, 'weight')
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return (False, chosen_paths)
            visited.update(path1)
            uncovered.remove(path1[-1])
            try:
                path2 = dijkstra_path(graph, path1[-1], dst, forbidden = visited)
                cost2 = nx.path_weight(graph, path2, 'weight')
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return (False, chosen_paths)
            if cost1 + cost2 > m:
                return (False, chosen_paths)
            i = None
            for j in range(k):
                if kpaths[j][0][-1] == path1[0]:
                    i = j
                    break
            cur_path = kpaths[i][0] + path1[1:]
            cur_cost = kpaths[i][1] + cost1
            kpaths[i] = (cur_path, cur_cost)

        kpaths.sort(key = lambda x: x[1], reverse=True)
        for i, (path1, cost1) in enumerate(kpaths):
            try:
                path2 = dijkstra_path(graph, path1[-1], dst, forbidden = visited)
                cost2 = nx.path_weight(graph, path2, 'weight')
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return (False, chosen_paths)
            visited.update(path2[:-1])
            if cost1 + cost2 > m:
                return (False, chosen_paths)
            cur_path = kpaths[i][0] + path1[1:]
            cur_cost = kpaths[i][1] + cost1
            kpaths[i] = (cur_path, cur_cost)
        else:
            return (True, [path for path, cost in kpaths])


        return (False, chosen_paths)


    return has_atmost_kpaths


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

class greedy_top(AnytimeAlgorithm):
    def __init__(self, graph, src, dst, k, m, timeout=None):
        super().__init__(timeout)
        self._check = has_atmost_kpaths_provider(graph, src, dst, k, m)
        self.nodes = {u: graph.nodes[u]["score"] for u in graph.nodes}
        # print(self.nodes)
        self.weight = nx.get_edge_attributes(graph, "weight")
        self.m = m
        self.graph = graph
        if src in self.nodes:
            del self.nodes[src]
        if dst in self.nodes:
            del self.nodes[dst]
        self.src = src
        self.dst = dst
        self.check = lambda x: self._check(tuple(self.nodes.keys() - set(x)))
        self.kpaths = None
        self.result = set()

    def _score(self, obstacles):
        return sum(self.nodes[x] for x in obstacles if x not in [self.src, self.dst])

    def _run(self):
        self._check.cache_clear()
        self.result = set()
        if self.graph.number_of_nodes() == 0:
            return
        if self.graph[self.src][self.dst]['weight'] > self.m:
            return

        obstacles = list(self.nodes)

        if self.src in obstacles:
            obstacles.remove(self.src)
        if self.dst in obstacles:
            obstacles.remove(self.dst)

        t, kpaths = self.check(obstacles)
        if t:
            cur_vertices = {x for path in kpaths for x in path}
            if self._score(cur_vertices) > self._score(self.result):
                self.result = cur_vertices
                self.kpaths = kpaths


        return self.result

class wdd_rand_top(AnytimeAlgorithm):
    def __init__(self, graph, src, dst, k, m, timeout=None,
                 variant=DD_VARIANT.OPDD, max_depth=0, repeat=1):
        super().__init__(timeout)
        self.max_depth = max_depth
        self.variant = variant
        self.max_repeat = repeat
        self.nodes = {u: graph.nodes[u]["score"] for u in graph.nodes}
        # print(self.nodes)
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

    # def _score(self, kpaths):
    #     if len (kpaths) == 0:
    #         return 0

    #     if isinstance(kpaths[0], list):
    #         all_vertices = {x for path in kpaths for x in path}
    #         return sum(self.nodes[x] for x in all_vertices)
    #     else:
    #         obstacles = kpaths
    #         return sum(self.nodes[x] for x in obstacles)

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

        #print(len(cur_obstacles[:len(cur_obstacles)-len(removed)]))
        return list(cur_obstacles - removed)

    # One Pass Delta Debugging
    def opdd(self, obstacles, removed=list(), n=2):
        cur_obstacles = obstacles

        # We replace the tail recursion from the paper by a loop
        while True:
            # tc = reachability(graph, src, dst, cur_obstacles + removed)
            # assert tc

            if n >= len(cur_obstacles):
                # Ensure 1-minimality
                soln = self.onemin(cur_obstacles, removed=removed)
                return soln

            # print(len(cur_obstacles), n)
            cs = multifit_partition(cur_obstacles, self.nodes, n)
            # cs = greedy_partition(cur_obstacles, self.nodes, n)
            # print(sorted([sum(x) for x in cs]))

            next_n = n

            for cs_i in cs:
                if len(cs_i) == 0:
                    continue
                # print("        ", len(cur_obstacles), len(cs_i))
                cbar = [obstacle for obstacle in cur_obstacles if obstacle not in cs_i]
                # k, m = divmod(len(cur_obstacles), n)
                # test = cur_obstacles[:i*k+min(i, m)] + cur_obstacles[(i+1)*k+min(i+1, m):]
                # assert sorted(test) == sorted(cbar)
                t, kpaths = self.check(cbar + removed)
                # print("ITER:", min(self._score(self.result), self._score({x for path in kpaths for x in path})), t, self._score({x for path in kpaths for x in path}) <= self._score(self.result))

                if t:
                    cur_vertices = {x for path in kpaths for x in path}
                    next_obstacles = self.nodes.keys() - cur_vertices
                    if self._score(cur_vertices) > self._score(self.result):
                        self.result = cur_vertices
                        self.kpaths = kpaths
                    if next_obstacles >= set(cur_obstacles):
                        # print("ITER:", self._score(self.result))
                        assert self._score(self.result) != 0
                        continue
                    next_obstacles = next_obstacles &  set(cur_obstacles)
                    next_n = next_n - 1
                    cur_obstacles = cbar
                # print("ITER:", self._score(self.result))
                assert self._score(self.result) != 0


            n = 2 * next_n

    def twophase(self, cur_obstacles, removed=list(), _depth=0):
        all_solns = set()
        for _ in range(self.max_repeat):
            cur_solns = list()
            list_c = list(cur_obstacles)
            while True:
                t, kpaths = self.check(list_c + removed)
                if not t:
                    break
                soln = self.opdd(list_c, removed=removed)
                # if self._score(soln) + self._score(removed) < self._score(self.nodes.keys() - self.result):
                #     self.result = self.nodes.keys() - (soln | set(removed))
                #     self.kpaths = kpaths
                set_soln = frozenset(soln)
                list_c = [x for x in list_c if x not in set_soln]
                if set_soln not in all_solns:
                    cur_solns.append(soln)
                    all_solns.add(set_soln)

            if _depth < self.max_depth:
                all_soln = set()
                all_soln = all_soln.union(*cur_solns)
                list_c = [x for x in cur_obstacles if x not in all_soln]
                for i, soln in enumerate(list(cur_solns)):
                    irreplaceable_deltas = self.opdd(soln, removed=list_c+removed)
                    if self._score(irreplaceable_deltas) + self._score(removed) >= self._score(self.nodes.keys() - self.result):
                        continue
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Pre Opt Soln size: ", len(cur_solns[i]) + len(removed))
                    verbose=0
                    if self._score(irreplaceable_deltas) == self._score(soln):
                        verbose=0
                        if verbose:
                            print(_depth*"  "+"Post Opt Soln size: ", len(cur_solns[i]) + len(removed), " (Irreplaceable)")
                        verbose=0
                        continue
                    result = self.twophase(list_c, removed=irreplaceable_deltas+removed, _depth=_depth+1)
                    if self._score(result) < self._score(soln) - self._score(irreplaceable_deltas):
                        cur_solns[i] = result + irreplaceable_deltas
                        # if self._score(solns[i]) + self._score(removed) < self._score(self.nodes.keys() - self.result):
                        #     self.result = self.nodes.keys() - set(solns[i] + removed)
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Post Opt Soln size: ", len(cur_solns[i]) + len(removed))
                    verbose=0
            random.shuffle(cur_obstacles)

        return list(min(all_solns, key = lambda x: sum(x)))

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
            print()
            random.shuffle(obstacles)
            if self.variant == DD_VARIANT.TWOPHASE:
                break

        return self.result

    def __str__(self):
        return f"Greedy_{self.variant}_{self.max_repeat}_{self.max_depth}"

class wdd_top(AnytimeAlgorithm):
    def __init__(self, graph, src, dst, k, m, timeout=None,
                 variant=DD_VARIANT.OPDD, max_depth=0, repeat=1):
        super().__init__(timeout)
        self.max_depth = max_depth
        self.variant = variant
        self.max_repeat = repeat
        self._check = has_atmost_kpaths_provider(graph, src, dst, k, m)
        self.nodes = {u: graph.nodes[u]["score"] for u in graph.nodes}
        # print(self.nodes)
        self.weight = nx.get_edge_attributes(graph, "weight")
        self.m = m
        self.graph = graph
        if src in self.nodes:
            del self.nodes[src]
        if dst in self.nodes:
            del self.nodes[dst]
        self.src = src
        self.dst = dst
        self.check = lambda x: self._check(tuple(self.nodes.keys() - set(x)))
        self.kpaths = None

    def _score(self, obstacles):
        score = sum(self.nodes[x] for x in obstacles if x not in [self.src, self.dst])
        return score

    # def _score(self, kpaths):
    #     if len (kpaths) == 0:
    #         return 0

    #     if isinstance(kpaths[0], list):
    #         all_vertices = {x for path in kpaths for x in path}
    #         return sum(self.nodes[x] for x in all_vertices)
    #     else:
    #         obstacles = kpaths
    #         return sum(self.nodes[x] for x in obstacles)

    def onemin(self, obstacles, removed=list()):
        removed = set(removed)
        obstacles = sorted(obstacles, key = lambda x: self.nodes[x], reverse=True)
        cur_obstacles = set(obstacles) | removed
        for obstacle in obstacles:
            cbar = set(cur_obstacles)
            cbar.remove(obstacle)
            t, kpaths = self.check(cbar | removed)
            if t:
                cur_vertices = {x for path in kpaths for x in path}
                if self._score(cur_vertices) > self._score(self.result):
                    self.result = cur_vertices
                    self.kpaths = kpaths
            #print("ITER:", self._score(self.result))

        #print(len(cur_obstacles[:len(cur_obstacles)-len(removed)]))
        return list(cur_obstacles - removed)

    # One Pass Delta Debugging
    def opdd(self, obstacles, removed=list(), n=2):
        cur_obstacles = obstacles

        # We replace the tail recursion from the paper by a loop
        while True:
            # tc = reachability(graph, src, dst, cur_obstacles + removed)
            # assert tc

            if n >= len(cur_obstacles):
                # Ensure 1-minimality
                soln = self.onemin(cur_obstacles, removed=removed)
                return soln

            cs = multifit_partition(cur_obstacles, self.nodes, n)
            # cs = greedy_partition(cur_obstacles, self.nodes, n)
            # print(sorted([sum(x) for x in cs]))

            next_n = n

            for cs_i in cs:
                cbar = [obstacle for obstacle in cur_obstacles if obstacle not in cs_i]
                # k, m = divmod(len(cur_obstacles), n)
                # test = cur_obstacles[:i*k+min(i, m)] + cur_obstacles[(i+1)*k+min(i+1, m):]
                # assert sorted(test) == sorted(cbar)
                t, kpaths = self.check(cbar + removed)

                if t:
                    cur_vertices = {x for path in kpaths for x in path}
                    next_obstacles = self.nodes.keys() - cur_vertices
                    if next_obstacles >= set(cur_obstacles):
                        # print("ITER:", self._score(self.result))
                        continue
                    if self._score(cur_vertices) > self._score(self.result):
                        self.result = cur_vertices
                        self.kpaths = kpaths
                    cur_obstacles = cbar
                    next_n = next_n - 1
                # print("ITER:", self._score(self.result))


            n = min(len(cur_obstacles), next_n * 2)

    def twophase(self, cur_obstacles, removed=list(), _depth=0):
        all_solns = set()
        for _ in range(self.max_repeat):
            cur_solns = list()
            list_c = list(cur_obstacles)
            while True:
                t, kpaths = self.check(list_c + removed)
                if not t:
                    break
                soln = self.opdd(list_c, removed=removed)
                # if self._score(soln) + self._score(removed) < self._score(self.nodes.keys() - self.result):
                #     self.result = self.nodes.keys() - (soln | set(removed))
                #     self.kpaths = kpaths
                set_soln = frozenset(soln)
                list_c = [x for x in list_c if x not in set_soln]
                if set_soln not in all_solns:
                    cur_solns.append(soln)
                    all_solns.add(set_soln)

            if _depth < self.max_depth:
                all_soln = set()
                all_soln = all_soln.union(*cur_solns)
                list_c = [x for x in cur_obstacles if x not in all_soln]
                for i, soln in enumerate(list(cur_solns)):
                    irreplaceable_deltas = self.opdd(soln, removed=list_c+removed)
                    if self._score(irreplaceable_deltas) + self._score(removed) >= self._score(self.nodes.keys() - self.result):
                        continue
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Pre Opt Soln size: ", len(cur_solns[i]) + len(removed))
                    verbose=0
                    if self._score(irreplaceable_deltas) == self._score(soln):
                        verbose=0
                        if verbose:
                            print(_depth*"  "+"Post Opt Soln size: ", len(cur_solns[i]) + len(removed), " (Irreplaceable)")
                        verbose=0
                        continue
                    result = self.twophase(list_c, removed=irreplaceable_deltas+removed, _depth=_depth+1)
                    if self._score(result) < self._score(soln) - self._score(irreplaceable_deltas):
                        cur_solns[i] = result + irreplaceable_deltas
                        # if self._score(solns[i]) + self._score(removed) < self._score(self.nodes.keys() - self.result):
                        #     self.result = self.nodes.keys() - set(solns[i] + removed)
                    verbose=0
                    if verbose:
                        print(_depth*"  "+"Post Opt Soln size: ", len(cur_solns[i]) + len(removed))
                    verbose=0
            random.shuffle(cur_obstacles)

        return list(min(all_solns, key = lambda x: sum(x)))

    def _run(self):
        self._check.cache_clear()
        self.result = set()
        self.kpaths = None
        # print("ITER:", self._score(self.result))

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
        return f"{self.variant}_{self.max_repeat}_{self.max_depth}"


def load_graph(file_path):
    N, P, Tmax, nodes = read_top_instance(file_path)

    G = nx.Graph()

    for node_info in nodes:
        node_id = node_info['id']
        x = node_info['x']
        y = node_info['y']
        score = node_info['score']
        # Add to the graph with attributes
        G.add_node(node_id, x=x, y=y, score=score)

    for i in range(N):
        for j in range(i + 1, N):
            # Could compute Euclidean distance, or read from file if provided
            dist = ((nodes[i]['x'] - nodes[j]['x'])**2
                    + (nodes[i]['y'] - nodes[j]['y'])**2) ** 0.5
            G.add_edge(i, j, weight=dist)

    return G, P, Tmax

if __name__ == '__main__':
    OUT_DIR = Path(sys.argv[1])
    # file_path = 'new_instances/bier127_gen1_m3.txt'
    # file_path = 'new_instances/pr136_gen2_m2.txt'
    # file_path = 'Chao/p1.2.n.txt'
    random.seed(42)
    def evaluate(file_path):
        graph, P, Tmax = load_graph(file_path)
        # src, dst = random.sample(graph.nodes, k=2)
        # while nx.shortest_path_length(graph, src, dst, weight='weight') >= Tmax:
        #     src, dst = random.sample(graph.nodes, k=2)
        src = 0
        dst = graph.number_of_nodes()-1
        graph = top_preprocess(graph, src, dst, P, Tmax)
        print(src, dst, P, Tmax)

        check = has_atmost_kpaths_provider(graph, src, dst, P, Tmax)

        timeout = None
        top_funcs = []
        for dep in range(6):
            for rep in range(1, 6):
                top_funcs.append(wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=rep, max_depth=dep))

        top_funcs = [
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=2, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=3, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=4, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=5, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=1),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=2),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=3),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=4),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=5),
                     ]
        top_funcs = [
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=3, max_depth=0),
            wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=5, max_depth=0),
            # wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=1, max_depth=3),
            # wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.TWOPHASE, repeat=3, max_depth=3),
                     ]
        # top_funcs = [wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.OPDD, repeat=3, max_depth=1)]
        top_funcs = [
            rand_top(graph, src, dst, P, Tmax, timeout, repeat=10),
            #greedy_top(graph, src, dst, P, Tmax, timeout),
            wdd_rand_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.OPDD, repeat=5),
            # wdd_top(graph, src, dst, P, Tmax, timeout, DD_VARIANT.OPDD, repeat=10),
            ]
        for top_func in top_funcs:
            print(top_func)
            time = timeit.timeit
            # time = timeit.timeit("top_func()", number=5, globals=locals())/5
            soln = top_func()
            print(time)
            print(top_func.kpaths)
            # print(check(soln))
            soln_len = 0 if soln is None else sum(graph.nodes[x]['score'] for x in soln)
            print(soln)
            print(soln_len)
            if top_func.kpaths is not None:
                all_vertices = set(x for path in top_func.kpaths for x in path)
                soln_len = sum(graph.nodes[x]['score'] for x in all_vertices)
                print(soln_len)
            else:
                print(0)
            filename = os.path.basename(file_path)
            file_stem, _ = os.path.splitext(filename)
            with open(OUT_DIR/f"{file_stem}.txt", 'a') as f:
                f.write(f"{soln_len}, {time}, ")
        with open(OUT_DIR/f"{file_stem}.txt", 'a') as f:
            f.write("\n")
        print()

    processes = list()
    for file in itertools.chain(Path("./new_instances").iterdir(), Path("./Chao").iterdir()):
    # for file in ['new_instances/kroB200_gen3_m2.txt']:
    # for file in ['Chao/p5.4.e.txt']:
    # for file in Path("./new_instances").iterdir():
    # for file in Path("./Chao").iterdir():
        print(file)
        evaluate(file)
        p = Process(target=evaluate, args=(file, ))
        # processes.append(p)
        # p.start()

    for p in processes:
        p.join()
