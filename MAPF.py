from collections import deque
from enum import Enum
import heapq
from itertools import zip_longest
from multiprocessing import Process
import os
from pathlib import Path
import random
import sys

import networkx as nx
from tqdm import tqdm

from Anytime import AnytimeAlgorithm

TQDM = False
TRANSPOSE=False

if not TQDM:
    def tqdm(x):
        return x

random.seed(42)

def load_map(filename):
    with open(filename, 'r') as f:
        map_type = f.readline().strip().split()
        assert map_type[1] == "octile"
        height = f.readline()
        height = int(height.split(' ')[1])
        width = f.readline()
        width = int(width.split(' ')[1])
        _ = f.readline()
        graph = nx.grid_graph((height, width))
        for i in range(height):
            row = f.readline()
            for j, char in enumerate(row):
                if j >= width:
                    break
                # https://web.archive.org/web/20160304051410id_/http://web.cs.du.edu/~sturtevant/papers/benchmarks.pdf
                if char in {'@', 'O', 'T', 'W'}:
                    if TRANSPOSE:
                        graph.remove_node((i, j))
                    else:
                        graph.remove_node((j, i))
                    # print(j, i)
        return (width, height), graph

def load_scenario(filepath, bucketed=False):
    filepath = Path(filepath)
    filename = '-'.join(filepath.name.split('-')[:-2]) + ".map"
    print(filepath)
    map_path = filepath.parent/filename
    assert map_path.exists()

    (width, height), graph = load_map(map_path)
    queries = list()
    buckets = dict()
    with open(filepath, 'r') as f:
        version = f.readline().strip()
        assert version == "version 1"
        for line in f:
            line = line.strip()
            if line.strip() == '':
                break
            bucket, mapname, width, height, src_x, src_y, dst_x, dst_y, opt_len = line.split()
            bucket = int(bucket)
            width = int(width)
            height = int(height)
            src = (int(src_x), int(src_y))
            dst = (int(dst_x), int(dst_y))
            opt_len = float(opt_len)
            query = (src, dst)
            queries.append(query)
            buckets.setdefault(bucket, list()).append(query)
    if not bucketed:
        return (height, width), graph, queries
    else:
        return (height, width), graph, buckets


def bfs(G, src, tgt, allowed_edges=set(), forbidden=set()):
    worklist = deque()
    worklist.append(src)
    explored = set(forbidden)

    while worklist:
        u = worklist.popleft()
        if u == tgt:
            return True
        explored.add(u)
        for v in G.neighbors(u):
            if v in explored:
                continue
            if (u,v) in allowed_edges or (v, u) in allowed_edges:
                worklist.append(v)
    else:
        return False


def dijkstra_path(G, src, tgt, allowed_edges=set(), forbidden=set(), final=dict()):
    if src not in G.nodes or tgt not in G.nodes:
        raise nx.NodeNotFound

    dist = {node: float('inf') for node in G.nodes()}
    prev = {}
    dist[src] = 0
    visited = set()
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
            if (u,v) not in allowed_edges and (v, u) not in allowed_edges:
                continue
            alt = d + 1
            if v in final and final[v] <= alt:
                continue
            while (v, alt) in forbidden:
                if (u, alt) in forbidden:
                    break
                if ((u, alt+1) in forbidden and (v, alt) in forbidden):
                    break
                alt += 1
            else:
                if u in final and final[u] <= alt-1:
                    continue
                if v in final and final[v] <= alt:
                    continue
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))
    else:
        raise nx.NetworkXNoPath

    path = []
    current = tgt
    cur_dist = dist[current]
    while current != src:
        path.append((current, cur_dist))
        prev_dist = cur_dist
        current = prev[current]
        cur_dist = dist[current]
        while prev_dist > cur_dist+1:
            prev_dist -= 1
            path.append((current, prev_dist))
    path.append((src, 0))
    path.reverse()
    return path

class DD_VARIANT(Enum):
    VANILLA = 1
    TWOPHASE = 2
    OPDD = 4
    PROBDD = 5
    CDD = 6
    ONEMIN = 7

class dd_mapf(AnytimeAlgorithm):
    def __init__(self, graph, queries,
                 timeout=None, variant=DD_VARIANT.VANILLA,
                 max_depth=2, repeat=1):
        super().__init__(timeout)
        self.max_depth = max_depth
        self.variant = variant
        self.max_repeat = repeat
        self.graph = graph
        self.queries = queries
        self.span = float('inf')
        self.kpaths = None

    def check(self, allowed_edges):
        allowed_edges = set(allowed_edges)
        queries = list()
        # print(len(self.queries))
        for src, tgt in self.queries:
            try:
                path = dijkstra_path(self.graph, src, tgt, allowed_edges)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return False, None, None
            pri = path[-1][1]
            queries.append((pri, src, tgt))
        queries.sort(reverse=True)
        kpaths = list()
        forbidden = set()
        final = dict()
        span = 0
        for _, src, tgt in queries:
            try:
                path = dijkstra_path(self.graph, src, tgt, allowed_edges, forbidden, final)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return False, None, None
            span += path[-1][1]
            final[path[-1][0]] = path[-1][1]
            kpaths.append(path)
            forbidden.update(path)
        return True, kpaths, span

    def onemin(self, obstacles, removed=list()):
        cur_obstacles = obstacles + removed
        t, kpaths, cur_span = self.check(cur_obstacles)
        for i, obstacle in tqdm(list(reversed(list(enumerate(obstacles))))):
            cbar = list(cur_obstacles)
            del cbar[i]
            t, kpaths, span = self.check(cbar)
            # print("ITER:", min(self.span, cur_span), t, span <= cur_span if t else False)
            if t and span <= cur_span:
                cur_obstacles = cbar
                cur_span = span
                if span < self.span:
                    self.result = list(cur_obstacles)
                    self.kpaths = kpaths
                    self.span = span
                    print(self.span)

        return cur_obstacles[:len(cur_obstacles)-len(removed)]

    # One Pass Delta Debugging
    def opdd(self, obstacles, removed=list(), n=2):
        cur_obstacles = list(obstacles)
        cur_result = cur_obstacles + removed
        t, kpaths, cur_span = self.check(cur_result)
        if self.result is None:
            self.result = cur_result

        while 1:
            if n >= len(cur_obstacles):
                return self.onemin(cur_obstacles, removed=removed)

            k, m = divmod(len(cur_obstacles), n)
            # print("cur_n: ", n)

            next_n = n

            for i in tqdm(range(n-1, -1, -1)):
                cbar = cur_obstacles[:i*k+min(i, m)] + cur_obstacles[(i+1)*k+min(i+1, m):]
                t, kpaths, span = self.check(cbar)

                # print("ITER:", min(self.span, cur_span), t, span <= cur_span if t else False)
                if t and span <= cur_span:
                    cur_obstacles = cbar
                    cur_span = span
                    if span < self.span:
                        self.result = cur_obstacles + removed
                        self.kpaths = kpaths
                        self.span = span
                    next_n = next_n - 1


            n = min(len(cur_obstacles), next_n * 2)

    def _run(self):
        top_func = self.opdd

        obstacles = [e for e in self.graph.edges]
        t, kpaths, span = self.check(list(obstacles))
        if not t:
            self.kpaths = []
            return self.result

        self.result = set(self.graph.edges)
        self.span = float('inf')
        self.kpaths = kpaths
        for _ in range(self.max_repeat):
            _ = top_func(obstacles)
            random.shuffle(obstacles)
            if self.variant == DD_VARIANT.TWOPHASE:
                break

        return self.result

    def __str__(self):
        return f"{self.variant}_{self.max_repeat}_{self.max_depth}"


class bound_mapf(AnytimeAlgorithm):
    def __init__(self, graph, queries, timeout=None):
        super().__init__(timeout)
        self.graph = graph
        self.queries = queries
        self.span = float('inf')
        self.kpaths = []

    def _run(self):
        self.kpaths = []
        denom = 0
        for (src, dst) in self.queries:
            denom += nx.shortest_path_length(self.graph, src, dst)
            self.kpaths.append([(src, 0), (dst,1)])

        self.span = denom


def verify_mapf(kpaths, queries):
    assert len(kpaths) <= len(queries)
    for i, path in enumerate(kpaths):
        src = path[0][0]
        dst = path[-1][0]
        assert (src, dst) in queries
        last_node = path[0][0]
        last_span = -1
        new_path = list()
        for node, span in path:
            if span == last_span + 1:
                last_span = span
                last_node = node
                new_path.append((node, span))
            else:
                last_span += 1
                new_path.append((last_node, last_span))

    # Vertex Disjoint
    for cur_nodes in zip_longest(*kpaths):
        cur_nodes = [node[0] for node in cur_nodes if node is not None]
        if len(cur_nodes) != len(set(cur_nodes)):
            print()
            print(sorted(cur_nodes))
            print()
        assert len(cur_nodes) == len(set(cur_nodes)), f"{len(cur_nodes)}, {len(set(cur_nodes))}"

    first = True
    prev = list()
    for cur_nodes in zip_longest(*kpaths):
        if first:
            prev = cur_nodes
            first = False
        else:
            edges = [tuple(sorted([fst, snd])) for fst, snd in zip(prev, cur_nodes) if snd is not None]
            assert len(edges) == len(set(edges))


def print_kpaths(kpaths):
    print("[")
    for path in kpaths:
        last_i = -1
        start = True
        for u, i in path:
            if start:
                start = False
            else:
                print(" -> ", end="")
            if not isinstance(i, int):
                print()
                print("HI", i, u)
                print(path)
                print()
            slack = i - last_i - 1
            last_i = i
            print(slack*(10*" " + " -> "), end="")
            x, y = u
            print(f"({x:3}, {y:3})", end="")
        print()
    print("]")

if __name__ == '__main__' and False:
    file_path = "./scenarios/den312d-even-1.scen"
    # file_path = "./scenarios/random-32-32-10-even-10.scen"
    (height, width), graph, real_queries = load_scenario(file_path)
    timeout = 600
    print(len(real_queries))



    for i in range(1, len(real_queries)+1):
        queries = real_queries[:i]
        mapf_func = dd_mapf(graph, queries, timeout, DD_VARIANT.OPDD, repeat=10)
        denom = 0
        for query in queries:
            denom += nx.shortest_path_length(graph, query[0], query[1])
        _ = mapf_func()
        verify_mapf(mapf_func.kpaths, queries)
        print(f"{i}, {denom}, {mapf_func.span}")
        filename = os.path.basename(file_path)
        file_stem, _ = os.path.splitext(filename)
        sys.stdout.flush()
    print()

# Convergence plot
if __name__ == '__main__' and False:
    file_path = "./scenarios/random-32-32-10-even-10.scen"
    (height, width), graph, real_queries = load_scenario(file_path)
    timeout = None
    print(len(real_queries))

    queries = real_queries[:100]
    mapf_func = dd_mapf(graph, queries, timeout, DD_VARIANT.OPDD, repeat=1)
    denom = 0
    for query in queries:
        denom += nx.shortest_path_length(graph, query[0], query[1])
    _ = mapf_func()
    sys.exit(0)
    verify_mapf(mapf_func.kpaths, queries)
    print(f"100, {denom}, {mapf_func.span}")
    filename = os.path.basename(file_path)
    file_stem, _ = os.path.splitext(filename)
    sys.stdout.flush()
    print()

if __name__ == '__main__':
    OUT_DIR = Path(sys.argv[1])
    def evaluate(file_path):
        (height, width), graph, queries = load_scenario(file_path)
        timeout = 600

        queries = queries[:100]

        denom = 0
        for query in queries:
            denom += nx.shortest_path_length(graph, query[0], query[1])

        mapf_funcs = [
            bound_mapf(graph, queries, timeout),
            dd_mapf(graph, queries, timeout, DD_VARIANT.OPDD, repeat=1000000000),
        ]

        print(file_path)
        print(20*'-')
        for mapf_func in mapf_funcs:
            _ = mapf_func()
            verify_mapf(mapf_func.kpaths, queries)
            print(mapf_func, "span:", mapf_func.span, "SoD:", mapf_func.span/denom)
            filename = os.path.basename(file_path)
            file_stem, _ = os.path.splitext(filename)
            with open(OUT_DIR/f"{file_stem}.txt", 'a') as f:
                f.write(f"{mapf_func.span},")
            sys.stdout.flush()
        print()

    processes = list()
    for file in sorted(Path("./scenarios").iterdir()):
        if file.suffix != ".scen":
            continue
        print(file)

        if not any(file.name.startswith(x) for x in ['random-64-64-10']):
            continue
    # for file in ['new_instances/bier127_gen2_m3.txt']:
        evaluate(file)
        # p = Process(target=evaluate, args=(file, ))
        # processes.append(p)
        # p.start()

    for p in processes:
        p.join()
