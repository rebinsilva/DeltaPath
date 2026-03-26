# ΔPath: A General Framework for Solving Path Planning Problems

Implementation of the ΔPath algorithm from the paper *"ΔPath: A General Framework for Solving Path Planning Problems"* (Rebin Silva Valan Arasu and Rajiv Gupta, UC Riverside).

ΔPath adapts the Delta Debugging search paradigm from software testing to solve NP-hard path planning problems by iteratively reducing a candidate solution while preserving feasibility.

---

## Dependencies

```bash
pip install networkx numpy tqdm shapely
```

---

## Repository Structure

```
DeltaPath/
├── Anytime.py          # Base class for all solvers (timeout via SIGVTALRM)
├── MAPF.py             # Multi-Agent Path Finding solvers
├── MCR.py              # Minimum Constraint Removal solvers
├── TOP.py              # Team Orienteering Problem solvers
├── maps/               # Grid map files (.map, octile format)
├── scenarios/          # MAPF benchmark scenarios (1,683 .scen files)
├── new_instances/      # TOP benchmark instances (bier127, cmt, etc.)
└── Chao/               # TOP benchmark instances (Chao et al. dataset)
```

---

## Running the Experiments

Each script takes one argument: the output directory. Create it first:

```bash
mkdir -p out/
```

---

### MCR — Figures 3 & 4

```bash
python MCR.py out/
```

Runs on a 64×64 grid, sweeping obstacle counts from 10 to 200 (in steps of 10), for **both** obstacle models automatically:
- `independentObstacleModel` (Figure 3)
- `randomRectangularObstacleModel` (Figure 4)

Solvers compared:
- `greedy_mcr1` — greedy heuristic (Greedy-MC baseline)
- `dd_mcr(OPDD, repeat=1)` — ΔPath

Each run prints to stdout:
```
<solver_name>    soln: <size>    time: <seconds>
```

> **Note:** `exact_mcr` (Exact-IDS-MC baseline) is present in the code but currently commented out in the active solver list since it has a longer runtime. Uncomment it at [MCR.py:602](MCR.py#L602) to reproduce the full three-curve comparison in Figures 3 and 4. Exact MCR is expensive and may not terminate for larger obstacle counts.

---

### TOP — Figures 5 & 6

```bash
python TOP.py out/
```

Iterates over all 469 instances in `./new_instances/` and `./Chao/`. Solvers compared:
- `rand_top(repeat=10)` — randomized heuristic, best of 10 trials (ZE-2016 baseline)
- `wdd_rand_top(OPDD, repeat=3)` — ΔPath with score-weighted OPDD, best of 3 trials

Each solver is timed using `timeit` averaged over 5 runs. Results are written to `out/<instance_stem>.txt`:
```
<score>, <time>, <score>, <time>,
```

**To reproduce Figures 5 and 6** (normalized quality difference and time difference vs ZE-2016):

The `repeat=3` parameter matches the paper's "best-of-3" strategy for ΔPath. The normalized quality difference plotted in Figure 5 is:
```
(ΔPath solution - ZE-2016 solution) / Best Known Solution
```
The 469 instances are the union of `new_instances/` (387 small-scale) and `Chao/` (82 large-scale).

---

### MAPF — Table II & Figure 7

```bash
python MAPF.py out/
```

Processes scenario files from `./scenarios/` filtered to `random-64-64-10` maps, running 100 agents per scenario with a 600-second timeout. Solvers compared:
- `bound_mapf` — lower bound (sum of individual shortest path lengths)
- `dd_mapf(OPDD, repeat=1000000000, timeout=600)` — ΔPath running OPDD with repeated shuffles until timeout

Results are written to `out/<scenario_stem>.txt`:
```
<bound_span>, <deltapath_span>,
```
and printed to stdout as:
```
<solver>  span: <value>  SoD: <value>
```

**To reproduce Table II** (SoD on Random, Empty, Maze, Warehouse maps):

Edit the map filter at [MAPF.py:443](MAPF.py#L443) to include the required prefixes:

```python
if not any(file.name.startswith(x) for x in [
    'random-64-64-10', 'empty-48-48', 'maze-128-128-10', 'warehouse-10-20-10-2-2'
]):
    continue
```

The Sum of Delays (SoD) metric from the paper is:
```
SoD = (sum of all agents' path lengths) / (sum of optimal single-agent shortest path lengths)
```

**To reproduce Figure 7** (scalability on `den312d`, 0–70 agents):

Enable the disabled block at [MAPF.py:363](MAPF.py#L363) by changing `if __name__ == '__main__' and False:` to `True`. Set `file_path` to a `den312d-even-*.scen` file. This iterates agents from 1 to the full query count and prints `i, denom, span` per row.

---

## Convergence Plots — Figure 8

Figure 8 shows solution quality improving over iterations.

### MCR (Fig. 8a)

The active `__main__` block already iterates over increasing obstacle counts and prints solution sizes at each step — the convergence trace is visible in the stdout output.

To observe per-iteration improvements within a single run, set `PRINT = True` in [Anytime.py:3](Anytime.py#L3). This activates logging of `self.result` updates with elapsed CPU time.

### TOP (Fig. 8b)

Set `PRINT = True` in [Anytime.py:3](Anytime.py#L3). Each time `self.result` improves during `_run()`, the new solution size and elapsed time are printed.

### MAPF (Fig. 8c)

Enable the convergence block at [MAPF.py:387](MAPF.py#L387) (`if __name__ == '__main__' and False:` → `True`). This runs OPDD with `repeat=1` on 100 agents from `random-32-32-10` and prints the final span. The `self.span` field in `dd_mapf` is updated each time a better solution is found.

---

## Table III — Summary of Iteration Counts

Table III reports iteration counts (improving, inferior, infeasible) across all three problems. This data requires adding counters inside the `opdd`/`vanilla` method loops in each file, tracking the outcome of each call to the resolve/eval functions.