# Traveling Salesman Problem – Optimization Project

This project implements heuristic algorithms for solving the Traveling Salesman Problem (TSP)
under a strict 60-second execution limit. Since TSP is NP-hard, exact solvers become impractical
for large graphs. The goal of this work is to achieve high-quality approximate solutions while
respecting time constraints using fast deterministic heuristics.

---

## Problem Description

The Traveling Salesman Problem (TSP) seeks the shortest possible closed tour that visits each
of N cities exactly once and returns to the starting city. Each problem instance consists of exactly 1,000 nodes.

Due to the exponential complexity of exact TSP solvers, heuristic approaches are required to
produce competitive solutions within strict runtime constraints.

Two input variants were considered in this project:

**Problem A:**  
Nodes represent points on a two-dimensional plane. Edge costs are calculated as the Euclidean
distance between each pair of nodes.

**Problem B:**  
Edge costs are independently sampled from a uniform random distribution over the range
[0, 100], forming a fully connected weighted graph with no geometric structure.

---

## Algorithm Selection Rationale

### Nearest Neighbor (NN) – Initial Construction

The Nearest Neighbor heuristic was chosen to create the initial tour due to its simplicity,
speed, and reliability. NN constructs a feasible tour by repeatedly selecting the nearest
unvisited city, completing a tour in O(N²) time.

NN was selected because:
- It produces a valid tour almost instantly.
- It has minimal memory and computational overhead.
- It provides a consistent baseline solution suitable for local optimization.
- More complex construction methods yielded only marginal quality improvements relative to their
  increased runtime cost under the 60-second constraint.

Although NN does not guarantee near-optimal tours, it reliably establishes a strong starting point
for refinement.

---

### 2-Opt – Local Optimization

After the initial NN solution, 2-opt local search is applied. This optimization method removes
two edges at a time and reconnects the tour to eliminate crossings and reduce total tour length.
Each accepted 2-opt move strictly improves the solution until convergence to a local minimum.

2-opt was chosen because:
- It offers excellent solution improvement per computation time.
- It converges rapidly.
- It incurs low algorithmic complexity compared to higher-order heuristics.
- Its deterministic improvement path ensures stable performance under strict time limits.

---

### Excluded Methods

More advanced stochastic methods such as multi-start heuristics and simulated annealing
were evaluated but were not adopted. Under the fixed 60-second runtime constraint,
these methods did not consistently outperform the simpler NN + 2-opt approach due to
their longer exploration phases.

## Compilation

Build the program using:

g++ -O2 -std=c++17 src/normalTSP.cpp src/TSPLoadData.cpp -o tsp

Run
./tsp inputfile


## Input file Format

N
node1 node2 distance
1     2     26.5
...

