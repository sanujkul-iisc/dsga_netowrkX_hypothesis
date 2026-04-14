
# Property-Based Testing for NetworkX - Project Plan

**Team Size:** 1  
**Course:** E0 251o  
**Subject:** Data Structures and Graph Analytics  
**SR. No.:** 13-19-01-19-52-24-1-24507  
**Name:** Sanuj Kulshrestha  
**E-mail:** sanujk@iisc.ac.in  
 
---

## Installation Commands

```bash
# Install required packages
pip install networkx hypothesis pytest

# Or with specific versions
pip install networkx>=3.0 hypothesis>=6.0 pytest>=7.0

# Run with Hypothesis statistics
pytest test_networkx_properties.py -v --hypothesis-show-statistics

# Run all tests without Hypothesis statistics
pytest test_networkx_properties.py -v

# Run with verbose output
pytest test_networkx_properties.py -v -s


```

---

# Algorithm Selection

## Selected Algorithms 

### 1. Minimum Spanning Tree (MST)
**NetworkX Function:** `nx.minimum_spanning_tree()`  
**Number of Tests:** 2

---

### 2. Betweenness Centrality
**NetworkX Function:** `nx.betweenness_centrality()`
**Number of Tests:** 2

---

### 3. Maximum Matching
**NetworkX Function:** `nx.max_weight_matching()`
**Number of Tests:** 1

---

# Test Specifications

## Test 1: MST Edge Count Property (Postcondition)

**Property Statement:**
For any connected graph G with n nodes, any minimum spanning tree T has exactly n-1 edges.

**Mathematical Basis:**
- A tree is an acyclic connected graph
- By fundamental graph theory theorem: any tree with n vertices has exactly n-1 edges (proven by induction)
- A spanning tree includes all vertices by definition
- Therefore: |E(MST)| = |V(G)| - 1
- This is a necessary condition for MST correctness

**Test Strategy:**
- Generating a random connected graphs (3-50 nodes)
- Varying following graph properties:
  - Size: small (3-10 nodes) to medium (20-50 nodes)
  - Density: sparse to dense
  - Weighted and unweighted variants
- Computing MST using NetworkX
- Verifying if: `len(MST.edges()) == len(G.nodes()) - 1`

**Graph Generation Strategy:**
- Using custom `connected_graphs()` Hypothesis strategy
- Building spanning tree first (guarantees connectivity)
- Add random additional edges for variety

**Failure Significance:**
- **Too few edges (m < n-1):** Not spanning - algorithm missed vertices
  - Could indicate premature termination or disconnected component handling bug
- **Too many edges (m > n-1):** Contains cycle - not a tree
  - Could indicate flawed cycle detection or union-find bug (Kruskal's algorithm)
- Either case indicates fundamental algorithm failure

**Property Type:** Postcondition (structural correctness)

---

## Test 2: MST Weight Minimality (Invariant)

**Property Statement:**
The total weight of an MST is less than or equal to the total weight of any other spanning tree of the same graph.

**Mathematical Basis:**
- By definition, MST minimizes total edge weight among all spanning trees
- For any spanning tree T' where T' ≠ MST: weight(MST) ≤ weight(T')
- This is the defining characteristic of MST algorithms
- Greedy algorithms (Kruskal's, Prim's) maintain this property through optimal substructure

**Test Strategy:**
- Generating random weighted connected graphs
- Edge weights: positive floats (0.1 to 100.0)
- Computing MST using NetworkX
- Generating alternative spanning trees by:
  - Random DFS/BFS traversal selecting n-1 edges
  - Repeat multiple times (5-10 alternatives per graph)
- Verifying if: `sum(MST edge weights) ≤ sum(alternative tree weights)` for all alternatives

**Graph Generation Strategy:**
- Use custom `weighted_connected_graphs()` strategy
- Ensure positive weights (avoid NaN, infinity, negative)
- Test various weight distributions (uniform, clustered)

**Failure Significance:**
- If any alternative spanning tree has strictly lower weight, MST algorithm failed its core objective
- Could indicate:
  - Incorrect edge weight comparison
  - Flawed greedy choice selection
  - Bugs in priority queue implementation (Prim's)
  - Bugs in edge sorting (Kruskal's)

**Property Type:** Invariant (optimality condition)

---

## Test 3: Betweenness Centrality Bounds (Invariant)

**Property Statement:**
For normalized betweenness centrality computation, all values must lie in the interval [0, 1].

**Mathematical Basis:**
- Betweenness centrality of node v: fraction of shortest paths passing through v
- Definition: BC(v) = Σ(σ_st(v) / σ_st) for all pairs s,t where s≠v≠t
  - σ_st = number of shortest paths from s to t
  - σ_st(v) = number of those paths passing through v
- Normalization divides by maximum possible value: (n-1)(n-2)/2 for undirected graphs
- Since σ_st(v) ≤ σ_st by definition, fraction ∈ [0,1]
- No node can have negative betweenness or >100% of paths

**Test Strategy:**
- Generating diverse graphs:
  - Directed and undirected
  - Connected and disconnected
  - Various sizes (3-30 nodes)
  - Various densities
- Computing normalized betweenness centrality
- Verifying if: `0 ≤ centrality[v] ≤ 1` for all nodes v
- Using floating-point tolerance (≈1e-9) for boundary cases

**Graph Generation Strategy:**
- Use both `connected_graphs()` and general graphs
- Test with directed=True/False parameter
- Include edge cases: star graphs, paths, complete graphs

**Failure Significance:**
- **Values < 0:** Arithmetic error or logical flaw in path counting
  - Incorrect subtraction in Brandes' algorithm
- **Values > 1:** Normalization failure or double-counting paths
  - Wrong normalization factor calculation
  - Path counting overflow or duplication
- Either indicates serious algorithmic error in centrality computation

**Property Type:** Invariant (mathematical bounds)

---

## Test 4: Betweenness Centrality Leaf Nodes Zero (Invariant)

**Property Statement:**
Any node with degree 1 (leaf node) must have betweenness centrality exactly zero. Leaf nodes cannot lie on shortest paths between other node pairs.

**Mathematical Basis:**
- A leaf node v has degree 1, meaning exactly one neighbor w
- By definition of simple paths: no vertex is repeated in a simple path
- Any path through v must enter via edge (w,v) and exit via edge (v,w)
- But this requires traversing edge (w,v) twice, violating the simple path definition
- Therefore, no simple shortest path can pass through v (except as endpoint)
- For any pair (s,t) where s ≠ v ≠ t: σ_st(v) = 0
- Thus: BC(v) = Σ(σ_st(v) / σ_st) = 0 for all leaf nodes
- This holds regardless of connectivity, weights, or normalization

**Test Strategy:**
- Generating diverse graphs using `arbitrary_graphs(min_nodes=3, max_nodes=30)`
- Both connected and disconnected graphs allowed
- Various topologies (paths, stars, trees, cycles, random)
- For each graph G:
  - Computing normalized betweenness centrality
  - Identifying all leaf nodes (degree = 1)
  - Verifying if: centrality[leaf] < 1e-9 for each leaf
- Run 100 random test cases

**Graph Generation Strategy:**
- Use `arbitrary_graphs()` (allows disconnected graphs with leaves)
- Tests diverse graph structures including trees
- Ensures coverage of leaf-heavy and leaf-sparse graphs

**Failure Significance:**
- Non-zero leaf centrality indicates:
  - Incorrect path counting in Brandes' algorithm
  - Bug in backward accumulation pass
  - Failure to respect simple path definition
  - Systematic error in centrality computation
- This is a precise mathematical guarantee, not approximation
- Detects fundamental algorithmic failures

**Property Type:** Invariant (mathematical correctness)

---

## Test 5: Maximum Matching Maximality (Postcondition)

**Property Statement:**
A maximum matching M in graph G has the property that no edge in G \ M can be added to M while maintaining the matching property (no two edges share an endpoint).

**Mathematical Basis:**
- A matching is a set of edges with no common vertices
- Maximal matching cannot be extended by adding any single edge
- For any edge (u,v) ∉ M, at least one of u or v must already be matched in M
- This is necessary condition for maximality
- Maximum matching is maximal (but not all maximal matchings are maximum)

**Test Strategy:**
- Generating random graphs (both bipartite and general)
- Graph sizes: 4-30 nodes
- Computing maximum matching M using NetworkX
- For every edge e = (u,v) not in M:
  - Checking if u is matched in M (exists edge in M incident to u)
  - Checking if v is matched in M (exists edge in M incident to v)
  - Verifying if: u is matched OR v is matched (or both)
- If verification fails, matching is not maximal

**Graph Generation Strategy:**
- Test both bipartite and general graphs
- Bipartite: use `nx.bipartite.random_graph()`
- General: use custom strategy
- Various densities to test different matching sizes

**Failure Significance:**
- If unmatched edge exists with both endpoints unmatched:
  - Matching is not maximal (can be extended)
  - Algorithm terminated prematurely
  - Missed augmenting paths
  - Bug in maximum matching algorithm (likely blossom algorithm)
- This directly tests the definition of maximal matching

**Property Type:** Postcondition (correctness of output)

---

## Resources

### Documentation
- **Hypothesis:** https://hypothesis.readthedocs.io/
- **NetworkX:** https://networkx.org/documentation/stable/
- **NetworkX Algorithms:** https://networkx.org/documentation/stable/reference/algorithms/
- **Pytest:** https://docs.pytest.org/

---

### Specific Algorithm Documentation
- **MST:** https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.minimum_spanning_tree.html
- **Betweenness Centrality:** https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
- **Maximum Matching:** https://networkx.org/documentation/stable/reference/algorithms/matching.html
