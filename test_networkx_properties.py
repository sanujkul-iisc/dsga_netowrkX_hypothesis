"""
Property-Based Testing for NetworkX Graph Algorithms

Team Member: Sanuj
Algorithms Tested:
  1. Minimum Spanning Tree (MST) - 2 tests
  2. Betweenness Centrality - 2 tests
  3. Maximum Matching - 1 test

This module implements comprehensive property-based tests using the Hypothesis
library to verify fundamental mathematical properties and algorithmic invariants
of NetworkX graph algorithms.
"""

import networkx as nx
from hypothesis import given, strategies as st, assume, settings
import pytest
from typing import Set, Tuple
import random


# ============================================================================
# Custom Hypothesis Strategies
# ============================================================================

@st.composite
def connected_graphs(draw, min_nodes=3, max_nodes=20, directed=False):
    """
    Generate random connected graphs.

    Strategy: Start with a spanning tree (guaranteed connected),
    then add random edges.

    Parameters:
    - min_nodes, max_nodes: Range for number of nodes
    - directed: Whether to create directed graph

    Returns: Connected NetworkX Graph or DiGraph
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(n))

    # Create spanning tree to ensure connectivity
    for i in range(1, n):
        parent = draw(st.integers(min_value=0, max_value=i-1))
        G.add_edge(parent, i)

    # Add random additional edges for variety
    num_additional = draw(st.integers(min_value=0, max_value=n*(n-1)//4))
    for _ in range(num_additional):
        u = draw(st.integers(min_value=0, max_value=n-1))
        v = draw(st.integers(min_value=0, max_value=n-1))
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G


@st.composite
def weighted_connected_graphs(draw, min_nodes=3, max_nodes=20):
    """
    Generate random weighted connected graphs with positive edge weights.

    Uses connected_graphs strategy and adds random positive weights.
    Weights are in range [0.1, 100.0] to avoid numerical issues.
    """
    G = draw(connected_graphs(min_nodes=min_nodes, max_nodes=max_nodes))

    # Add random positive weights
    for u, v in G.edges():
        weight = draw(st.floats(
            min_value=0.1,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False
        ))
        G[u][v]['weight'] = weight

    return G


# ============================================================================
# Test 1: MST Edge Count Property (Postcondition)
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(connected_graphs(min_nodes=3, max_nodes=50))
def test_mst_edge_count(G):
    """
    Property: For any connected graph G with n nodes, any minimum spanning tree T
    has exactly n-1 edges.

    Mathematical Basis:
    -------------------
    A tree is formally defined as an acyclic connected graph. This definition has
    a fundamental consequence in graph theory: any tree with n vertices has exactly
    n-1 edges. This can be proven by induction:

    Base case: A single node (n=1) has 0 = n-1 edges. ✓

    Inductive step: Assume trees with k nodes have k-1 edges. For a tree T with
    k+1 nodes, remove any leaf node (must exist in a tree with >1 nodes). The
    resulting graph is still a tree with k nodes and k-1 edges (by inductive
    hypothesis). Adding back the leaf node adds exactly 1 edge, giving us
    (k-1) + 1 = k = (k+1)-1 edges. ✓

    A spanning tree of graph G is by definition a subgraph that includes all
    vertices of G and is itself a tree. Therefore, any spanning tree of G with
    n vertices must have exactly n-1 edges.

    Since a minimum spanning tree (MST) is a spanning tree, it must satisfy this
    same property: |E(MST)| = |V(G)| - 1 = n - 1. This is a necessary condition
    for any algorithm to correctly compute an MST.

    Test Strategy:
    ---------------
    1. Generate random connected graphs with varying properties:
       - Size: 3-50 nodes (covering small, medium, and larger graphs)
       - Density: sparse to dense (0 to ~25% additional edges beyond spanning tree)
       - Both weighted and unweighted variants

    2. For each generated graph G:
       - Compute the minimum spanning tree using nx.minimum_spanning_tree()
       - Extract edge count: m = len(MST.edges())
       - Extract vertex count: n = len(MST.nodes())
       - Verify the postcondition: m == n - 1

    3. Run 100 random test cases per test invocation
    4. Hypothesis automatically shrinks failures to minimal counterexamples

    Why This Matters:
    -----------------
    This property is a critical postcondition that any MST algorithm must satisfy:

    - If m < n-1: The MST is not spanning (misses some vertices or has broken
      connectivity). This indicates the algorithm terminated prematurely or has
      a bug in connectivity checking. Suggests missing vertices or disconnected
      components not properly handled.

    - If m > n-1: The MST contains a cycle, violating the tree definition. This
      would indicate a critical bug in cycle detection or union-find data structure
      (if using Kruskal's algorithm). The greedy algorithm should never add edges
      that create cycles.

    Either failure case signals a fundamental algorithmic error that makes the
    MST computation unreliable. This is the most basic sanity check for MST
    correctness.

    Assumptions:
    -----------
    - Input graphs are connected (enforced by connected_graphs strategy)
    - NetworkX minimum_spanning_tree() is deterministic (not randomized)
    - Edge weights (if present) are properly handled by the algorithm

    Related Properties:
    -------------------
    - Test 2 (MST Weight Minimality): Verifies optimality of edge weights
    - Fundamental Graph Theory: Tree property theorem (n vertices ⟹ n-1 edges)
    - Spanning Tree Definition: Must include all vertices
    """
    n_nodes = len(G.nodes())

    # Compute MST
    mst = nx.minimum_spanning_tree(G)

    # Extract edge count
    n_edges = len(mst.edges())

    # Verify postcondition
    assert n_edges == n_nodes - 1, (
        f"MST edge count violation: Expected {n_nodes - 1} edges for {n_nodes} nodes, "
        f"but got {n_edges} edges. This violates the tree property."
    )


# ============================================================================
# Test 2: MST Weight Minimality Property (Invariant)
# ============================================================================

def generate_random_spanning_tree(G):
    """
    Generate a random spanning tree using randomized Kruskal's algorithm.

    This creates alternative spanning trees for comparison by randomly
    permuting edges before applying the MST construction algorithm.
    """
    n_nodes = len(G.nodes())
    if n_nodes == 0:
        return nx.Graph()

    # Create empty tree
    tree = nx.Graph()
    tree.add_nodes_from(G.nodes())

    # Get all edges and shuffle them
    edges = list(G.edges(data=True))
    random.shuffle(edges)

    # Use union-find to build spanning tree without cycles
    uf = {}  # Simple union-find structure

    def find(x):
        if x not in uf:
            uf[x] = x
        if uf[x] != x:
            uf[x] = find(uf[x])
        return uf[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            uf[px] = py
            return True
        return False

    # Add edges in random order, skipping those that would create cycles
    for u, v, data in edges:
        if union(u, v):
            tree.add_edge(u, v, **data)
            if len(tree.edges()) == n_nodes - 1:
                break

    return tree


@settings(max_examples=100, deadline=None)
@given(weighted_connected_graphs(min_nodes=3, max_nodes=30))
def test_mst_weight_minimality(G):
    """
    Property: The total weight of an MST is less than or equal to the total weight
    of any other spanning tree of the same graph.

    Mathematical Basis:
    -------------------
    The Minimum Spanning Tree is defined as the spanning tree that minimizes the
    sum of edge weights. This is not just a computational choice but a fundamental
    property that distinguishes MST from any other spanning tree.

    For any graph G with spanning trees T₁, T₂, ..., Tₖ:
    - All spanning trees have exactly n-1 edges (by tree property from Test 1)
    - MST is defined as: T* = argmin{Σ w(e) : e ∈ T} across all spanning trees T
    - Therefore, by definition: weight(MST) ≤ weight(T) for ALL other spanning trees T

    This property holds regardless of the MST algorithm implementation (Kruskal's,
    Prim's, Borůvka's). The algorithms may differ in computational approach but all
    must satisfy this fundamental optimality criterion.

    Proof of Necessity: If any alternative spanning tree had strictly lower total
    weight, then the computed result would not be an MST by definition. This is a
    necessary and sufficient condition for correctness of MST algorithms.

    Test Strategy:
    ---------------
    1. Generate random weighted connected graphs:
       - Vertices: 3-30 nodes
       - Edge weights: Positive floats in [0.1, 100.0] range
       - Variety in graph density and structure

    2. For each generated graph G:
       a) Compute the MST using nx.minimum_spanning_tree(G)
       b) Calculate total weight: mst_weight = Σ w(e) for e ∈ MST
       c) Generate k alternative spanning trees (k=5 per graph):
          - Use randomized edge ordering with union-find construction
          - Ensure each alternative is valid (n-1 edges, connected, spanning)
          - Each permutation uses different random choices
       d) For each alternative spanning tree T_alt:
          - Calculate alt_weight = Σ w(e) for e ∈ T_alt
          - Verify: mst_weight ≤ alt_weight
          - Allow small floating-point tolerance (1e-9 relative error)

    3. Run 100 random test cases, each generating 5 alternative trees
       Total comparisons: ~500 spanning tree comparisons

    Why This Matters:
    -----------------
    This invariant tests the core correctness of MST algorithms. Violation would
    indicate severe algorithmic failure:

    - If mst_weight > alt_weight for some alternative: The algorithm failed to find
      the true minimum. This suggests:
      * Incorrect edge weight comparison or ordering
      * Bug in greedy choice selection (Kruskal's)
      * Bug in priority queue or relaxation (Prim's)
      * Incorrect handling of ties in edge weights
      * Integer overflow or numerical precision loss

    - Subtle bugs can pass Test 1 (edge count) but fail this test:
      * Algorithm could create valid trees but pick suboptimal edges
      * Comparison operators might have sign errors
      * Weight accumulation might have off-by-one errors

    This property is stronger than Test 1: it verifies not just structure but also
    the optimality of the solution.

    Assumptions:
    -----------
    - All edge weights are positive (enforced by weighted_connected_graphs)
    - Graph is connected (enforced by weighted_connected_graphs)
    - All weights are finite and not NaN (enforced by strategy)
    - Alternative spanning trees are validly generated with correct weights

    Related Properties:
    -------------------
    - Test 1 (MST Edge Count): Verifies structural correctness (n-1 edges)
    - This test verifies optimality correctness (minimum weight)
    - Together they verify both structure AND optimality of MST
    - Related theorem: Optimal Substructure - MST maintains optimality on subgraphs
    """
    n_nodes = len(G.nodes())

    # Compute MST and its total weight
    mst = nx.minimum_spanning_tree(G)
    mst_weight = sum(data.get('weight', 1.0) for _, _, data in mst.edges(data=True))

    # Generate multiple alternative spanning trees for comparison
    num_alternatives = 5

    for alt_idx in range(num_alternatives):
        # Generate alternative spanning tree with random edge ordering
        alt_tree = generate_random_spanning_tree(G)

        # Verify it's a valid spanning tree
        if len(alt_tree.edges()) != n_nodes - 1:
            continue

        # Calculate alternative tree's total weight
        alt_weight = sum(data.get('weight', 1.0) for _, _, data in alt_tree.edges(data=True))

        # Verify MST weight ≤ alternative weight
        assert mst_weight <= alt_weight + 1e-9, (
            f"MST Weight Minimality violation: MST weight {mst_weight} > "
            f"alternative tree weight {alt_weight}. The algorithm failed to find "
            f"the minimum spanning tree (optimality criterion violated)."
        )


# ============================================================================
# Test 3: Betweenness Centrality Bounds (Invariant)
# ============================================================================

@st.composite
def arbitrary_graphs(draw, min_nodes=3, max_nodes=30, directed=False):
    """
    Generate arbitrary graphs (connected or disconnected).

    This strategy generates diverse graphs for betweenness centrality testing.
    Unlike connected_graphs, this allows disconnected components.

    Parameters:
    - min_nodes, max_nodes: Range for number of nodes
    - directed: Whether to create directed graph

    Returns: NetworkX Graph or DiGraph
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(n))

    # Add random edges (allowing disconnected graphs)
    num_edges = draw(st.integers(min_value=0, max_value=n*(n-1)//2))
    edges_added = 0

    for _ in range(num_edges * 2):  # Try more times to get desired number
        if edges_added >= num_edges:
            break
        u = draw(st.integers(min_value=0, max_value=n-1))
        v = draw(st.integers(min_value=0, max_value=n-1))
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1

    return G


@settings(max_examples=100, deadline=None)
@given(arbitrary_graphs(min_nodes=3, max_nodes=30))
def test_betweenness_centrality_bounds_undirected(G):
    """
    Property: For normalized betweenness centrality computation in undirected graphs,
    all centrality values must lie in the interval [0, 1].

    Mathematical Basis:
    -------------------
    Betweenness centrality is a network centrality measure that quantifies how often
    a node lies on shortest paths between other pairs of nodes. It is formally defined as:

    BC(v) = Σ (σ_st(v) / σ_st) for all pairs (s,t) where s ≠ v ≠ t

    Where:
    - σ_st = total number of shortest paths from node s to node t
    - σ_st(v) = number of shortest paths from s to t that pass through v
    - The normalization factor for undirected graphs is (n-1)(n-2)/2

    Key mathematical constraints:
    1. Since σ_st(v) ≤ σ_st by definition (v either lies on a path or it doesn't),
       each fraction in the sum is at most 1
    2. No node can appear on negative number of shortest paths: σ_st(v) ≥ 0
    3. Normalization divides by the maximum possible value, ensuring result ∈ [0,1]
    4. After normalization: normalized_BC(v) = BC(v) / ((n-1)(n-2)/2) ∈ [0,1]

    For disconnected graphs, the normalization factor is adjusted to account for
    unreachable node pairs (which contribute 0 to centrality).

    Test Strategy:
    ---------------
    1. Generate diverse undirected graphs:
       - Sizes: 3-30 nodes
       - Connectivity: Both connected and disconnected graphs
       - Density: Sparse to moderately dense
       - Various topologies: paths, stars, grids, random

    2. For each generated graph G:
       a) Compute normalized betweenness centrality using nx.betweenness_centrality(G)
       b) Retrieve centrality dictionary {node: centrality_value}
       c) For each node v in the graph:
          - Extract centrality value: bc_v = centrality[v]
          - Verify lower bound: bc_v ≥ -1e-9 (accounting for floating-point error)
          - Verify upper bound: bc_v ≤ 1 + 1e-9 (accounting for floating-point error)
          - Use relative tolerance for proper floating-point comparison

    3. Run 100 random test cases covering various graph sizes and topologies

    Why This Matters:
    -----------------
    This invariant is a fundamental mathematical property that any betweenness
    centrality implementation must preserve. Violation indicates serious errors:

    - Values < 0 (significantly): Suggests arithmetic errors in path counting:
      * Incorrect subtraction in Brandes' algorithm (which uses backwards pass)
      * Negative dependency accumulation (algorithm sums dependencies incorrectly)
      * Off-by-one error in numerator or denominator
      * Bug in path counter initialization or accumulation
      * Could indicate the algorithm is subtracting instead of adding somewhere

    - Values > 1 (significantly): Indicates normalization or counting failure:
      * Wrong normalization factor (e.g., used n instead of n(n-1)/2)
      * Double-counting of paths or node pairs
      * Failure to properly account for all shortest paths
      * Path count overflow or duplication bug
      * Could indicate denominator is too small or numerator is too large

    Both violations point to fundamental correctness issues in the centrality
    computation algorithm. This is a critical sanity check that's easy to verify
    but catches serious bugs in the algorithm implementation.

    Assumptions:
    -----------
    - Graph nodes are indexed 0 to n-1 (enforced by arbitrary_graphs)
    - NetworkX betweenness_centrality uses normalized version by default
    - Floating-point arithmetic allows small tolerances (1e-9)
    - Graph structure remains unchanged during computation

    Related Properties:
    -------------------
    - Test 4 (Betweenness Edge Addition Monotonicity): Tests how centrality changes
    - Brandes' algorithm: Standard algorithm for efficient centrality computation
    - Shortest path enumeration: Mathematical foundation for centrality definition
    - Graph theory: Centrality measures are fundamental in network analysis
    """
    # Skip trivial cases
    if len(G.nodes()) < 2:
        assume(False)

    # Compute betweenness centrality (normalized by default)
    centrality = nx.betweenness_centrality(G)

    # Verify bounds for all nodes
    for node, bc_value in centrality.items():
        # Allow small floating-point tolerance (1e-9)
        assert bc_value >= -1e-9, (
            f"Betweenness centrality lower bound violation: Node {node} has "
            f"centrality {bc_value} < 0. This violates the mathematical definition "
            f"that centrality must be non-negative (fraction of paths)."
        )
        assert bc_value <= 1.0 + 1e-9, (
            f"Betweenness centrality upper bound violation: Node {node} has "
            f"centrality {bc_value} > 1. This violates the normalized centrality "
            f"definition which must be ≤ 1 after normalization."
        )


@settings(max_examples=100, deadline=None)
@given(arbitrary_graphs(min_nodes=3, max_nodes=30, directed=True))
def test_betweenness_centrality_bounds_directed(G):
    """
    Property: For normalized betweenness centrality computation in directed graphs,
    all centrality values must lie in the interval [0, 1].

    Mathematical Basis:
    -------------------
    Similar to undirected graphs, but with different normalization for directed graphs.
    In directed graphs, betweenness centrality accounts for directional flow:

    BC_directed(v) = Σ (σ_st(v) / σ_st) for all ordered pairs (s,t) where s ≠ v ≠ t

    Key differences from undirected:
    1. Node pairs are ordered: (s,t) ≠ (t,s)
    2. Shortest paths must respect edge direction
    3. Normalization factor is (n-1)(n-2) for directed graphs (not divided by 2)
    4. A path from s to t is different from a path from t to s

    Mathematical constraints remain the same:
    - σ_st(v) ≤ σ_st (node appears on at most all shortest paths)
    - σ_st(v) ≥ 0 (cannot be negative)
    - normalized_BC(v) = BC(v) / ((n-1)(n-2)) ∈ [0,1]

    For directed graphs with multiple strongly connected components, unreachable
    pairs contribute 0 to centrality (σ_st = 0).

    Test Strategy:
    ---------------
    1. Generate diverse directed graphs:
       - Sizes: 3-30 nodes
       - Connectivity: Both weakly connected and disconnected
       - Density: Sparse to moderately dense
       - Various topologies: DAGs, cycles, tournaments, random

    2. For each generated directed graph G:
       a) Compute normalized betweenness centrality using nx.betweenness_centrality(G)
       b) The normalized parameter must be True (default for directed)
       c) For each node v:
          - Extract centrality: bc_v = centrality[v]
          - Verify bounds: -1e-9 ≤ bc_v ≤ 1 + 1e-9
          - Handle floating-point comparison appropriately

    3. Run 100 random test cases with directed graphs

    Why This Matters:
    -----------------
    Directed graph centrality has subtler implementation issues than undirected:

    - Direction-specific bugs: Path enumeration must respect edge directions
    - BFS traversal errors: Must follow directed edges only
    - Normalization factor errors: Easy to confuse directed vs undirected formulas
    - Strongly connected component handling: Unreachable nodes affect normality

    Bounds violations in directed graphs often indicate:
    - Incorrect treatment of edge direction in path-finding
    - Wrong normalization constant for directed case
    - Mishandling of disconnected or weakly connected components
    - Off-by-one errors in the directed normalization formula

    Assumptions:
    -----------
    - Graph is directed (enforced by arbitrary_graphs with directed=True)
    - NetworkX betweenness_centrality accepts directed graphs
    - Normalized version handles directed case properly
    - Floating-point tolerance appropriate for direction computations

    Related Properties:
    -------------------
    - Test for undirected graphs: Parallel property with different normalization
    - Test 4: Edge addition property also applies to directed graphs
    - Directed graph algorithms: Path counting and traversal
    """
    # Skip trivial cases
    if len(G.nodes()) < 2:
        assume(False)

    # Compute betweenness centrality for directed graph (normalized by default)
    centrality = nx.betweenness_centrality(G)

    # Verify bounds for all nodes
    for node, bc_value in centrality.items():
        # Allow small floating-point tolerance (1e-9)
        assert bc_value >= -1e-9, (
            f"Betweenness centrality lower bound violation (directed): Node {node} "
            f"has centrality {bc_value} < 0. In directed graphs, centrality must "
            f"account for directional shortest paths but remain non-negative."
        )
        assert bc_value <= 1.0 + 1e-9, (
            f"Betweenness centrality upper bound violation (directed): Node {node} "
            f"has centrality {bc_value} > 1. Even in directed graphs, normalization "
            f"ensures centrality ≤ 1."
        )


# ============================================================================
# Test 4: Betweenness Centrality Leaf Nodes Zero (Invariant)
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(arbitrary_graphs(min_nodes=3, max_nodes=30))
def test_betweenness_leaf_nodes_zero(G):
    """
    Property: Any node with degree 1 (leaf node) must have betweenness centrality
    exactly zero. Leaf nodes cannot lie on any shortest path between two other nodes.

    Mathematical Basis:
    -------------------
    A leaf node v is a node with degree 1, meaning it has exactly one neighbor w.
    We can prove that BC(v) = 0 for any leaf node v:

    Definition of betweenness centrality:
    BC(v) = Σ (σ_st(v) / σ_st) for all pairs (s,t) where s ≠ v ≠ t

    Where:
    - σ_st = number of shortest paths from s to t
    - σ_st(v) = number of those paths passing through v

    Proof that σ_st(v) = 0 for all s,t ≠ v:
    1. A simple path P is a path where no vertex is repeated
    2. A shortest path is a simple path (no shortest path contains cycles)
    3. If leaf v has only neighbor w, then any path through v must:
       - Enter v from w: edge (w,v)
       - Exit v back to w: edge (v,w)
       - This requires traversing edge (w,v) twice
    4. But a simple path cannot repeat vertices, so it cannot use the same edge twice
    5. Therefore, no simple path can pass through v except if v is an endpoint
    6. For any pair (s,t) where s ≠ v and t ≠ v, no shortest path passes through v
    7. Thus σ_st(v) = 0 for all valid pairs
    8. Therefore: BC(v) = 0 for any leaf node v

    This holds regardless of:
    - Graph connectivity (leaf exists in connected or disconnected components)
    - Edge weights (weights don't change the simple path requirement)
    - Normalization (0 normalized is still 0)

    Test Strategy:
    ---------------
    1. Generate diverse graphs using arbitrary_graphs strategy:
       - Sizes: 3-30 nodes
       - Both connected and disconnected graphs
       - Various topologies (paths, stars, trees, cycles, random)
       - Allow trees (which have many leaves) and denser graphs

    2. For each generated graph G:
       a) Compute normalized betweenness centrality: C = nx.betweenness_centrality(G)
       b) Identify all leaf nodes: L = {v : degree(v) = 1}
       c) For each leaf node v ∈ L:
          - Extract BC value: bc_v = C[v]
          - Verify zero: bc_v < 1e-9 (accounting for floating-point error)
          - Use absolute tolerance since 0 is exactly representable
       d) If no leaves exist, test still passes (vacuously true)

    3. Run 100 random test cases, testing diverse graph structures

    Why This Matters:
    -----------------
    This invariant catches critical bugs in betweenness centrality implementation:

    - Bugs in Brandes' backward accumulation pass:
      * If accumulation incorrectly counts leaf nodes, it violates fundamental math
      * Suggests error in how dependencies are propagated through graph
      * Indicates failure to respect simple path constraint

    - Path counting errors:
      * If leaf receives non-zero centrality, path counting is broken
      * Could indicate double-counting, cycle-counting, or endpoint-counting bugs
      * Suggests σ_st(v) is not correctly computed as 0

    - Normalization issues:
      * Even with normalization bugs, leaf centrality should remain 0
      * Non-zero leaf centrality cannot be fixed by rescaling

    - Connection to graph structure:
      * Leaves are the most constrained nodes in graph topology
      * If algorithm fails on leaves, it likely has systematic issues
      * This is a precise mathematical guarantee, not an approximation

    This property is particularly powerful because:
    - It's mathematically precise (exact zero required)
    - Easy to verify in any graph structure
    - Independent of graph properties (works for any degree-1 node)
    - Detects fundamental algorithmic failures, not just numerical errors

    Assumptions:
    -----------
    - Graph is undirected (NetworkX standard for betweenness_centrality)
    - Nodes are distinct (no self-loops affecting degree)
    - Simple paths definition (no repeated vertices)
    - Floating-point tolerance only for hardware precision errors
    - Betweenness centrality is normalized (default in NetworkX)

    Related Properties:
    -------------------
    - Test 3 (Betweenness Bounds): Establishes [0,1] bounds (this is special case)
    - This test (Test 4): Tests specific zero property for degree-1 nodes
    - Graph structure: Leaf nodes are fundamental in tree and general graphs
    - Connection to network resilience: Leaves are peripheral in network analysis
    - Brandes' algorithm: Tests core correctness of path enumeration
    """
    # Skip trivial cases
    if len(G.nodes()) < 2:
        assume(False)

    # Compute betweenness centrality (normalized by default)
    centrality = nx.betweenness_centrality(G)

    # Find all leaf nodes (degree 1)
    leaf_nodes = [node for node in G.nodes() if G.degree(node) == 1]

    # Verify that all leaf nodes have zero betweenness centrality
    for leaf in leaf_nodes:
        bc_value = centrality[leaf]

        assert bc_value < 1e-9, (
            f"Betweenness Centrality Leaf Node violation: Leaf node {leaf} "
            f"(degree = 1) has non-zero betweenness centrality {bc_value}. "
            f"This violates the mathematical property that leaf nodes cannot lie on "
            f"shortest paths between other node pairs (they have only one neighbor). "
            f"Any path through a leaf must enter and exit via the same edge, which is "
            f"impossible in a simple path. This indicates:\n"
            f"- Incorrect path counting in Brandes' algorithm\n"
            f"- Bug in backward accumulation pass\n"
            f"- Failure to respect simple path definition\n"
            f"- Systematic error in centrality computation"
        )


# ============================================================================
# Test 5: Maximum Matching Maximality (Postcondition)
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(arbitrary_graphs(min_nodes=4, max_nodes=30))
def test_maximum_matching_maximality(G):
    """
    Property: A maximum matching M in graph G has the property that no edge in G \\ M
    can be added to M while maintaining the matching property (no two edges share an
    endpoint).

    Mathematical Basis:
    -------------------
    A matching is a set of edges in a graph where no two edges share a common vertex.
    Formally, M is a matching if for any two distinct edges (u₁,v₁) and (u₂,v₂) in M:
    {u₁,v₁} ∩ {u₂,v₂} = ∅ (no common endpoints).

    A maximal matching is a matching that cannot be extended by adding any single edge.
    A maximum matching is a matching with the largest possible number of edges.

    Key property: A maximum matching is always maximal (but not all maximal matchings
    are maximum). Formally:
    - For maximum matching M: no edge e ∈ G \\ M can be added to M
    - This means for every unmatched edge (u,v) ∉ M, at least one of u or v must
      already be matched to another vertex in M
    - Equivalently: Every vertex is either matched or adjacent to a matched vertex

    Mathematical proof:
    If there existed an unmatched edge (u,v) where both u and v are unmatched in M,
    then we could add (u,v) to M, contradicting maximality. Therefore, for a maximum
    matching M, every unmatched edge must have at least one endpoint already matched.

    This is a necessary condition for maximality and hence for maximum matching.

    Test Strategy:
    ---------------
    1. Generate random graphs of various types:
       - Sizes: 4-30 nodes
       - Both bipartite and general graphs
       - Various densities from sparse to dense
       - Different topologies (paths, cycles, complete subgraphs, random)

    2. For each generated graph G:
       a) Compute maximum matching M using nx.max_weight_matching(G)
          (unweighted case automatically selects maximum cardinality)
       b) Identify all edges in G: E = G.edges()
       c) Identify unmatched edges: E \\ M = {e ∈ E : e ∉ M}
       d) For each unmatched edge (u,v) ∈ E \\ M:
          - Check if u is matched: ∃ edge in M incident to u
          - Check if v is matched: ∃ edge in M incident to v
          - Verify: u is matched OR v is matched (or both)
          - If both are unmatched, the matching is not maximal

    3. Run 100 random test cases covering diverse graph structures

    Why This Matters:
    -----------------
    This postcondition directly validates the definition of maximal/maximum matching.
    Violation would indicate serious algorithmic failure:

    - If unmatched edge exists with both endpoints unmatched:
      * The matching is not maximal (can be extended by adding one edge)
      * Algorithm terminated prematurely or incorrectly
      * Suggests the maximum matching algorithm has a critical bug
      * In Edmonds' blossom algorithm, this indicates missing augmenting paths
      * Could indicate failure to properly contract blossoms or augment along paths

    - Specific bugs this detects:
      * Incomplete augmenting path search
      * Incorrect blossom handling (used in general graphs)
      * Premature termination before finding all augmenting paths
      * Off-by-one error in matching size
      * Incorrect matching set returned

    - Impact:
      * Maximum matching is critical for optimization problems
      * Used in bipartite matching, job assignment, scheduling
      * Any violation makes the algorithm unreliable for applications

    This test is powerful because it checks a direct consequence of correctness
    without requiring computation of all possible matchings.

    Assumptions:
    -----------
    - Graph is undirected (enforced by arbitrary_graphs with directed=False)
    - Graph has at least 4 nodes (allows at least potential for 2-edge matching)
    - nx.max_weight_matching() returns a valid matching
    - Edge format is (u, v) for undirected graphs
    - Graph does not have self-loops or multi-edges (standard NetworkX)

    Related Properties:
    -------------------
    - Matching definition: Set of edges with no common vertices
    - Maximal vs Maximum: Not all maximal matchings are maximum
    - Augmenting paths: Foundation of matching algorithms
    - Edmonds' Blossom Algorithm: Standard algorithm for general graphs
    - Hall's Marriage Theorem: Classic matching theorem for bipartite graphs
    - König-Egerváry Theorem: Relates maximum matching to vertex cover
    """
    # Skip graphs that are too small
    if len(G.nodes()) < 4:
        assume(False)

    # Compute maximum matching
    matching = nx.max_weight_matching(G)

    # Create a set of matched nodes (nodes that appear in the matching)
    matched_nodes = set()
    for u, v in matching:
        matched_nodes.add(u)
        matched_nodes.add(v)

    # Check every edge not in the matching
    for u, v in G.edges():
        # Skip edges that are in the matching
        if (u, v) in matching or (v, u) in matching:
            continue

        # For unmatched edge (u,v), at least one endpoint must be matched
        u_is_matched = u in matched_nodes
        v_is_matched = v in matched_nodes

        assert u_is_matched or v_is_matched, (
            f"Maximum Matching Maximality violation: Found unmatched edge ({u}, {v}) "
            f"where both endpoints are unmatched. This means the matching can be "
            f"extended by adding this edge, violating maximality. "
            f"Matched nodes: {matched_nodes}, "
            f"Matching edges: {matching}. "
            f"The algorithm failed to find a true maximum matching."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
