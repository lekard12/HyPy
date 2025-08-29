from collections import deque, defaultdict
from typing import Dict, List, Set, Hashable, Optional, Tuple
import heapq

def bfs(
    graph: Dict[Hashable, List[Hashable]],
    start: Hashable,
    sort: bool = False
) -> List[Hashable]:
    # TODO: Docstring
    visited: Set[Hashable] = set()
    queue: deque = deque([start])
    result: List[Hashable] = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            neighbors = sorted(graph[node]) if sort else graph[node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    return result

def dfs(
    graph: Dict[Hashable, List[Hashable]],
    start: Hashable,
    sort: bool = False
) -> List[Hashable]:
    # TODO: Docstring
    visited: Set[Hashable] = set()
    stack: List[Hashable] = [start]
    result: List[Hashable] = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            neighbors = sorted(graph[node], reverse=True) if sort else reversed(graph[node])
            stack.extend(neighbors)

    return result

def unweighted_shortest_path(
    graph: Dict[Hashable, List[Hashable]],
    start: Hashable,
    goal: Hashable
) -> List[Hashable]:
    # TODO: Docstring
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return []

def floyd_warshall(
    graph: Dict[Hashable, Dict[Hashable, float]]
) -> Dict[Hashable, Dict[Hashable, float]]:
    # TODO: Docstring
    nodes = list(graph.keys())
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}

    for u in graph:
        dist[u][u] = 0
        for v, w in graph[u].items():
            dist[u][v] = w

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist

def dijkstra(
    graph: Dict[Hashable, List[Tuple[Hashable, float]]],
    start: Hashable,
    goal: Optional[Hashable] = None
) -> Tuple[Dict[Hashable, float], Dict[Hashable, List[Hashable]]]:
    # TODO: Docstring
    distances = {node: float('inf') for node in graph}
    paths = {node: [] for node in graph}
    distances[start] = 0
    paths[start] = [start]
    pq = [(0, start)]

    while pq:
        current_dist, node = heapq.heappop(pq)
        if current_dist > distances[node]:
            continue
        if goal is not None and node == goal:
            break
        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[node] + [neighbor]
                heapq.heappush(pq, (distance, neighbor))

    return distances, (paths if goal is None else {goal: paths[goal]})

def bellman_ford(
    graph: Dict[Hashable, List[Tuple[Hashable, float]]],
    start: Hashable,
    goal: Optional[Hashable] = None
) -> Tuple[Dict[Hashable, float], Dict[Hashable, List[Hashable]]]:
    # TODO: Docstring
    distances = {node: float('inf') for node in graph}
    predecessors: Dict[Hashable, Optional[Hashable]] = {node: None for node in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    predecessors[neighbor] = node

    for node in graph:
        for neighbor, weight in graph[node]:
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Graph contains a negative-weight cycle")

    paths: Dict[Hashable, List[Hashable]] = {n: [] for n in graph}
    for node in graph:
        if distances[node] != float('inf'):
            path = []
            current = node
            while current is not None:
                path.append(current)
                current = predecessors[current]
            paths[node] = list(reversed(path))

    return distances, (paths if goal is None else {goal: paths[goal]})

def kruskal(
    graph: Dict[Hashable, List[Tuple[Hashable, float]]]
) -> List[Tuple[Hashable, Hashable, float]]:
    # TODO: Docstring
    parent: Dict[Hashable, Hashable] = {}
    rank: Dict[Hashable, int] = {}

    def find(u: Hashable) -> Hashable:
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u: Hashable, v: Hashable) -> bool:
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            return False
        if rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_u] = root_v
            if rank[root_u] == rank[root_v]:
                rank[root_v] += 1
        return True

    edges: List[Tuple[Hashable, Hashable, float]] = []
    for u in graph:
        parent[u] = u
        rank[u] = 0
        for v, w in graph[u]:
            if (v, u, w) not in edges:
                edges.append((u, v, w))
    edges.sort(key=lambda x: x[2])

    mst: List[Tuple[Hashable, Hashable, float]] = []
    for u, v, w in edges:
        if union(u, v):
            mst.append((u, v, w))
    return mst

def prim(
    graph: Dict[Hashable, List[Tuple[Hashable, float]]],
    start: Hashable
) -> List[Tuple[Hashable, Hashable, float]]:
    # TODO: Docstring
    visited: Set[Hashable] = set([start])
    edges: List[Tuple[float, Hashable, Hashable]] = [(w, start, v) for v, w in graph[start]]
    heapq.heapify(edges)
    mst: List[Tuple[Hashable, Hashable, float]] = []

    while edges:
        w, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, w))
            for to, weight in graph[v]:
                if to not in visited:
                    heapq.heappush(edges, (weight, v, to))
    return mst

def topological_sort(
    graph: Dict[Hashable, List[Hashable]]
) -> List[Hashable]:
    # TODO: Docstring
    in_degree: Dict[Hashable, int] = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue: deque = deque([u for u in graph if in_degree[u] == 0])
    order: List[Hashable] = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(graph):
        raise ValueError("Graph contains a cycle")
    return order

def detect_cycle(
    graph: Dict[Hashable, List[Hashable]]
) -> bool:
    # TODO: Docstring
    visited: Set[Hashable] = set()
    rec_stack: Set[Hashable] = set()

    def dfs_cycle(node: Hashable) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs_cycle(node):
                return True
    return False

def kosaraju(
    graph: Dict[Hashable, List[Hashable]]
) -> List[List[Hashable]]:
    # TODO: Docstring
    visited: Set[Hashable] = set()
    finish_order: List[Hashable] = []

    def dfs1(node: Hashable) -> None:
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(node)

    for node in graph:
        if node not in visited:
            dfs1(node)

    reversed_graph: Dict[Hashable, List[Hashable]] = {u: [] for u in graph}
    for u in graph:
        for v in graph[u]:
            reversed_graph[v].append(u)

    visited.clear()
    sccs: List[List[Hashable]] = []

    def dfs2(node: Hashable, component: List[Hashable]) -> None:
        visited.add(node)
        component.append(node)
        for neighbor in reversed_graph[node]:
            if neighbor not in visited:
                dfs2(neighbor, component)

    for node in reversed(finish_order):
        if node not in visited:
            component: List[Hashable] = []
            dfs2(node, component)
            sccs.append(component)

    return sccs

def a_star(
    graph: Dict[Hashable, List[Tuple[Hashable, float]]],
    start: Hashable,
    goal: Hashable,
    heuristic: Dict[Hashable, float]
) -> List[Hashable]:
    # TODO: Docstring
    open_set: List[Tuple[float, Hashable]] = [(heuristic[start], start)]
    came_from: Dict[Hashable, Optional[Hashable]] = {start: None}
    g_score: Dict[Hashable, float] = {node: float('inf') for node in graph}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path: List[Hashable] = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))

        for neighbor, weight in graph[current]:
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic.get(neighbor, float('inf'))
                heapq.heappush(open_set, (f_score, neighbor))

    return []

def tarjan(
    graph: Dict[Hashable, List[Hashable]]
) -> List[List[Hashable]]:
    # TODO: Docstring
    index: Dict[Hashable, int] = {}
    lowlink: Dict[Hashable, int] = {}
    stack: List[Hashable] = []
    on_stack: Set[Hashable] = set()
    sccs: List[List[Hashable]] = []
    current_index = 0

    def dfs(node: Hashable) -> None:
        nonlocal current_index
        index[node] = current_index
        lowlink[node] = current_index
        current_index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in index:
                dfs(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], index[neighbor])

        if lowlink[node] == index[node]:
            component: List[Hashable] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == node:
                    break
            sccs.append(component)

    for node in graph:
        if node not in index:
            dfs(node)

    return sccs

def johnson_all_pairs(
    graph: Dict[Hashable, List[Tuple[Hashable, float]]]
) -> Dict[Hashable, Dict[Hashable, float]]:
    # TODO: Docstring
    new_graph: Dict[Hashable, List[Tuple[Hashable, float]]] = {u: edges[:] for u, edges in graph.items()}
    q_node = "__Q__"
    new_graph[q_node] = [(u, 0) for u in graph]

    distances, _ = bellman_ford(new_graph, q_node)
    h = distances

    reweighted_graph: Dict[Hashable, List[Tuple[Hashable, float]]] = {}
    for u in graph:
        reweighted_graph[u] = [(v, w + h[u] - h[v]) for v, w in graph[u]]

    all_pairs: Dict[Hashable, Dict[Hashable, float]] = {}
    for u in graph:
        dist, _ = dijkstra(reweighted_graph, u)
        all_pairs[u] = {v: dist[v] - h[u] + h[v] for v in dist}

    return all_pairs

def edmonds_karp(
    graph: Dict[Hashable, Dict[Hashable, float]],
    source: Hashable,
    sink: Hashable
) -> float:
    # TODO: Docstring
    flow: Dict[Hashable, Dict[Hashable, float]] = {u: {v: 0 for v in graph[u]} for u in graph}
    max_flow = 0

    def bfs() -> Optional[List[Hashable]]:
        parent: Dict[Hashable, Optional[Hashable]] = {u: None for u in graph}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if parent[v] is None and v != source and graph[u][v] - flow[u][v] > 0:
                    parent[v] = u
                    if v == sink:
                        path: List[Hashable] = []
                        cur = sink
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        return list(reversed(path))
                    queue.append(v)
        return None

    path = bfs()
    while path:
        residual = min(graph[path[i]][path[i+1]] - flow[path[i]][path[i+1]] for i in range(len(path)-1))
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            flow[u][v] += residual
            flow[v][u] = flow.get(v, {}).get(u, 0) - residual
        max_flow += residual
        path = bfs()

    return max_flow

def find_articulation_points(
    graph: Dict[Hashable, List[Hashable]]
) -> Set[Hashable]:
    # TODO: Docstring
    visited: Set[Hashable] = set()
    disc: Dict[Hashable, int] = {}
    low: Dict[Hashable, int] = {}
    parent: Dict[Hashable, Optional[Hashable]] = {u: None for u in graph}
    aps: Set[Hashable] = set()
    time = 0

    def dfs(u: Hashable) -> None:
        nonlocal time
        visited.add(u)
        disc[u] = low[u] = time
        time += 1
        children = 0
        for v in graph[u]:
            if v not in visited:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] is None and children > 1:
                    aps.add(u)
                if parent[u] is not None and low[v] >= disc[u]:
                    aps.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for u in graph:
        if u not in visited:
            dfs(u)

    return aps

def find_bridges(
    graph: Dict[Hashable, List[Hashable]]
) -> List[Tuple[Hashable, Hashable]]:
    # TODO: Docstring
    visited: Set[Hashable] = set()
    disc: Dict[Hashable, int] = {}
    low: Dict[Hashable, int] = {}
    parent: Dict[Hashable, Optional[Hashable]] = {u: None for u in graph}
    bridges: List[Tuple[Hashable, Hashable]] = []
    time = 0

    def dfs(u: Hashable) -> None:
        nonlocal time
        visited.add(u)
        disc[u] = low[u] = time
        time += 1
        for v in graph[u]:
            if v not in visited:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for u in graph:
        if u not in visited:
            dfs(u)

    return bridges

def find_eulerian_path(
    graph: Dict[Hashable, List[Hashable]]
) -> Optional[List[Hashable]]:
    # TODO: Docstring
    in_deg: Dict[Hashable, int] = defaultdict(int)
    out_deg: Dict[Hashable, int] = defaultdict(int)
    for u in graph:
        out_deg[u] = len(graph[u])
        for v in graph[u]:
            in_deg[v] += 1

    start = None
    end = None
    for node in set(list(graph.keys()) + list(in_deg.keys())):
        out_d = out_deg.get(node, 0)
        in_d = in_deg.get(node, 0)
        if out_d - in_d == 1:
            if start is not None:
                return None
            start = node
        elif in_d - out_d == 1:
            if end is not None:
                return None
            end = node
        elif in_d != out_d:
            return None
    if start is None:
        start = next(iter(graph))

    graph_copy: Dict[Hashable, List[Hashable]] = {u: graph[u][:] for u in graph}
    path: List[Hashable] = []
    stack: List[Hashable] = [start]

    while stack:
        u = stack[-1]
        if graph_copy.get(u):
            v = graph_copy[u].pop()
            stack.append(v)
        else:
            path.append(stack.pop())

    return path[::-1] if len(path) == sum(len(edges) for edges in graph.values()) + 1 else None
