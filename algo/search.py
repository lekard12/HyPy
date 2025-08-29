from collections import deque
from heapq import heappop, heappush
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import math
import random

def reconstruct_path(
        came_from: Dict[Any, Optional[Any]], 
        end: Any
) -> List[Any]:
    # TODO: Docstring
    path: List[Any] = []
    current = end
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return list(reversed(path))

def bfs_search(
        start: Any, 
        goal_test: Callable[[Any], bool], 
        successors: Callable[[Any], List[Any]]
) -> List[Any]:
    # TODO: Docstring
    visited: Set[Any] = set()
    queue: deque = deque([(start, [start])])

    while queue:
        state, path = queue.popleft()
        if goal_test(state):
            return path
        if state not in visited:
            visited.add(state)
            for next_state in successors(state):
                if next_state not in visited:
                    queue.append((next_state, path + [next_state]))
    return []

def dfs_search(
        start: Any, 
        goal_test: Callable[[Any], bool], 
        successors: Callable[[Any], List[Any]]
) -> List[Any]:
    visited: Set[Any] = set()
    stack: List[Tuple[Any, List[Any]]] = [(start, [start])]
    # TODO: Docstring
    while stack:
        state, path = stack.pop()
        if goal_test(state):
            return path
        if state not in visited:
            visited.add(state)
            for next_state in reversed(successors(state)):
                if next_state not in visited:
                    stack.append((next_state, path + [next_state]))
    return []

def iddfs_search(
        start: Any, 
        goal_test: Callable[[Any], bool], 
        successors: Callable[[Any], List[Any]], 
        max_depth: int = 50
) -> List[Any]:
    # TODO: Docstring
    def dfs_depth(state: Any, path: List[Any], depth: int, visited: Set[Any]) -> Optional[List[Any]]:
        if goal_test(state):
            return path
        if depth <= 0:
            return None
        visited.add(state)
        for next_state in successors(state):
            if next_state not in visited:
                result = dfs_depth(next_state, path + [next_state], depth-1, visited)
                if result:
                    return result
        visited.remove(state)
        return None

    for depth in range(max_depth):
        visited: Set[Any] = set()
        result = dfs_depth(start, [start], depth, visited)
        if result:
            return result
    return []

def bidirectional_search(
    start: Any,
    goal: Any,
    successors: Callable[[Any], List[Any]]
) -> Optional[List[Any]]:
    # TODO: Docstring
    if start == goal:
        return [start]

    visited_start = {start: [start]}
    visited_goal = {goal: [goal]}
    queue_start = deque([start])
    queue_goal = deque([goal])

    while queue_start and queue_goal:
        for _ in range(len(queue_start)):
            s = queue_start.popleft()
            for neighbor in successors(s):
                if neighbor not in visited_start:
                    visited_start[neighbor] = visited_start[s] + [neighbor]
                    queue_start.append(neighbor)
                    if neighbor in visited_goal:
                        return visited_start[neighbor] + visited_goal[neighbor][-2::-1]

        for _ in range(len(queue_goal)):
            s = queue_goal.popleft()
            for neighbor in successors(s):
                if neighbor not in visited_goal:
                    visited_goal[neighbor] = visited_goal[s] + [neighbor]
                    queue_goal.append(neighbor)
                    if neighbor in visited_start:
                        return visited_start[neighbor] + visited_goal[neighbor][-2::-1]

    return None

def uniform_cost_search(
    start: Any,
    goal_test: Callable[[Any], bool],
    successors: Callable[[Any], List[Tuple[Any, float]]]
) -> List[Any]:
    # TODO: Docstring
    visited: Set[Any] = set()
    pq: List[Tuple[float, Any, List[Any]]] = [(0, start, [start])]

    while pq:
        cost, state, path = heappop(pq)
        if goal_test(state):
            return path
        if state not in visited:
            visited.add(state)
            for next_state, step_cost in successors(state):
                if next_state not in visited:
                    heappush(pq, (cost + step_cost, next_state, path + [next_state]))
    return []

def greedy_search(
    start: Any,
    goal_test: Callable[[Any], bool],
    successors: Callable[[Any], List[Tuple[Any, float]]],
    heuristic: Callable[[Any], float]
) -> List[Any]:
    # TODO: Docstring
    open_set: List[Tuple[float, Any]] = [(heuristic(start), start)]
    came_from: Dict[Any, Optional[Any]] = {start: None}
    visited: Set[Any] = set()

    while open_set:
        _, current = heappop(open_set)
        if goal_test(current):
            return reconstruct_path(came_from, current)
        if current not in visited:
            visited.add(current)
            for neighbor, _ in successors(current):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    heappush(open_set, (heuristic(neighbor), neighbor))
    return []

def a_star_search(
    start: Any,
    goal_test: Callable[[Any], bool],
    successors: Callable[[Any], List[Tuple[Any, float]]],
    heuristic: Callable[[Any], float]
) -> List[Any]:
    # TODO: Docstring
    open_set: List[Tuple[float, Any]] = [(heuristic(start), start)]
    came_from: Dict[Any, Optional[Any]] = {start: None}
    g_score: Dict[Any, float] = {start: 0}

    while open_set:
        _, current = heappop(open_set)
        if goal_test(current):
            return reconstruct_path(came_from, current)
        for neighbor, cost in successors(current):
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heappush(open_set, (tentative_g + heuristic(neighbor), neighbor))
    return []

def weighted_a_star_search(
    start: Any,
    goal_test: Callable[[Any], bool],
    successors: Callable[[Any], List[Tuple[Any, float]]],
    heuristic: Callable[[Any], float],
    weight: float = 1.5
) -> List[Any]:
    # TODO: Docstring
    open_set: List[Tuple[float, Any]] = [(weight*heuristic(start), start)]
    came_from: Dict[Any, Optional[Any]] = {start: None}
    g_score: Dict[Any, float] = {start: 0}

    while open_set:
        _, current = heappop(open_set)
        if goal_test(current):
            return reconstruct_path(came_from, current)
        for neighbor, cost in successors(current):
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heappush(open_set, (tentative_g + weight*heuristic(neighbor), neighbor))
    return []

def ida_star_search(
    start: Any,
    goal_test: Callable[[Any], bool],
    successors: Callable[[Any], List[Tuple[Any, float]]],
    heuristic: Callable[[Any], float]
) -> List[Any]:
    # TODO: Docstring
    bound = heuristic(start)
    path = [start]

    def dfs(node: Any, g: float, bound: float) -> Tuple[Optional[List[Any]], float]:
        f = g + heuristic(node)
        if f > bound:
            return None, f
        if goal_test(node):
            return path[:], f
        min_bound = float('inf')
        for neighbor, cost in successors(node):
            if neighbor not in path:
                path.append(neighbor)
                result, t = dfs(neighbor, g + cost, bound)
                if result:
                    return result, t
                min_bound = min(min_bound, t)
                path.pop()
        return None, min_bound

    while True:
        result, t = dfs(start, 0, bound)
        if result:
            return result
        if t == float('inf'):
            return []
        bound = t

def binary_search(
        arr: List[float], 
        target: float
) -> Optional[int]:
    # TODO: Docstring
    left, right = 0, len(arr)-1
    while left <= right:
        mid = (left+right)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid+1
        else:
            right = mid-1
    return None

def ternary_search(
        arr: List[float], 
        target: float
) -> Optional[int]:
    # TODO: Docstring
    left, right = 0, len(arr)-1
    while left <= right:
        third = (right-left)//3
        mid1 = left + third
        mid2 = right - third
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        if target < arr[mid1]:
            right = mid1-1
        elif target > arr[mid2]:
            left = mid2+1
        else:
            left = mid1+1
            right = mid2-1
    return None

def hill_climbing(
    start: Any,
    successors: Callable[[Any], List[Tuple[Any, float]]],
    objective: Callable[[Any], float]
) -> Any:
    # TODO: Docstring
    current = start
    while True:
        neighbors = successors(current)
        if not neighbors:
            break
        next_node = max(neighbors, key=lambda x: objective(x[0]))[0]
        if objective(next_node) <= objective(current):
            break
        current = next_node
    return current

def simulated_annealing(
    start: Any,
    successors: Callable[[Any], List[Tuple[Any, float]]],
    objective: Callable[[Any], float],
    temp: float = 1000.0,
    cooling_rate: float = 0.95,
    min_temp: float = 1e-3
) -> Any:
    # TODO: Docstring
    current = start
    while temp > min_temp:
        neighbors = successors(current)
        if not neighbors:
            break
        next_node = random.choice(neighbors)[0]
        delta = objective(next_node) - objective(current)
        if delta > 0 or math.exp(delta/temp) > random.random():
            current = next_node
        temp *= cooling_rate
    return current

def beam_search(
    start: Any,
    successors: Callable[[Any], List[Tuple[Any, float]]],
    objective: Callable[[Any], float],
    beam_width: int = 3,
    steps: int = 50
) -> Any:
    # TODO: Docstring
    beam: List[Any] = [start]
    for _ in range(steps):
        candidates: List[Tuple[Any, float]] = []
        for node in beam:
            for neighbor, _ in successors(node):
                candidates.append((neighbor, objective(neighbor)))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = [x[0] for x in candidates[:beam_width]]
    return beam[0] if beam else start
