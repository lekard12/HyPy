from typing import Any, Callable, List, Optional
from collections import deque
import random
import heapq

def bubble_sort(
        arr: List[Any], 
        key: Optional[Callable] = None, 
        reverse: bool = False, 
        inplace: bool = False
) -> List[Any]:
    # TODO: Docstring
    a = arr if inplace else arr[:]
    n = len(a)
    for i in range(n):
        for j in range(0, n-i-1):
            x, y = a[j], a[j+1]
            if key:
                x, y = key(x), key(y)
            if (x > y) ^ reverse:
                a[j], a[j+1] = a[j+1], a[j]
    return a

def selection_sort(
        arr: List[Any], 
        key: Optional[Callable] = None, 
        reverse: bool = False, 
        inplace: bool = False
) -> List[Any]:
    # TODO: Docstring
    a = arr if inplace else arr[:]
    n = len(a)
    for i in range(n):
        idx = i
        for j in range(i+1, n):
            x, y = a[j], a[idx]
            if key:
                x, y = key(x), key(y)
            if (x < y) ^ reverse:
                idx = j
        a[i], a[idx] = a[idx], a[i]
    return a

def insertion_sort(
        arr: List[Any], 
        key: Optional[Callable] = None, 
        reverse: bool = False, 
        inplace: bool = False
) -> List[Any]:
    # TODO: Docstring
    a = arr if inplace else arr[:]
    for i in range(1, len(a)):
        current = a[i]
        j = i - 1
        while j >= 0:
            x, y = a[j], current
            if key:
                x, y = key(x), key(y)
            if (x > y) ^ reverse:
                a[j+1] = a[j]
                j -= 1
            else:
                break
        a[j+1] = current
    return a

def merge_sort(
        arr: List[Any], 
        key: Optional[Callable] = None, 
        reverse: bool = False
) -> List[Any]:
    # TODO: Docstring
    if len(arr) <= 1:
        return arr[:]

    def merge(
            left: List[Any], 
            right: List[Any]
) -> List[Any]:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            x, y = left[i], right[j]
            if key:
                x, y = key(x), key(y)
            if (x <= y) ^ reverse:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key, reverse)
    right = merge_sort(arr[mid:], key, reverse)
    return merge(left, right)

def quick_sort(
        arr: List[Any], 
        key: Optional[Callable] = None, 
        reverse: bool = False, 
        inplace: bool = False
) -> List[Any]:
    # TODO: Docstring
    a = arr if inplace else arr[:]

    def partition(low: int, high: int) -> int:
        pivot_idx = random.randint(low, high)
        a[pivot_idx], a[high] = a[high], a[pivot_idx]
        pivot = a[high]
        i = low - 1
        for j in range(low, high):
            x, y = a[j], pivot
            if key:
                x, y = key(x), key(y)
            if (x <= y) ^ reverse:
                i += 1
                a[i], a[j] = a[j], a[i]
        a[i+1], a[high] = a[high], a[i+1]
        return i+1

    def qs(low: int, high: int):
        if low < high:
            pi = partition(low, high)
            qs(low, pi-1)
            qs(pi+1, high)

    qs(0, len(a)-1)
    return a

def heap_sort(
        arr: List[Any], 
        key: Optional[Callable] = None, 
        reverse: bool = False
) -> List[Any]:
    # TODO: Docstring
    a = [(key(x) if key else x, x) for x in arr]
    heapq.heapify(a)
    sorted_list = [heapq.heappop(a)[1] for _ in range(len(a))]
    if reverse:
        sorted_list.reverse()
    return sorted_list

def counting_sort(
        arr: List[int], 
        max_val: Optional[int] = None
) -> List[int]:
    # TODO: Docstring
    if not arr:
        return []
    max_val = max_val if max_val is not None else max(arr)
    count = [0] * (max_val + 1)
    for num in arr:
        count[num] += 1
    result = []
    for i, c in enumerate(count):
        result.extend([i] * c)
    return result

def radix_sort(
        arr: List[int]
) -> List[int]:
    # TODO: Docstring
    if not arr:
        return []
    max_val = max(arr)
    exp = 1
    a = arr[:]
    while max_val // exp > 0:
        count = [0] * 10
        output = [0] * len(a)
        for num in a:
            count[(num // exp) % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for i in reversed(range(len(a))):
            idx = (a[i] // exp) % 10
            output[count[idx] - 1] = a[i]
            count[idx] -= 1
        a = output[:]
        exp *= 10
    return a

def sliding_window_max(
        arr: List[int], 
        k: int
) -> List[int]:
    # TODO: Docstring
    if not arr or k <= 0:
        return []
    dq = deque()
    result = []
    for i, n in enumerate(arr):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and arr[dq[-1]] <= n:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result

def quickselect(
        arr: List[int], 
        k: int
) -> int:
    # TODO: Docstring
    a = arr[:]
    def partition(low: int, high: int) -> int:
        pivot_idx = random.randint(low, high)
        a[pivot_idx], a[high] = a[high], a[pivot_idx]
        pivot = a[high]
        i = low
        for j in range(low, high):
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[high] = a[high], a[i]
        return i

    low, high = 0, len(a)-1
    while low <= high:
        pi = partition(low, high)
        if pi == k:
            return a[pi]
        elif pi < k:
            low = pi + 1
        else:
            high = pi - 1
    return -1

def dutch_national_flag(
        arr: List[int]
) -> List[int]:
    # TODO: Docstring
    a = arr[:]
    low, mid, high = 0, 0, len(a) - 1
    while mid <= high:
        if a[mid] == 0:
            a[low], a[mid] = a[mid], a[low]
            low += 1
            mid += 1
        elif a[mid] == 1:
            mid += 1
        else:
            a[mid], a[high] = a[high], a[mid]
            high -= 1
    return a

def move_zeros(
        arr: List[int]
) -> List[int]:
    # TODO: Docstring
    a = arr[:]
    pos = 0
    for i in range(len(a)):
        if a[i] != 0:
            a[pos], a[i] = a[i], a[pos]
            pos += 1
    return a

def remove_duplicates_sorted(
        arr: List[int]
) -> List[int]:
    # TODO: Docstring
    if not arr:
        return []
    a = arr[:]
    idx = 1
    for i in range(1, len(a)):
        if a[i] != a[i-1]:
            a[idx] = a[i]
            idx += 1
    return a[:idx]

def rotate_array(
        arr: List[Any], 
        k: int
) -> List[Any]:
    # TODO: Docstring
    n = len(arr)
    k %= n
    return arr[-k:] + arr[:-k]

def prefix_sums(
        arr: List[int]
) -> List[int]:
    # TODO: Docstring
    ps = [0]
    for num in arr:
        ps.append(ps[-1] + num)
    return ps
