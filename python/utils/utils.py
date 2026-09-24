import json
import random

def read_json(jsonpath):
    with open(jsonpath) as f:
        js = json.load(f)
    return js

def write_json(jsonpath, js):
    with open(jsonpath, "w") as f:
        json.dump(js, f, indent=4)

def get_unique_random_integers(a, b, n):
    if n > (b - a + 1):
        raise ValueError("n cannot be larger than the number of elements in the range")
    return sorted(random.sample(range(a, b + 1), n))

def tomatrix(mat, is_sparse, size=None):
    if not is_sparse: return mat

    ids = {i for i, j, _ in mat} | {j for i, j, _ in mat}
    if size is None:
        if ids != set(range(max(ids) + 1)): raise ValueError("id is not continuous")
        size = max(ids) + 1
    elif any(i >= size or j >= size for i, j, _ in mat):
        raise ValueError("id is out of range")

    mat2 = [[0] * size for _ in range(size)]
    for i, j, v in mat: mat2[i][j] = mat2[j][i] = v
    return mat2
