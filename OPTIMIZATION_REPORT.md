# TOMAS ARC-AGI-3 Solver — v2.1 全面优化报告

**日期**: 2026-06-22  
**版本**: v2.1 (优化版)  
**测试**: 227/227 PASSED | CLI 3任务×3模式全通过

---

## 优化总览

| # | 类别 | 文件 | 优化内容 | 预估加速 |
|---|------|------|----------|----------|
| 1 | 向量化 | `dsl_primitives.py` | `_resize`: np.ix_替代双重循环 | 10-50x |
| 2 | 向量化 | `dsl_primitives.py` | `_move`: 切片替代双重循环 | 10-50x |
| 3 | 向量化 | `dsl_primitives.py` | `_copy`: np.where+fancy indexing | 10-30x |
| 4 | 向量化 | `octonion_hyperedge.py` | `decode_to_grid`: np.where批量解码 | 20-50x |
| 5 | 向量化 | `octonion_hyperedge.py` | `_compute_area_ratios`: bincount lookup | 10-20x |
| 6 | 向量化 | `dsl_primitives.py` | `_symmetry_detect`: np.fliplr+mask | 20-50x |
| 7 | 算法 | `video_tensor.py` | Betti₀预计算缓存 | 2x |
| 8 | 算法 | `dsl_primitives.py` | `_extract_pattern`: np.tile替代四重循环 | 5-10x |
| 9 | 算法 | `dsl_primitives.py` | `_skeleton`: ndimage.skeletonize | 10-20x |
| 10 | 算法 | `dsl_primitives.py` | `_gravity`: 紧凑循环+提前退出 | 1.5x |
| 11 | 架构 | `gaussex_verifier.py` | CRC32 hash + fiber缓存 | 2-5x |
| 12 | 架构 | `delta_composer.py` | clone()替代deepcopy | ~3x |
| 13 | 架构 | `kappa_snap_searcher.py` | Phase B ThreadPoolExecutor并行 | 2-4x |
| 14 | 架构 | `tomas_solver.py` | parse_input缓存避免重复 | 2x |

---

## 详细变更

### 1. `_resize` 向量化
```python
# Before: O(H*W) Python double loop
for i in range(new_h):
    for j in range(new_w):
        orig_i = int(i * h / new_h)
        ...
# After: O(1) vectorized np.ix_
row_idx = np.clip((np.arange(new_h) * h / new_h).astype(np.int32), 0, h - 1)
col_idx = np.clip((np.arange(new_w) * w / new_w).astype(np.int32), 0, w - 1)
return grid[np.ix_(row_idx, col_idx)]
```

### 2. `_move` 向量化
```python
# Before: O(H*W) double loop
for i in range(h):
    for j in range(w):
        ni, nj = i + dy, j + dx
        if 0 <= ni < h and 0 <= nj < w:
            result[ni, nj] = grid[i, j]
# After: O(1) slicing
src_r0, src_r1 = max(0, -dy), min(h, h - dy)
src_c0, src_c1 = max(0, -dx), min(w, w - dx)
dst_r0, dst_c0 = max(0, dy), max(0, dx)
result[dst_r0:..., dst_c0:...] = grid[src_r0:..., src_c0:...]
```

### 3. `_copy` 向量化
```python
# Before: O(H*W) loop with condition
# After: np.where + boolean mask fancy indexing
mask = grid != 0
yy, xx = np.where(mask)
valid = (0 <= ny) & (ny < h) & (0 <= nx) & (nx < w)
result[ny[valid], nx[valid]] = grid[yy[valid], xx[valid]]
```

### 4. `decode_to_grid` 向量化
```python
# Before: O(H*W) per-pixel loop
for i in range(h * w):
    row, col = divmod(i, w)
    mask = encoded[i, 0]
    if mask > 0.5:
        color = int(round(phase / (2*np.pi) * MAX_COLOR))
        ...
# After: O(#non_zero) batch indexing
pixel_indices = np.where(mask)[0]
rows = pixel_indices // w
cols = pixel_indices % w
colors = np.round(phases / (2*np.pi) * MAX_COLOR).astype(np.int8)
grid[rows, cols] = colors
```

### 5. `_compute_area_ratios` 向量化
```python
# Before: O(#labels) loops with np.sum(==label)
for label in unique_labels:
    count = np.sum(comp_labels == label)
    ratios[comp_labels == label] = count / total
# After: O(1) bincount + lookup table
counts = np.bincount(flat, minlength=max_label + 1)
ratio_lookup[1:] = counts[1:] / max(total, 1)
return ratio_lookup[comp_labels]
```

### 6. `_symmetry_detect` 完全向量化
```python
# Before: O(H*W) double nested loop
for i in range(h):
    for j in range(w // 2):
        mirror_j = w - 1 - j
        if result[i, j] == 0 and result[i, mirror_j] != 0: ...
# After: O(1) vectorized flip + mask
h_flipped = np.fliplr(result)
h_mask = (result == 0) & (h_flipped != 0)
result[h_mask] = h_flipped[h_mask]
```

### 7. Betti₀ 预计算缓存
```python
# Before: per-iteration compute_betti0()→compute_topo_invariants()
for i in range(1, len(self.temporal_hypergraphs)):
    prev_betti = self.temporal_hypergraphs[i - 1].compute_betti0()
    curr_betti = self.temporal_hypergraphs[i].compute_betti0()
# After: pre-compute once
betti_vals = self.get_betti_sequence()
for i in range(1, len(betti_vals)):
    ...
```

### 8. `_extract_pattern` np.tile
```python
# Before: O(H*W*min(H,W)²) quadruple nested loop
for i in range(0, h, tile_h):
    for j in range(0, w, tile_w):
        if not np.array_equal(grid[i:i+tile_h, j:j+tile_w], tile):
            matches = False
# After: O(1) np.tile single comparison
if np.array_equal(grid, np.tile(tile, (h // tile_h, w // tile_w))):
    return tile.copy()
```

### 9. `_skeleton` 使用 scipy
```python
# Before: O(H*W) manual neighbor sum loop
for i in range(1, h-1):
    for j in range(1, w-1):
        neighbors = binary[i-1:i+2, j-1:j+2]
        if np.sum(neighbors) <= 4: ...
# After: O(1) scipy built-in
skeleton = ndimage.skeletonize(binary)
```

### 10. `_gravity` 紧凑化
```python
# Before: redundant zero-assignment
result[-len(non_zero):, j] = non_zero if len(non_zero) > 0 else 0
# After: early-exit guard
if len(nz):
    result[dst, j] = nz
```

### 11. GaussEx hash优化
```python
# Before: Python hash(tobytes()) + no caching
state_hash = hash(result.tobytes())
# After: CRC32 (faster) + per-program fiber cache
@staticmethod
def _fast_array_hash(arr):
    import zlib
    return zlib.crc32(arr.tobytes())
# + cache_key = (id(program), pair_idx)
```

### 12. delta_composer clone()
```python
# Before: copy.deepcopy (full recursive deep copy)
deltas = [copy.deepcopy(d) for d in self.delta_list]
# After: custom clone() (~3x faster)
deltas = [d.clone() for d in self.delta_list]
# clone() recursively copies element + children without inspecting
# all Python internals like deepcopy does
```

### 13. Phase B 并行化
```python
# Before: serial for-loop verification
for program in ranked:
    if self._verify_against_demos(program, demo_pairs):
        valid.append(program)
# After: ThreadPoolExecutor (numpy releases GIL)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_program = {
        executor.submit(self._verify_against_demos, p, demo_pairs): p
        for p in filtered
    }
    for future in as_completed(future_to_program):
        if future.result():
            valid.append(program)
```

### 14. tomas_solver 解析缓存
```python
# Before: parse_input called in solve() AND solve_video/bayesian/fusion
video_task = self.parse_input(task)  # in solve()
video_task = self.parse_input(task)  # again in solve_video()
# After: parse once, pass to mode solvers
video_task = self.parse_input(task)  # once
result = self.solve_video(task, video_task)  # pass cached
```

---

## 验证结果

| 验证项 | 结果 |
|--------|------|
| 单元测试 | 227/227 PASSED |
| CLI task_001 (video) | ✅ 11.66s, MDL=22, 5 programs |
| CLI task_001 (bayesian) | ✅ 13.27s, MDL=5 |
| CLI task_001 (fusion) | ✅ 15.39s, MDL=5 |
| CLI task_002 (video) | ✅ ~16s, 750 programs |
| CLI task_003 (video) | ✅ ~10s, 5 programs |

---

## 后续建议

1. **Phase A 并行化** — 拓扑哈希过滤也可并行
2. **JIT编译** — 热路径用numba.jit进一步加速
3. **depth≥3剪枝** — 基于Betti₀不变量提前剪枝组合爆炸
4. **C扩展** — 核心八元数运算用Cython重写
5. **GPU加速** — 批量程序验证用CuPy替代numpy
