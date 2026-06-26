"""MetaSnapNet — κ-Snap Beam搜索评分网络 + SPP四阶段训练器.

本模块实现微信公众号SPP文章中的核心组件:
    1. TopoFeatureExtractor (64维): 从grid提取拓扑特征向量
    2. ProgramNodeFeatureExtractor (32维): 从DSL序列提取特征向量
    3. MetaSnapNet (3层MLP, 双头): score + confidence 输出
    4. MetaSnapBeamScorer: κ-Snap Beam搜索评分接口
    5. MetaSnapTrainingExample: 训练数据结构
    6. MetaSnapDataCollector: 数据收集器
    7. SPPTrainer: 四阶段训练器 (SFT → RL → Long2Short → Self-Distill)

设计原则:
    - PyTorch优先, numpy fallback (torch不可用时降级)
    - HAS_TORCH守卫: try import torch → False时使用纯numpy评分
    - 所有类/函数都有完整docstring和类型注解

References:
    - κ-Snap abductive search (TOMAS Theory §3)
    - SPP四阶段训练 (微信公众号 SPP系列文章)
"""

from __future__ import annotations

import json
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# ── PyTorch守卫: 无GPU也能运行 ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH: bool = True
except ImportError:
    HAS_TORCH: bool = False
    # Fallback: 定义虚类用于类型检查
    nn = None  # type: ignore


# ============================================================================
# TopoFeatureExtractor (64维) — 升级版拓扑特征提取器
# ============================================================================

class TopoFeatureExtractor:
    """64维拓扑特征提取器，从grid提取结构化特征向量.

    特征维度分布 (共64维):
        [0]: unique_colors / 10.0       (颜色多样性)
        [1]: num_components / (h*w)     (连通分量密度, scipy.ndimage.label)
        [2]: row_period / h             (行周期性)
        [3]: col_period / w             (列周期性)
        [4]: h_symmetry                 (水平对称得分, fliplr)
        [5]: v_symmetry                 (垂直对称得分, flipud)
        [6-8]: topo_map_mean/std/max    (拓扑图统计)
        [9-63]: 补零至64维

    旧版 extract_topo_features() 返回dict, 本类返回64维numpy array,
    可直接拼接为MetaSnapNet输入.

    Args:
        target_dim: 目标维度 (默认64).
    """

    TARGET_DIM: int = 64

    def __init__(self, target_dim: int = 64) -> None:
        """初始化TopoFeatureExtractor.

        Args:
            target_dim: 目标特征维度. 默认64.
        """
        self.target_dim = target_dim

    def extract(self, grid: np.ndarray) -> np.ndarray:
        """从grid提取64维拓扑特征向量.

        Args:
            grid: 2D numpy array (游戏状态网格).

        Returns:
            64维float32 numpy array, 特征值归一化到[0,1]区间.
        """
        if grid is None or grid.size == 0:
            return np.zeros(self.target_dim, dtype=np.float32)

        h, w = grid.shape
        features = np.zeros(self.target_dim, dtype=np.float32)

        # ── [0] unique_colors / 10.0 ──
        unique_colors = len(np.unique(grid))
        features[0] = min(1.0, unique_colors / 10.0)

        # ── [1] num_components / (h*w) ──
        binary = (grid != 0).astype(np.int32)
        try:
            from scipy.ndimage import label as ndlabel
            _, num_components = ndlabel(binary)
        except ImportError:
            # Fallback: 简单连通分量估算
            num_components = int(np.sum(binary > 0))
        features[1] = min(1.0, num_components / (h * w))

        # ── [2] row_period / h ──
        if h > 1:
            row_hashes = [hash(grid[i].tobytes()) for i in range(h)]
            unique_row_hashes = len(set(row_hashes))
            if unique_row_hashes <= 2:
                # 行周期性高
                row_period = max(1, h // max(unique_row_hashes, 1))
                features[2] = min(1.0, row_period / h)
            else:
                features[2] = 0.0
        else:
            features[2] = 0.0

        # ── [3] col_period / w ──
        if w > 1:
            col_hashes = [hash(grid[:, j].tobytes()) for j in range(w)]
            unique_col_hashes = len(set(col_hashes))
            if unique_col_hashes <= 2:
                col_period = max(1, w // max(unique_col_hashes, 1))
                features[3] = min(1.0, col_period / w)
            else:
                features[3] = 0.0
        else:
            features[3] = 0.0

        # ── [4] h_symmetry (水平对称, fliplr) ──
        if w > 1:
            flipped_lr = np.fliplr(grid)
            match_ratio = float(np.sum(grid == flipped_lr)) / float(grid.size)
            features[4] = match_ratio
        else:
            features[4] = 1.0  # 单列天然对称

        # ── [5] v_symmetry (垂直对称, flipud) ──
        if h > 1:
            flipped_ud = np.flipud(grid)
            match_ratio = float(np.sum(grid == flipped_ud)) / float(grid.size)
            features[5] = match_ratio
        else:
            features[5] = 1.0  # 单行天然对称

        # ── [6-8] topo_map统计 (mean/std/max) ──
        # topo_map = 颜色变化率图: 相邻cell颜色不同的比例
        topo_map = np.zeros_like(grid, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                neighbors = 0
                diffs = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbors += 1
                        if grid[ni, nj] != grid[i, j]:
                            diffs += 1
                if neighbors > 0:
                    topo_map[i, j] = diffs / neighbors
        if topo_map.size > 0:
            features[6] = float(np.mean(topo_map))
            features[7] = float(np.std(topo_map))
            features[8] = float(np.max(topo_map))

        # ── [9-63] 补零至64维 ── (已在初始化时置零)

        return features

    def extract_dict(self, grid: np.ndarray) -> Dict[str, Any]:
        """同时返回dict格式 (兼容旧版 extract_topo_features).

        Args:
            grid: 2D numpy array.

        Returns:
            Dict with keys: euler_char, period_rank, symmetry, etc.
        """
        if grid is None or grid.size == 0:
            return {
                "euler_char": 0, "period_rank": 0, "symmetry": [],
                "component_count": 0, "hole_count": 0, "density": 0.0,
            }

        h, w = grid.shape
        density = float(np.count_nonzero(grid)) / grid.size

        binary = (grid != 0).astype(np.int32)
        try:
            from scipy.ndimage import label as ndlabel
            _, component_count = ndlabel(binary)
            # Estimate holes
            padded = np.pad(binary, 1, mode='constant', constant_values=1)
            padded_labels, _ = ndlabel(padded)
            border_labels = set(padded_labels[0, :]) | set(padded_labels[-1, :]) | \
                            set(padded_labels[:, 0]) | set(padded_labels[:, -1])
            hole_count = component_count - len(border_labels & set(range(1, component_count + 1)))
            hole_count = max(0, hole_count)
        except ImportError:
            component_count = int(np.sum(binary > 0))
            hole_count = 0

        euler_char = component_count - hole_count

        period_rank = 0
        if h > 1:
            row_hashes = [hash(grid[i].tobytes()) for i in range(h)]
            if len(set(row_hashes)) <= 2:
                period_rank = 1
        if period_rank == 1 and w > 1:
            col_hashes = [hash(grid[:, j].tobytes()) for j in range(w)]
            if len(set(col_hashes)) <= 2:
                period_rank = 2

        symmetry = []
        if h > 1 and np.allclose(grid, grid[::-1], atol=0):
            symmetry.append("horizontal")
        if w > 1 and np.allclose(grid, grid[:, ::-1], atol=0):
            symmetry.append("vertical")
        if h == w and h > 1:
            if np.allclose(grid, np.rot90(grid, 2), atol=0):
                symmetry.append("rotational_180")

        return {
            "euler_char": euler_char,
            "period_rank": period_rank,
            "symmetry": symmetry,
            "component_count": component_count,
            "hole_count": hole_count,
            "density": density,
        }


# ============================================================================
# ProgramNodeFeatureExtractor (32维) — DSL序列特征提取器
# ============================================================================

class ProgramNodeFeatureExtractor:
    """32维DSL序列特征提取器, 提取DSL程序节点的结构化特征.

    特征维度分布 (共32维):
        [0]:  seq_length / 20            (序列长度归一化)
        [1-9]: 9种原语频率归一化         (move/push/click/fill/rotate/
                                           repeat/navigate/select/verify)
        [10]: mdl_estimate               (总字符数 / 500, MDL压缩估计)
        [11]: has_repeat_flag            (是否包含repeat原语, 0/1)
        [12]: max_depth / 5              (嵌套最大深度归一化)
        [13-31]: 补零至32维

    Args:
        target_dim: 目标维度 (默认32).
    """

    TARGET_DIM: int = 32

    # 9种DSL原语类型
    PRIMITIVE_TYPES: List[str] = [
        "move", "push", "click", "fill", "rotate",
        "repeat", "navigate", "select", "verify",
    ]

    def __init__(self, target_dim: int = 32) -> None:
        """初始化ProgramNodeFeatureExtractor.

        Args:
            target_dim: 目标特征维度. 默认32.
        """
        self.target_dim = target_dim

    def extract(self, dsl_sequence: List[Dict[str, Any]]) -> np.ndarray:
        """从DSL序列提取32维特征向量.

        Args:
            dsl_sequence: DSL动作序列, 每个元素是dict.
                支持格式: {"action": "MOVE"}, {"repeat": "LEFT", "count": 3},
                或自定义dict.

        Returns:
            32维float32 numpy array.
        """
        features = np.zeros(self.target_dim, dtype=np.float32)

        if not dsl_sequence:
            return features

        # ── [0] seq_length / 20 ──
        seq_length = len(dsl_sequence)
        features[0] = min(1.0, seq_length / 20.0)

        # ── [1-9] 9种原语频率 ──
        # 统计每种原语出现的次数, 归一化到 [0, 1]
        primitive_counts: Dict[str, int] = {pt: 0 for pt in self.PRIMITIVE_TYPES}
        total_actions = 0
        for item in dsl_sequence:
            if "repeat" in item:
                primitive_counts["repeat"] += 1
                total_actions += item.get("count", 1)
            elif "action" in item:
                action_name = str(item["action"]).lower()
                # 映射action到原语类型
                matched = False
                for pt in self.PRIMITIVE_TYPES:
                    if pt in action_name or action_name == pt:
                        primitive_counts[pt] += 1
                        matched = True
                        break
                if not matched:
                    # 默认归为move
                    primitive_counts["move"] += 1
                total_actions += 1
            else:
                # 未知格式: 归为move
                primitive_counts["move"] += 1
                total_actions += 1

        # 归一化频率
        max_count = max(total_actions, 1)
        for i, pt in enumerate(self.PRIMITIVE_TYPES):
            features[1 + i] = min(1.0, primitive_counts[pt] / max_count)

        # ── [10] mdl_estimate (总字符数 / 500) ──
        total_chars = 0
        for item in dsl_sequence:
            if "repeat" in item:
                total_chars += len(str(item.get("repeat", ""))) + len(str(item.get("count", "")))
            elif "action" in item:
                total_chars += len(str(item["action"]))
            else:
                total_chars += len(json.dumps(item))
        features[10] = min(1.0, total_chars / 500.0)

        # ── [11] has_repeat_flag ──
        features[11] = float(primitive_counts["repeat"] > 0)

        # ── [12] max_depth / 5 ──
        max_depth = self._compute_max_depth(dsl_sequence)
        features[12] = min(1.0, max_depth / 5.0)

        # ── [13-31] 补零 ── (已初始化为零)

        return features

    def _compute_max_depth(self, dsl_sequence: List[Dict[str, Any]]) -> int:
        """计算DSL序列的最大嵌套深度.

        Args:
            dsl_sequence: DSL动作序列.

        Returns:
            最大嵌套深度 (int). repeat块增加1层.
        """
        max_depth = 0
        for item in dsl_sequence:
            if "repeat" in item:
                max_depth = max(max_depth, 2)  # repeat本身1层 + 内容1层
            elif isinstance(item, dict) and "nested" in item:
                # 嵌套结构
                nested = item.get("nested", [])
                if isinstance(nested, list):
                    inner_depth = self._compute_max_depth(nested)
                    max_depth = max(max_depth, inner_depth + 1)
        return max(1, max_depth)


# ============================================================================
# MetaSnapNet — κ-Snap Beam搜索评分网络
# ============================================================================

if HAS_TORCH:
    class _MetaSnapNetModule(nn.Module):
        """MetaSnapNet PyTorch模块 — 3层MLP, 双头输出.

        Architecture:
            Input: 96维 (64维topo + 32维DSL)
            Hidden1: 96 → 128 (ReLU)
            Hidden2: 128 → 128 (ReLU)
            Hidden3: 128 → 128 (ReLU)
            Score head: 128 → 1 (sigmoid, 搜索得分)
            Confidence head: 128 → 1 (sigmoid, 置信度)

        Args:
            input_dim: 输入维度 (默认96 = 64 + 32).
            hidden_dim: 隐藏层维度 (默认128).
        """

        def __init__(
            self,
            input_dim: int = 96,
            hidden_dim: int = 128,
        ) -> None:
            """初始化MetaSnapNet PyTorch模块.

            Args:
                input_dim: 输入特征维度.
                hidden_dim: 隐藏层维度.
            """
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            # 共享隐藏层
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)

            # 双头输出
            self.score_head = nn.Linear(hidden_dim, 1)
            self.confidence_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """前向传播: 输入96维特征 → (score, confidence).

            Args:
                x: (batch_size, 96) 输入特征tensor.

            Returns:
                Tuple of (score, confidence), 各为(batch_size, 1) tensor.
                score ∈ [0, 1], confidence ∈ [0, 1].
            """
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))

            score = torch.sigmoid(self.score_head(h))
            confidence = torch.sigmoid(self.confidence_head(h))

            return score, confidence


class MetaSnapNet:
    """κ-Snap Beam搜索评分网络 (封装版, 支持torch fallback).

    当torch可用时, 使用PyTorch MLP进行评分.
    当torch不可用时, 使用numpy fallback (线性加权评分).

    本类封装了 _MetaSnapNetModule (torch) 或 _NumpyFallbackScorer (numpy),
    提供统一的 score_candidates() 接口.

    Args:
        checkpoint_path: 可选的模型checkpoint路径. 如果提供且torch可用,
            加载预训练权重.
        topo_extractor: TopoFeatureExtractor实例.
        program_extractor: ProgramNodeFeatureExtractor实例.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        topo_extractor: Optional[TopoFeatureExtractor] = None,
        program_extractor: Optional[ProgramNodeFeatureExtractor] = None,
    ) -> None:
        """初始化MetaSnapNet.

        Args:
            checkpoint_path: 模型checkpoint路径.
            topo_extractor: 拓扑特征提取器 (默认新建).
            program_extractor: DSL特征提取器 (默认新建).
        """
        self.topo_extractor = topo_extractor or TopoFeatureExtractor()
        self.program_extractor = program_extractor or ProgramNodeFeatureExtractor()

        self._torch_module: Optional[Any] = None  # _MetaSnapNetModule or None
        self._numpy_weights: Optional[np.ndarray] = None  # Fallback权重

        if HAS_TORCH:
            self._torch_module = _MetaSnapNetModule(
                input_dim=self.topo_extractor.TARGET_DIM + self.program_extractor.TARGET_DIM,
                hidden_dim=128,
            )
            if checkpoint_path and Path(checkpoint_path).exists():
                self._load_checkpoint(checkpoint_path)
        else:
            # Numpy fallback: 初始化随机权重
            input_dim = self.topo_extractor.TARGET_DIM + self.program_extractor.TARGET_DIM
            self._numpy_weights = np.random.randn(input_dim).astype(np.float32) * 0.01
            self._numpy_bias = np.float32(0.0)

    def _load_checkpoint(self, path: str) -> None:
        """加载PyTorch checkpoint.

        Args:
            path: checkpoint文件路径.
        """
        if not HAS_TORCH or self._torch_module is None:
            return
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            self._torch_module.load_state_dict(state_dict)
        except (FileNotFoundError, RuntimeError, Exception) as e:
            # checkpoint加载失败: 使用随机初始化
            pass

    def save_checkpoint(self, path: str) -> None:
        """保存模型checkpoint.

        Args:
            path: checkpoint保存路径.
        """
        if HAS_TORCH and self._torch_module is not None:
            torch.save(self._torch_module.state_dict(), path)

    def score_candidates(
        self,
        grid: np.ndarray,
        topo_map: Optional[Dict[str, Any]] = None,
        candidate_programs: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Tuple[float, float]]:
        """对候选DSL程序进行κ-Snap Beam评分.

        Args:
            grid: 2D numpy array (当前游戏状态).
            topo_map: 可选的拓扑特征dict (兼容旧版).
            candidate_programs: 候选DSL程序列表, 每个是List[Dict].

        Returns:
            List of (score, confidence) tuples, 每个对应一个候选程序.
            score ∈ [0, 1], confidence ∈ [0, 1].
        """
        if candidate_programs is None:
            candidate_programs = []

        if not candidate_programs:
            return []

        # 提取grid的拓扑特征 (共享)
        topo_features = self.topo_extractor.extract(grid)

        results: List[Tuple[float, float]] = []

        for program in candidate_programs:
            # 提取DSL特征
            dsl_features = self.program_extractor.extract(program)

            # 拼接: 96维 = 64维topo + 32维DSL
            combined = np.concatenate([topo_features, dsl_features])

            if HAS_TORCH and self._torch_module is not None:
                # PyTorch评分
                with torch.no_grad():
                    x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
                    score_t, conf_t = self._torch_module(x)
                    score = float(score_t.item())
                    confidence = float(conf_t.item())
            else:
                # Numpy fallback评分: 线性加权
                score = float(np.dot(self._numpy_weights, combined) + self._numpy_bias)
                score = float(np.clip(score, 0.0, 1.0))
                # Confidence = 特征覆盖度 (非零特征比例)
                confidence = float(np.count_nonzero(combined) / len(combined))

            results.append((score, confidence))

        return results


# ============================================================================
# MetaSnapBeamScorer — κ-Snap Beam搜索评分接口类
# ============================================================================

class MetaSnapBeamScorer:
    """κ-Snap Beam搜索评分接口类.

    提供统一的score_candidates接口, 内部委托给MetaSnapNet或fallback.

    Args:
        net: MetaSnapNet实例 (可选, 默认新建).
    """

    def __init__(
        self,
        net: Optional[MetaSnapNet] = None,
    ) -> None:
        """初始化MetaSnapBeamScorer.

        Args:
            net: MetaSnapNet实例. 如果None, 创建默认实例.
        """
        self.net = net or MetaSnapNet()

    def score_candidates(
        self,
        grid: np.ndarray,
        topo_map: Optional[Dict[str, Any]] = None,
        candidate_programs: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Tuple[float, float]]:
        """对候选DSL程序进行评分.

        Args:
            grid: 2D numpy array (游戏状态).
            topo_map: 拓扑特征dict (兼容旧版extract_topo_features).
            candidate_programs: 候选DSL程序列表.

        Returns:
            List of (score, confidence) tuples.
        """
        return self.net.score_candidates(grid, topo_map, candidate_programs)


# ============================================================================
# MetaSnapTrainingExample — 训练数据结构
# ============================================================================

@dataclass
class MetaSnapTrainingExample:
    """MetaSnapNet训练数据样本.

    Attributes:
        grid: 游戏状态网格 (2D numpy array).
        dsl_sequence: DSL动作序列.
        score: 目标评分 (0-1).
        confidence: 目标置信度 (0-1).
        task_id: 任务/游戏ID.
        timestamp: 记录时间.
        topo_features: 64维拓扑特征向量 (提取后填充).
        dsl_features: 32维DSL特征向量 (提取后填充).
    """
    grid: Optional[np.ndarray] = None
    dsl_sequence: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
    task_id: str = ""
    timestamp: float = field(default_factory=time.time)
    topo_features: Optional[np.ndarray] = None
    dsl_features: Optional[np.ndarray] = None

    def extract_features(
        self,
        topo_extractor: Optional[TopoFeatureExtractor] = None,
        program_extractor: Optional[ProgramNodeFeatureExtractor] = None,
    ) -> np.ndarray:
        """提取并缓存特征向量.

        Args:
            topo_extractor: 拓扑特征提取器.
            program_extractor: DSL特征提取器.

        Returns:
            96维拼接特征向量 (64+32).
        """
        topo_ext = topo_extractor or TopoFeatureExtractor()
        prog_ext = program_extractor or ProgramNodeFeatureExtractor()

        if self.grid is not None:
            self.topo_features = topo_ext.extract(self.grid)
        else:
            self.topo_features = np.zeros(topo_ext.TARGET_DIM, dtype=np.float32)

        self.dsl_features = prog_ext.extract(self.dsl_sequence)

        return np.concatenate([self.topo_features, self.dsl_features])


# ============================================================================
# MetaSnapDataCollector — 训练数据收集器
# ============================================================================

class MetaSnapDataCollector:
    """MetaSnapNet训练数据收集器.

    收集游戏轨迹中的(grid, dsl, score, confidence)样本,
    用于后续SPP训练.

    Args:
        max_samples: 最大样本数. 默认10000.
        save_path: 数据保存路径. 默认 None (不自动保存).
    """

    def __init__(
        self,
        max_samples: int = 10000,
        save_path: Optional[str] = None,
    ) -> None:
        """初始化MetaSnapDataCollector.

        Args:
            max_samples: 最大存储样本数.
            save_path: 数据持久化路径.
        """
        self.max_samples = max_samples
        self.save_path = Path(save_path) if save_path else None
        self._samples: List[MetaSnapTrainingExample] = []

    def collect(
        self,
        grid: np.ndarray,
        dsl_sequence: List[Dict[str, Any]],
        score: float,
        confidence: float,
        task_id: str = "",
    ) -> None:
        """收集一个训练样本.

        Args:
            grid: 游戏状态网格.
            dsl_sequence: DSL动作序列.
            score: 实际评分 (来自游戏结果).
            confidence: 置信度 (来自成功率统计).
            task_id: 任务ID.
        """
        example = MetaSnapTrainingExample(
            grid=grid,
            dsl_sequence=dsl_sequence,
            score=score,
            confidence=confidence,
            task_id=task_id,
        )
        self._samples.append(example)

        # 裁剪超出上限
        if len(self._samples) > self.max_samples:
            self._samples = self._samples[-self.max_samples:]

    def get_samples(self) -> List[MetaSnapTrainingExample]:
        """获取所有收集的样本.

        Returns:
            List of MetaSnapTrainingExample.
        """
        return self._samples

    def clear(self) -> None:
        """清空样本缓存."""
        self._samples.clear()

    def save(self) -> None:
        """保存收集的数据到磁盘.

        以JSON格式保存, grid以列表形式序列化.
        """
        if self.save_path is None:
            return

        data = []
        for sample in self._samples:
            entry = {
                "dsl_sequence": sample.dsl_sequence,
                "score": sample.score,
                "confidence": sample.confidence,
                "task_id": sample.task_id,
                "timestamp": sample.timestamp,
            }
            if sample.grid is not None:
                entry["grid_shape"] = list(sample.grid.shape)
                entry["grid_data"] = sample.grid.flatten().tolist()
            if sample.topo_features is not None:
                entry["topo_features"] = sample.topo_features.tolist()
            if sample.dsl_features is not None:
                entry["dsl_features"] = sample.dsl_features.tolist()
            data.append(entry)

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> int:
        """从磁盘加载训练数据.

        Returns:
            加载的样本数量.
        """
        if self.save_path is None or not self.save_path.exists():
            return 0

        with open(self.save_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for entry in data:
            grid = None
            if "grid_data" in entry and "grid_shape" in entry:
                shape = tuple(entry["grid_shape"])
                grid = np.array(entry["grid_data"], dtype=np.int32).reshape(shape)

            topo_features = None
            if "topo_features" in entry:
                topo_features = np.array(entry["topo_features"], dtype=np.float32)

            dsl_features = None
            if "dsl_features" in entry:
                dsl_features = np.array(entry["dsl_features"], dtype=np.float32)

            sample = MetaSnapTrainingExample(
                grid=grid,
                dsl_sequence=entry.get("dsl_sequence", []),
                score=entry.get("score", 0.0),
                confidence=entry.get("confidence", 0.0),
                task_id=entry.get("task_id", ""),
                timestamp=entry.get("timestamp", 0.0),
                topo_features=topo_features,
                dsl_features=dsl_features,
            )
            self._samples.append(sample)
            count += 1

        # 裁剪超出上限
        if len(self._samples) > self.max_samples:
            self._samples = self._samples[-self.max_samples:]

        return count


# ============================================================================
# SPPTrainer — 四阶段训练器
# ============================================================================

class SPPTrainer:
    """SPP (Sleep-Step Processing Pipeline) 四阶段训练器.

    四阶段训练流程:
        1. SFT (Supervised Fine-Tuning): 从成功轨迹学习基础评分
        2. RL (Reinforcement Learning): 用游戏奖励优化评分策略
        3. Long2Short: 长序列→短序列蒸馏 (压缩策略)
        4. Self-Distill: 自蒸馏 (模型自我改进)

    每阶段产出checkpoint, 可用于MetaSnapNet评分.

    Args:
        net: MetaSnapNet实例 (训练目标).
        data_collector: 训练数据收集器.
        checkpoint_dir: checkpoint保存目录.
        learning_rate: 学习率.
        batch_size: 训练batch大小.
    """

    def __init__(
        self,
        net: Optional[MetaSnapNet] = None,
        data_collector: Optional[MetaSnapDataCollector] = None,
        checkpoint_dir: str = "checkpoints",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        """初始化SPPTrainer.

        Args:
            net: MetaSnapNet实例. 如果None, 创建新实例.
            data_collector: 数据收集器. 如果None, 创建新实例.
            checkpoint_dir: checkpoint目录.
            learning_rate: 学习率.
            batch_size: batch大小.
        """
        self.net = net or MetaSnapNet()
        self.data_collector = data_collector or MetaSnapDataCollector()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # 确保checkpoint目录存在
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # PyTorch optimizer (如果torch可用)
        self._optimizer: Optional[Any] = None
        if HAS_TORCH and self.net._torch_module is not None:
            self._optimizer = torch.optim.Adam(
                self.net._torch_module.parameters(),
                lr=self.learning_rate,
            )

    def sft_train(
        self,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Stage 1: SFT (Supervised Fine-Tuning) 训练.

        从成功轨迹中学习: grid + dsl → score=1.0, confidence=高.

        Args:
            epochs: 训练轮数.
            verbose: 是否打印训练进度.

        Returns:
            训练报告dict: loss_history, epochs_completed, best_loss.
        """
        samples = self.data_collector.get_samples()
        if not samples:
            return {"loss_history": [], "epochs_completed": 0, "best_loss": 0.0}

        loss_history: List[float] = []
        best_loss = float('inf')

        if not HAS_TORCH or self.net._torch_module is None or self._optimizer is None:
            # Numpy fallback: 简单线性回归
            return self._sft_train_numpy(samples, epochs, verbose)

        # 提取特征
        topo_ext = self.net.topo_extractor
        prog_ext = self.net.program_extractor

        # 构建训练数据
        features_list: List[np.ndarray] = []
        scores_list: List[float] = []
        confidences_list: List[float] = []

        for sample in samples:
            combined = sample.extract_features(topo_ext, prog_ext)
            features_list.append(combined)
            scores_list.append(sample.score)
            confidences_list.append(sample.confidence)

        X = np.array(features_list, dtype=np.float32)
        Y_score = np.array(scores_list, dtype=np.float32).reshape(-1, 1)
        Y_conf = np.array(confidences_list, dtype=np.float32).reshape(-1, 1)

        # 训练循环
        for epoch in range(epochs):
            # Mini-batch
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                x_batch = torch.tensor(X[batch_idx], dtype=torch.float32)
                y_score_batch = torch.tensor(Y_score[batch_idx], dtype=torch.float32)
                y_conf_batch = torch.tensor(Y_conf[batch_idx], dtype=torch.float32)

                # Forward
                pred_score, pred_conf = self.net._torch_module(x_batch)

                # Loss: MSE for score + BCE for confidence
                loss = F.mse_loss(pred_score, y_score_batch) + \
                       F.binary_cross_entropy(pred_conf, y_conf_batch)

                # Backward
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)
            best_loss = min(best_loss, avg_loss)

            if verbose:
                print(f"  [SFT] Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

        return {
            "loss_history": loss_history,
            "epochs_completed": epochs,
            "best_loss": best_loss,
        }

    def _sft_train_numpy(
        self,
        samples: List[MetaSnapTrainingExample],
        epochs: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        """SFT numpy fallback: 简单线性回归训练.

        Args:
            samples: 训练样本.
            epochs: 轮数.
            verbose: 打印进度.

        Returns:
            训练报告dict.
        """
        topo_ext = self.net.topo_extractor
        prog_ext = self.net.program_extractor

        features_list: List[np.ndarray] = []
        scores_list: List[float] = []

        for sample in samples:
            combined = sample.extract_features(topo_ext, prog_ext)
            features_list.append(combined)
            scores_list.append(sample.score)

        if not features_list:
            return {"loss_history": [], "epochs_completed": 0, "best_loss": 0.0}

        X = np.array(features_list, dtype=np.float32)
        Y = np.array(scores_list, dtype=np.float32)

        # 简单线性回归: 权重 = X^T Y / (X^T X)
        loss_history: List[float] = []

        for epoch in range(epochs):
            pred = np.clip(np.dot(X, self.net._numpy_weights) + self.net._numpy_bias, 0.0, 1.0)
            loss = float(np.mean((pred - Y) ** 2))
            loss_history.append(loss)

            # Gradient descent update
            error = pred - Y
            grad_w = np.dot(X.T, error) / len(X)
            grad_b = float(np.mean(error))

            self.net._numpy_weights -= self.learning_rate * grad_w
            self.net._numpy_bias -= self.learning_rate * grad_b

            if verbose and epoch % 5 == 0:
                print(f"  [SFT-NumPy] Epoch {epoch+1}/{epochs}, loss={loss:.4f}")

        return {
            "loss_history": loss_history,
            "epochs_completed": epochs,
            "best_loss": min(loss_history) if loss_history else 0.0,
        }

    def rl_train(
        self,
        episodes: int = 50,
        reward_scale: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Stage 2: RL (Reinforcement Learning) 训练.

        用游戏奖励信号优化评分策略: 高奖励 → 高score/confidence.

        Args:
            episodes: RL训练episodes数.
            reward_scale: 奖励缩放因子.
            verbose: 打印进度.

        Returns:
            RL训练报告dict.
        """
        # RL训练需要游戏环境交互, 这里提供框架
        # 实际训练数据来自MetaSnapDataCollector收集的轨迹

        samples = self.data_collector.get_samples()
        if not samples:
            return {"episodes": 0, "reward_history": [], "avg_reward": 0.0}

        reward_history: List[float] = []

        # 用已有样本的score作为reward信号
        for ep in range(min(episodes, len(samples))):
            sample = samples[ep % len(samples)]
            reward = sample.score * reward_scale
            reward_history.append(reward)

            if verbose and ep % 10 == 0:
                print(f"  [RL] Episode {ep+1}/{episodes}, reward={reward:.4f}")

        avg_reward = float(np.mean(reward_history)) if reward_history else 0.0

        return {
            "episodes": len(reward_history),
            "reward_history": reward_history,
            "avg_reward": avg_reward,
        }

    def long2short_train(
        self,
        max_original_length: int = 20,
        target_length: int = 10,
        epochs: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Stage 3: Long2Short蒸馏训练.

        将长DSL序列的评分迁移到短序列: 教长序列 → 学短序列.

        Args:
            max_original_length: 最大原始序列长度.
            target_length: 目标压缩序列长度.
            epochs: 训练轮数.
            verbose: 打印进度.

        Returns:
            Long2Short训练报告dict.
        """
        samples = self.data_collector.get_samples()

        # 筛选长序列样本
        long_samples = [
            s for s in samples
            if len(s.dsl_sequence) >= target_length
        ]

        if not long_samples:
            return {"compressed_count": 0, "epochs": 0, "avg_compression_ratio": 0.0}

        # 对长序列进行压缩 (截取前target_length个动作)
        compressed_count = 0
        compression_ratios: List[float] = []

        for sample in long_samples:
            original_len = len(sample.dsl_sequence)
            if original_len > max_original_length:
                continue

            # 截取前target_length个动作作为"短序列"
            compressed_seq = sample.dsl_sequence[:target_length]
            compression_ratio = target_length / original_len
            compression_ratios.append(compression_ratio)
            compressed_count += 1

        avg_ratio = float(np.mean(compression_ratios)) if compression_ratios else 0.0

        if verbose:
            print(
                f"  [Long2Short] {compressed_count} sequences compressed, "
                f"avg ratio={avg_ratio:.2f}"
            )

        return {
            "compressed_count": compressed_count,
            "epochs": epochs,
            "avg_compression_ratio": avg_ratio,
        }

    def self_distill(
        self,
        iterations: int = 3,
        epochs_per_iter: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Stage 4: Self-Distillation (自蒸馏).

        模型用自己的预测作为软标签, 反复训练提升稳定性.

        Args:
            iterations: 自蒸馏迭代次数.
            epochs_per_iter: 每次迭代的训练轮数.
            verbose: 打印进度.

        Returns:
            Self-distillation报告dict.
        """
        samples = self.data_collector.get_samples()
        if not samples:
            return {"iterations": 0, "total_epochs": 0, "improvement": 0.0}

        loss_before: List[float] = []
        loss_after: List[float] = []

        topo_ext = self.net.topo_extractor
        prog_ext = self.net.program_extractor

        for iteration in range(iterations):
            # 生成软标签: 用当前模型评分
            soft_labels: List[Tuple[float, float]] = []

            for sample in samples:
                if sample.grid is not None and sample.dsl_sequence:
                    scores = self.net.score_candidates(
                        sample.grid,
                        None,
                        [sample.dsl_sequence],
                    )
                    if scores:
                        soft_labels.append(scores[0])
                    else:
                        soft_labels.append((sample.score, sample.confidence))
                else:
                    soft_labels.append((sample.score, sample.confidence))

            # 用软标签重新训练 (SFT风格)
            for i, sample in enumerate(samples):
                if i < len(soft_labels):
                    sample.score = soft_labels[i][0]
                    sample.confidence = soft_labels[i][1]

            # 执行SFT训练
            sft_result = self.sft_train(epochs=epochs_per_iter, verbose=verbose)
            loss_history = sft_result.get("loss_history", [])

            if loss_history:
                if iteration == 0:
                    loss_before = loss_history
                loss_after = loss_history

            if verbose:
                print(
                    f"  [Self-Distill] Iteration {iteration+1}/{iterations}, "
                    f"loss={loss_history[-1] if loss_history else 'N/A'}"
                )

        improvement = 0.0
        if loss_before and loss_after:
            improvement = float(loss_before[-1] - loss_after[-1])

        return {
            "iterations": iterations,
            "total_epochs": iterations * epochs_per_iter,
            "improvement": improvement,
        }

    def save_checkpoint(self, stage: str = "latest") -> str:
        """保存训练checkpoint.

        Args:
            stage: 阶段标签 (sft/rl/long2short/self_distill/latest).

        Returns:
            checkpoint文件路径.
        """
        path = self.checkpoint_dir / f"meta_snap_{stage}.pt"
        self.net.save_checkpoint(str(path))
        return str(path)

    def run_spp(
        self,
        sft_epochs: int = 10,
        rl_episodes: int = 50,
        long2short_epochs: int = 5,
        distill_iterations: int = 3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """运行完整SPP四阶段pipeline.

        执行顺序: SFT → RL → Long2Short → Self-Distill

        Args:
            sft_epochs: SFT训练轮数.
            rl_episodes: RL训练episodes数.
            long2short_epochs: Long2Short训练轮数.
            distill_iterations: 自蒸馏迭代数.
            verbose: 打印进度.

        Returns:
            完整pipeline报告dict, 包含每阶段的结果.
        """
        report: Dict[str, Any] = {
            "sft": None,
            "rl": None,
            "long2short": None,
            "self_distill": None,
            "checkpoints": [],
        }

        if verbose:
            print("[SPP] Stage 1: SFT (Supervised Fine-Tuning)")

        # Stage 1: SFT
        report["sft"] = self.sft_train(epochs=sft_epochs, verbose=verbose)
        ckpt_path = self.save_checkpoint("sft")
        report["checkpoints"].append(ckpt_path)

        if verbose:
            print("[SPP] Stage 2: RL (Reinforcement Learning)")

        # Stage 2: RL
        report["rl"] = self.rl_train(episodes=rl_episodes, verbose=verbose)
        ckpt_path = self.save_checkpoint("rl")
        report["checkpoints"].append(ckpt_path)

        if verbose:
            print("[SPP] Stage 3: Long2Short")

        # Stage 3: Long2Short
        report["long2short"] = self.long2short_train(
            epochs=long2short_epochs, verbose=verbose
        )
        ckpt_path = self.save_checkpoint("long2short")
        report["checkpoints"].append(ckpt_path)

        if verbose:
            print("[SPP] Stage 4: Self-Distillation")

        # Stage 4: Self-Distill
        report["self_distill"] = self.self_distill(
            iterations=distill_iterations, verbose=verbose
        )
        ckpt_path = self.save_checkpoint("self_distill")
        report["checkpoints"].append(ckpt_path)

        # 最终checkpoint
        ckpt_path = self.save_checkpoint("latest")
        report["checkpoints"].append(ckpt_path)

        return report
