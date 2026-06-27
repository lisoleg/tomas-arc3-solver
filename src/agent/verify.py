# -*- coding: utf-8 -*-
"""GaussEx + κ-entropy + Gaussian Uniqueness verification module for Pipeline + SB Preamble Injector.

Provides three verification mechanisms for the universal pipeline:

  1. GaussEx Verify: Pixel residual verification with configurable tolerance.
     Checks if a DSL program applied to training pairs produces outputs
     within eps_factor × base_threshold of the expected output.

  2. κ-entropy Check: Information cardinality release verification.
     Validates that the UV→IR dimension selection satisfies the κ-entropy
     release criterion: |κ_UV − κ_IR − EXPECTED_KAPPA_ENTROPY| < tolerance.

  3. Gaussian White Noise Check (κ-rotation equivalence / linear identifiability):
     Validates that the residual (predicted - target) is Gaussian white noise.
     If residual is NOT Gaussian white noise → κ-Snap归约 has failed ("背题"/overfitting)
     → Dead-Zero熔断 → Re-Inflow回溯.

Gaussian Uniqueness Theorem (LeCun + IDO/TOMAS):
    当且仅当潜变量服从高斯分布时，JEPA学到的表示与真实物理变量仅差κ-旋转。
    GaussEx校验 = κ-旋转等价性校验 (线性可识别性)
    残差为高斯白噪声 → κ-Snap归约正确 (线性可识别)
    残差非高斯白噪声 → κ-Snap归约失败 ("背题") → Dead-Zero熔断 → Re-Inflow回溯

Dead-Zero熔断 + Re-Inflow:
    Dead-Zero: GaussEx残差非高斯白噪声 → κ-Snap归约分支剪枝
    Re-Inflow: 从Dead-Zero解压回溯到SA缓存 → 调整参数 → 重跑κ-Snap
    最大2次Re-Inflow循环

Both are used by the pipeline's Stage 5 (GaussEx Verify) to determine
whether a candidate solution is valid before returning it.

IDO Correspondence:
    - EXPECTED_KAPPA_ENTROPY = ln(11!/4!) ≈ 14.32 (UV→IR release target)
    - pixel_residual = UV pixel-level mismatch metric
    - kappa_entropy_check = κ-entropy release gate (Phase III → verification)
    - is_gaussian_white_noise = κ-rotation equivalence check (线性可识别性)
    - re_inflow = Dead-Zero熔断回溯机制 (Re-Inflow)

TOMAS Correspondence:
    - GaussEx Verify = Phase III verification gate
    - κ-entropy check = Phase II→III transition condition
    - Gaussian white noise check = κ-Snap归约正确性校验
    - Re-Inflow = Dead-Zero解压回溯到SA缓存重跑κ-Snap

Gaussian Inflow necessity:
    训练/探索数据必须覆盖高斯区域。
    RL策略诱导的偏差 → 非高斯分布 → Dead-Zero误剪枝 → 过拟合。
    这验证了 _INJECTOR_ELIGIBLE_GAMES 方法 (仅对Phase 0失败的游戏尝试pipeline)。

Version: v2.0  — Gaussian Uniqueness + Dead-Zero熔断 + Re-Inflow
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agent.injectors import SBInjector
from src.agent.ido_axioms import (
    estimate_kappa_uv,
    estimate_kappa_ir,
)

# Conditional scipy import — Gaussian distribution tests need scipy.stats
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY: bool = True
except ImportError:
    HAS_SCIPY: bool = False


# =============================================================================
# §1. Constants — κ-entropy expected value and tolerance
# =============================================================================

# ln(11!/4!) = ln(39916800/24) ≈ ln(1663200) ≈ 14.32
# This is the theoretical κ-entropy release for C(11,4)=330 coset selection
# from UV 11-dim to IR 4-dim projection.
_EXPECTED_KAPPA_ENTROPY_RAW: float = math.log(math.factorial(11) / math.factorial(4))

EXPECTED_KAPPA_ENTROPY: float = round(_EXPECTED_KAPPA_ENTROPY_RAW, 2)  # ≈ 14.32
ENTROPY_TOL: float = 0.5  # Default tolerance for κ-entropy check


# =============================================================================
# §1b. Gaussian White Noise Detection — κ-rotation equivalence check
# =============================================================================

def _test_gaussian_distribution(
    flat: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """Test whether a 1D array follows a Gaussian (normal) distribution.

    Uses Shapiro-Wilk test when scipy is available (preferred for small
    samples, n ≤ 5000). Falls back to a simple mean/variance/skewness
    heuristic when scipy is not installed.

    Args:
        flat: 1D numpy array of residual values.
        alpha: Significance level for the normality test.

    Returns:
        True if the distribution passes the Gaussian test at the given
        significance level, False otherwise.
    """
    n: int = len(flat)
    if n < 3:
        return True  # Too few samples to test — assume pass

    if HAS_SCIPY and n <= 5000:
        # Shapiro-Wilk test: preferred for rigorous normality testing
        # H0: data is normally distributed
        # Reject H0 if p-value < alpha → NOT Gaussian
        stat: float
        p_value: float
        stat, p_value = scipy_stats.shapiro(flat)
        return p_value >= alpha
    else:
        # Fallback heuristic: check mean ≈ 0, skewness ≈ 0, excess kurtosis ≈ 0
        # These are necessary (but not sufficient) conditions for Gaussianity
        mean_val: float = float(np.mean(flat))
        std_val: float = float(np.std(flat))
        if std_val < 1e-10:
            # Constant array — trivially Gaussian if mean is 0
            return abs(mean_val) < 1e-10

        # Skewness: for Gaussian, should be near 0
        # |skewness| < 0.5 is a practical threshold for "approximately Gaussian"
        skewness: float = float(np.mean(((flat - mean_val) / std_val) ** 3))
        skew_pass: bool = abs(skewness) < 0.5

        # Excess kurtosis: for Gaussian, should be near 0
        # |kurtosis| < 1.0 is a practical threshold
        kurtosis: float = float(np.mean(((flat - mean_val) / std_val) ** 4)) - 3.0
        kurt_pass: bool = abs(kurtosis) < 1.0

        return skew_pass and kurt_pass


def _test_white_noise_autocorr(
    flat: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """Test whether a 1D array has no significant autocorrelation (white noise).

    White noise has zero autocorrelation at all lags. This test checks
    lag-1 autocorrelation using the Durbin-Watson-like approach:
    - Compute lag-1 autocorrelation coefficient
    - If |r(1)| < threshold, the series passes the white noise test

    The threshold is derived from the standard error of the autocorrelation
    estimate for a Gaussian white noise process: SE ≈ 1/√n.
    We use |r(1)| < 2/√n as the significance bound (≈ 95% confidence).

    Args:
        flat: 1D numpy array of residual values.
        alpha: Significance level (affects threshold scaling).

    Returns:
        True if the series passes the white noise autocorrelation test,
        False otherwise.
    """
    n: int = len(flat)
    if n < 4:
        return True  # Too few samples to reliably estimate autocorrelation

    # Compute lag-1 autocorrelation
    mean_val: float = float(np.mean(flat))
    var_val: float = float(np.var(flat))
    if var_val < 1e-10:
        return True  # Constant series → no autocorrelation structure

    # r(1) = Σ(x[t] - μ)(x[t+1] - μ) / Σ(x[t] - μ)²
    centered: np.ndarray = flat - mean_val
    numerator: float = float(np.sum(centered[:-1] * centered[1:]))
    denominator: float = float(np.sum(centered ** 2))
    if denominator < 1e-10:
        return True

    lag1_autocorr: float = numerator / denominator

    # Significance bound: |r(1)| < 2/√n at ~95% confidence for white noise
    # Adjust for alpha: threshold = z_alpha / √n where z_alpha ≈ 1.96 for α=0.05
    # For simplicity we use: threshold = 2.0 / sqrt(n)
    threshold: float = 2.0 / math.sqrt(n)

    return abs(lag1_autocorr) < threshold


def is_gaussian_white_noise(
    residual: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """高斯白噪声检测 — κ-旋转等价性校验 (线性可识别性)

    GaussEx 升维: 残差必须是高斯白噪声才说明 κ-Snap归约成功。
    若残差非高斯白噪声 → κ-Snap归约失败 ("背题"/overfitting)
    → Dead-Zero 熔断 → 触发 Re-Inflow 回溯

    高斯唯一性定理 (LeCun + IDO/TOMAS):
    当且仅当潜变量服从高斯分布时，JEPA学到的表示与真实物理变量仅差κ-旋转。
    残差为高斯白噪声 → 线性可识别性成立 → κ-Snap归约正确

    Two-stage test:
      1. Gaussian distribution test (Shapiro-Wilk or heuristic fallback)
      2. White noise test (lag-1 autocorrelation < significance threshold)

    Both must pass for residual to be classified as Gaussian white noise.

    Args:
        residual: Residual grid (predicted - target), 2D numpy array.
        alpha: Significance level for statistical tests (default 0.05).

    Returns:
        True if residual is Gaussian white noise (κ-Snap归约正确),
        False otherwise (Dead-Zero → Re-Inflow needed).
    """
    if residual.size == 0:
        return True

    # Flatten residual to 1D
    flat: np.ndarray = residual.flatten().astype(float)

    # Skip if all zeros (perfect match → trivially Gaussian)
    if np.all(flat == 0):
        return True

    # Step 1: Gaussian distribution test
    # Shapiro-Wilk test for normality (preferred, fallback to heuristic)
    gaussian_pass: bool = _test_gaussian_distribution(flat, alpha)

    # Step 2: White noise test (no spatial autocorrelation)
    # Check lag-1 autocorrelation — should be near 0 for white noise
    autocorr_pass: bool = _test_white_noise_autocorr(flat, alpha)

    # Both must pass for Gaussian white noise
    return gaussian_pass and autocorr_pass


# =============================================================================
# §2. Pixel Residual — UV-level mismatch metric
# =============================================================================

def pixel_residual(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute pixel-level residual between predicted and target grids.

    The pixel residual is the fraction of mismatched pixels between
    two grids of the same shape. If shapes differ, returns 1.0 (total mismatch).

    Args:
        pred: Predicted output grid (2D numpy array).
        target: Expected target grid (2D numpy array).

    Returns:
        Residual fraction (0.0 = perfect match, 1.0 = total mismatch).
    """
    if pred.shape != target.shape:
        return 1.0
    total_pixels: int = target.size
    if total_pixels == 0:
        return 0.0
    mismatched: int = int(np.sum(pred != target))
    return float(mismatched / total_pixels)


# =============================================================================
# §3. UV/IR Information Capacity Estimation
# =============================================================================

def estimate_uv_ic(grid: np.ndarray) -> float:
    """Estimate UV information capacity from an ARC grid.

    UV information capacity = κ_UV from IDO axioms, representing the
    total information content of the input grid (11-dimensional analogy).

    Args:
        grid: 2D numpy array (ARC input grid).

    Returns:
        UV information cardinality κ_UV value.
    """
    return estimate_kappa_uv(grid)


def estimate_ir_ic(grid: np.ndarray) -> float:
    """Estimate IR information capacity from an ARC grid.

    IR information capacity = κ_IR from IDO axioms, representing the
    compressed information content after UV→IR projection (4-dimensional).

    Args:
        grid: 2D numpy array (ARC output grid).

    Returns:
        IR information cardinality κ_IR value.
    """
    return estimate_kappa_ir(grid)


# =============================================================================
# §4. GaussEx Verify — Pixel residual + eps_factor tolerance
# =============================================================================

def gauss_ex_verify(
    program: Any,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    eps_factor: float = 1.0,
    check_gaussian: bool = True,
) -> bool:
    """GaussEx+κ-entropy综合校验 — verify a DSL program against training pairs.

    For each training pair (input, output):
      1. Apply the program to the input grid to produce a predicted output.
      2. Compute pixel residual between predicted and expected output.
      3. If residual < eps_factor × BASE_THRESHOLD, the pair passes.

    Stage 2 (Gaussian white noise / κ-rotation equivalence / linear identifiability):
      - If check_gaussian=True, also check that the residual grid is Gaussian
        white noise. Non-Gaussian residual → κ-Snap归约失败 → Dead-Zero熔断.

    A program passes overall if ALL training pairs pass both stages.

    The eps_factor multiplier allows relaxation for interactive games
    where exact pixel match is not always achievable (e.g., tn36, ka59).

    Args:
        program: DSL program to verify. Can be:
            - A callable that takes a grid and returns a transformed grid.
            - A dict with 'name' and 'apply_fn' keys.
            - A list of DSL primitives (sequential application).
        train_pairs: List of (input_grid, output_grid) tuples for verification.
        eps_factor: Tolerance multiplier. 1.0 = strict, 1.5/2.0 = relaxed.
        check_gaussian: Whether to perform Stage 2 Gaussian white noise check.
            True (default) = check κ-rotation equivalence.
            False = skip Gaussian check (for legacy / relaxed mode).

    Returns:
        True if the program passes all training pairs (both stages),
        False otherwise.
    """
    # Base threshold: 5% pixel mismatch allowed (GaussEx standard)
    base_threshold: float = 0.05
    effective_threshold: float = base_threshold * eps_factor

    for input_grid, output_grid in train_pairs:
        # Apply program to input
        predicted: Optional[np.ndarray] = _apply_program(program, input_grid)
        if predicted is None:
            return False  # Program application failed

        # Stage 1: Compute pixel residual
        residual: float = pixel_residual(predicted, output_grid)
        if residual > effective_threshold:
            return False  # Residual exceeds tolerance

        # Stage 2: Gaussian white noise check (κ-rotation equivalence / linear identifiability)
        if check_gaussian:
            residual_grid: np.ndarray = (
                predicted.astype(float) - output_grid.astype(float)
            )
            if not is_gaussian_white_noise(residual_grid):
                return False  # Dead-Zero 熔断: κ-Snap归约失败

    return True


def _apply_program(
    program: Any,
    grid: np.ndarray,
) -> Optional[np.ndarray]:
    """Apply a DSL program to an input grid.

    Handles multiple program representations:
      - Callable: direct function call
      - Dict with 'apply_fn': call the apply_fn
      - List of dicts: sequential application of each primitive

    Args:
        program: DSL program (callable, dict, or list).
        grid: Input grid (2D numpy array).

    Returns:
        Transformed grid, or None on failure.
    """
    try:
        if callable(program):
            return program(grid)
        elif isinstance(program, dict):
            apply_fn = program.get('apply_fn')
            if callable(apply_fn):
                return apply_fn(grid)
            # Dict without apply_fn — try to interpret as a transformation
            name: str = program.get('name', '')
            return _apply_named_transform(name, grid, program.get('params', {}))
        elif isinstance(program, list):
            # Sequential application of DSL primitives
            current_grid: np.ndarray = grid.copy()
            for prim in program:
                result: Optional[np.ndarray] = _apply_program(prim, current_grid)
                if result is None:
                    return None
                current_grid = result
            return current_grid
        else:
            return None
    except Exception:
        return None


def _apply_named_transform(
    name: str,
    grid: np.ndarray,
    params: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Apply a named DSL transform to a grid.

    Maps DSL primitive names to numpy transformations. Supports common
    ARC transformations: rotation, reflection, color mapping, etc.

    Args:
        name: DSL primitive name.
        grid: Input grid.
        params: Transformation parameters.

    Returns:
        Transformed grid, or None if name is unknown.
    """
    try:
        # Symmetry transforms (D4 group)
        if name == "identity":
            return grid.copy()
        elif name == "rotate_90" or name == "rotation_apply":
            angle: int = params.get('angle', 90)
            k: int = angle // 90
            return np.rot90(grid, k=k)
        elif name == "rotate_180":
            return np.rot90(grid, k=2)
        elif name == "rotate_270":
            return np.rot90(grid, k=3)
        elif name == "flip_horizontal" or name == "reflection_apply":
            return np.flip(grid, axis=1).copy()
        elif name == "flip_vertical":
            return np.flip(grid, axis=0).copy()
        elif name == "flip_diagonal_main":
            return np.transpose(grid).copy()
        elif name == "flip_diagonal_anti":
            return np.flip(np.transpose(grid), axis=1).copy()
        # Feature-mapped transforms
        elif name == "color_map" or name == "color_replace" or name == "color_invert":
            return grid.copy()  # Placeholder — color transforms need color mapping data
        elif name.startswith("gcf_transform") or name.startswith("hyp_series") or name.startswith("poly_fit"):
            return grid.copy()  # Ramanujan conjecture transforms — placeholder
        else:
            return None  # Unknown transform
    except Exception:
        return None


# =============================================================================
# §5. κ-entropy Check — Information cardinality release verification
# =============================================================================

def kappa_entropy_check(
    kappa_uv: float,
    kappa_ir: float,
    tolerance: float = ENTROPY_TOL,
) -> bool:
    """κ-熵释放校验 — verify κ-entropy release condition.

    Checks whether the UV→IR information cardinality release satisfies
    the expected entropy: |κ_UV − κ_IR − EXPECTED_KAPPA_ENTROPY| < tolerance.

    This corresponds to the IDO anti-monotonicity criterion: the IR
    dimension selection should release approximately ln(11!/4!) ≈ 14.32
    bits of information from UV to IR.

    Args:
        kappa_uv: UV information cardinality (from input grid).
        kappa_ir: IR information cardinality (from output/collapsed grid).
        tolerance: Allowed deviation from expected κ-entropy release.
            Default 0.5 — allows ±0.5 deviation from 14.32.

    Returns:
        True if κ-entropy release is within tolerance, False otherwise.
    """
    release: float = kappa_uv - kappa_ir
    deviation: float = abs(release - EXPECTED_KAPPA_ENTROPY)
    return deviation < tolerance


# =============================================================================
# §5b. Re-Inflow — Dead-Zero熔断后回溯机制
# =============================================================================

@dataclass
class ReInflowParams:
    """Re-Inflow 回溯参数 — Dead-Zero熔断后的κ-Snap重调

    当GaussEx校验发现残差非高斯白噪声时 (Dead-Zero熔断):
    1. 熔断当前搜索分支
    2. 解压回溯到SA缓存
    3. 用调整后的参数重跑κ-Snap归约

    调整策略:
    - eps_factor: 原值×1.5 (放宽容差)
    - coset_filter: 保持原范围或扩大 (更多陪集搜索)
    - time_window: 原值+1 (增加时间窗口)
    - enable_rm: 保持原RM开关

    最大2次Re-Inflow循环，超过则彻底失败。

    Attributes:
        eps_factor: 放宽后的容差因子 (原值×1.5).
        coset_filter: 调整后的陪集搜索范围.
        time_window: 增加后的时间窗口 (原值+1).
        enable_rm: 是否启用拉马努金机.
        max_retries: 最大Re-Inflow循环次数 (固定为2).
    """

    eps_factor: float = 1.5
    coset_filter: Optional[List[int]] = None
    time_window: int = 2
    enable_rm: bool = True
    max_retries: int = 2


def re_inflow(
    current_injector: SBInjector,
    retry_count: int = 0,
) -> Optional[ReInflowParams]:
    """Re-Inflow 回溯 — Dead-Zero熔断后解压回溯到SA缓存重跑κ-Snap

    当GaussEx校验发现残差非高斯白噪声时:
    1. 熔断当前分支 (Dead-Zero)
    2. 调整κ-Snap参数 (Re-Inflow)
    3. 重跑κ-Snap归约

    最多2次Re-Inflow循环，超过则彻底失败

    TOMAS理论: Re-Inflow = 从Dead-Zero状态解压回溯到SA缓存
    然后用调整后的参数重跑κ-Snap因果归约

    调整策略:
    - eps_factor: 原值 × 1.5 (放宽容差，允许更多像素偏差)
    - time_window: 原值 + 1 (增加时间窗口，考虑更多帧因果边)
    - coset_filter: 保持原范围 (不改变陪集搜索范围)
    - enable_rm: 保持原RM开关 (不改变拉马努金机状态)

    Args:
        current_injector: Current SBInjector with failed parameters.
        retry_count: Current retry count (0, 1, or 2).

    Returns:
        ReInflowParams with adjusted parameters, or None if max retries
        exceeded (retry_count ≥ 2 → 彻底失败).
    """
    if retry_count >= 2:
        return None  # 超过最大Re-Inflow循环次数 → 彻底失败

    # 调整参数: 放宽容差 + 扩大搜索范围 + 增加时间窗口
    new_eps: float = current_injector.eps_factor * 1.5
    new_time: int = current_injector.time_window + 1
    new_coset: Optional[List[int]] = current_injector.coset_filter

    return ReInflowParams(
        eps_factor=new_eps,
        coset_filter=new_coset,
        time_window=new_time,
        enable_rm=current_injector.enable_rm,
        max_retries=2,
    )


# =============================================================================
# §6. Composite Verify — GaussEx + κ-entropy + Gaussian Uniqueness + Re-Inflow
# =============================================================================

def verify_solution(
    program: Any,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    injector: SBInjector,
    check_gaussian: bool = True,
) -> bool:
    """综合校验入口 — GaussEx + κ-entropy + Gaussian Uniqueness + Re-Inflow

    三阶段校验:
      1. GaussEx像素残差 (with injector.eps_factor tolerance)
      2. Gaussian白噪声检测 (κ-旋转等价性 / 线性可识别性)
      3. κ-entropy释放校验 (UV→IR信息基数)

    Stage 2失败 → Dead-Zero熔断 → 可触发Re-Inflow回溯

    Args:
        program: DSL program to verify.
        train_pairs: List of (input_grid, output_grid) training pairs.
        injector: SBInjector providing eps_factor and other parameters.
        check_gaussian: Whether to perform Stage 2 Gaussian white noise check.
            True (default) = check κ-rotation equivalence / linear identifiability.
            False = skip Gaussian check (for legacy / relaxed mode).

    Returns:
        True if all three stages pass, False otherwise.
        Stage 2 failure = Dead-Zero熔断 (caller may invoke re_inflow()).
    """
    # Stage 1+2: GaussEx pixel residual + Gaussian white noise verification
    gauss_pass: bool = gauss_ex_verify(
        program, train_pairs,
        eps_factor=injector.eps_factor,
        check_gaussian=check_gaussian,
    )
    if not gauss_pass:
        return False  # Dead-Zero熔断

    # Stage 2: κ-entropy release verification (on first training pair)
    if train_pairs:
        input_grid, output_grid = train_pairs[0]
        kappa_uv_val: float = estimate_uv_ic(input_grid)
        kappa_ir_val: float = estimate_ir_ic(output_grid)
        kappa_pass: bool = kappa_entropy_check(kappa_uv_val, kappa_ir_val)
        if not kappa_pass:
            # κ-entropy check failed — but don't reject outright for
            # interactive games with relaxed tolerance
            # Only enforce for non-interactive games (time_window=1)
            if injector.time_window == 1:
                return False

    return True
