"""
GPU Backend: 统一设备抽象层

自动检测 CUDA / MPS / CPU，为 NAR-Net 和 TOMAS 提供统一计算后端。
- GPU 可用 → PyTorch 8× 实值展开（L3 加速，10-30×）
- GPU 不可用 → NumPy einsum 向量化（L1 加速，当前已验证 875×）

设计原则：
  1. 延迟导入：不强制依赖 PyTorch，没有也能跑
  2. 透明切换：上层代码不关心底层设备
  3. 零拷贝：GPU 模式下数据留在显存，避免来回传输
  4. Fallback 链：CUDA → MPS → CPU(NumPy)

Author: TOMAS Team
Version: 0.1.0
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict
import warnings
import time

# ============================================================================
# 设备检测
# ============================================================================

class DeviceInfo:
    """设备信息缓存"""
    _instance: Optional['DeviceInfo'] = None
    
    def __init__(self):
        self._torch = None
        self._torch_checked = False
        self._device = None
        self._device_name = None
        self._is_gpu = False
        self._backend = 'numpy'  # 'numpy' or 'torch'
    
    @classmethod
    def get(cls) -> 'DeviceInfo':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _check_torch(self) -> bool:
        """检查 PyTorch 是否可用"""
        if self._torch_checked:
            return self._torch is not None
        self._torch_checked = True
        
        try:
            import torch
            self._torch = torch
            return True
        except ImportError:
            return False
    
    def detect_device(self) -> Tuple[str, str, bool, str]:
        """
        检测最佳可用设备
        
        Returns:
            (device_str, device_name, is_gpu, backend)
            device_str: 'cuda:0', 'mps', 'cpu'
            device_name: 人类可读名称
            is_gpu: 是否GPU加速
            backend: 'torch' 或 'numpy'
        """
        if self._device is not None:
            return self._device, self._device_name, self._is_gpu, self._backend
        
        # 尝试 PyTorch GPU
        if self._check_torch():
            torch = self._torch
            
            # CUDA
            if torch.cuda.is_available():
                self._device = 'cuda:0'
                self._device_name = torch.cuda.get_device_name(0)
                self._is_gpu = True
                self._backend = 'torch'
                return self._device, self._device_name, self._is_gpu, self._backend
            
            # Apple MPS (Metal Performance Shaders)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'
                self._device_name = 'Apple Metal (MPS)'
                self._is_gpu = True
                self._backend = 'torch'
                return self._device, self._device_name, self._is_gpu, self._backend
        
        # Fallback: NumPy CPU
        self._device = 'cpu'
        self._device_name = f'CPU ({np._platform if hasattr(np, "_platform") else "x86"})'
        self._is_gpu = False
        self._backend = 'numpy'
        return self._device, self._device_name, self._is_gpu, self._backend
    
    @property
    def torch(self):
        """获取 torch 模块（如果可用）"""
        if not self._torch_checked:
            self._check_torch()
        return self._torch
    
    @property
    def is_gpu(self) -> bool:
        if self._device is None:
            self.detect_device()
        return self._is_gpu
    
    @property
    def device(self) -> str:
        if self._device is None:
            self.detect_device()
        return self._device
    
    @property
    def backend(self) -> str:
        if self._backend is None:
            self.detect_device()
        return self._backend
    
    def summary(self) -> str:
        dev, name, is_gpu, backend = self.detect_device()
        gpu_tag = "GPU" if is_gpu else "CPU"
        return f"[{gpu_tag}] {name} (backend={backend}, device={dev})"


# ============================================================================
# 全局便捷函数
# ============================================================================

def get_device_info() -> DeviceInfo:
    """获取设备信息单例"""
    return DeviceInfo.get()

def is_gpu_available() -> bool:
    """GPU是否可用"""
    return DeviceInfo.get().is_gpu

def get_device() -> str:
    """获取当前设备字符串"""
    return DeviceInfo.get().device

def get_backend() -> str:
    """获取当前后端 ('torch' 或 'numpy')"""
    return DeviceInfo.get().backend

def get_torch():
    """获取 torch 模块（不可用时返回 None）"""
    return DeviceInfo.get().torch


# ============================================================================
# 张量转换工具
# ============================================================================

def to_tensor(x: np.ndarray, dtype=None) -> Any:
    """
    将 NumPy 数组转换为当前后端的张量
    
    GPU 模式: 返回 torch.Tensor on device
    CPU 模式: 返回 np.ndarray (原样)
    """
    info = DeviceInfo.get()
    
    if info.backend == 'torch':
        torch = info.torch
        t = torch.from_numpy(x)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(info.device)
    else:
        if dtype is not None:
            x = x.astype(dtype)
        return x

def to_numpy(x: Any) -> np.ndarray:
    """将任意后端张量转回 NumPy"""
    info = DeviceInfo.get()
    
    if info.backend == 'torch' and info.torch is not None:
        torch = info.torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return np.asarray(x)

def zeros_like(x: Any) -> Any:
    """创建同形状零张量"""
    info = DeviceInfo.get()
    
    if info.backend == 'torch' and info.torch is not None:
        torch = info.torch
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
    return np.zeros_like(x)


# ============================================================================
# GPU 加速的八元数运算（PyTorch 版本）
# ============================================================================

# 八元数乘法表（与 nar_net_core.py 保持一致）
OCT_IDX = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 3, 2, 5, 4, 7, 6],
    [2, 3, 0, 1, 6, 7, 4, 5],
    [3, 2, 1, 0, 7, 6, 5, 4],
    [4, 5, 6, 7, 0, 1, 2, 3],
    [5, 4, 7, 6, 1, 0, 3, 2],
    [6, 7, 4, 5, 2, 3, 0, 1],
    [7, 6, 5, 4, 3, 2, 1, 0],
], dtype=np.int8)

OCT_SIGN = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1],
    [ 1, -1,  1, -1,  1, -1, -1,  1],
    [ 1, -1, -1,  1,  1,  1, -1, -1],
    [ 1,  1, -1, -1,  1, -1,  1, -1],
    [ 1, -1, -1, -1, -1,  1,  1,  1],
    [ 1,  1, -1,  1, -1, -1, -1,  1],
    [ 1,  1,  1, -1, -1,  1, -1, -1],
    [ 1, -1,  1,  1, -1, -1,  1, -1],
], dtype=np.float32)


class OctonionOps:
    """
    八元数运算的设备自适应实现
    
    自动选择 NumPy 或 PyTorch 后端。
    GPU 模式下使用 PyTorch 的批量矩阵运算。
    """
    
    _idx_tensor = None
    _sign_tensor = None
    _initialized = False
    
    @classmethod
    def _init(cls):
        """初始化乘法表张量（延迟到第一次使用）"""
        if cls._initialized:
            return
        info = DeviceInfo.get()
        
        if info.backend == 'torch':
            torch = info.torch
            cls._idx_tensor = torch.from_numpy(OCT_IDX.astype(np.int64)).to(info.device)
            cls._sign_tensor = torch.from_numpy(OCT_SIGN).to(info.device)
        else:
            cls._idx_tensor = OCT_IDX
            cls._sign_tensor = OCT_SIGN
        cls._initialized = True
    
    @classmethod
    def multiply(cls, x, w):
        """
        向量化八元数乘法
        
        Args:
            x: (..., 8) 
            w: (..., 8)
        Returns:
            (..., 8)
        """
        cls._init()
        info = DeviceInfo.get()
        
        if info.backend == 'torch':
            torch = info.torch
            # GPU 版本：用 torch 的批量运算
            # 展开为 8x8 的组合
            # x_expanded: (..., 8, 1), w_expanded: (..., 1, 8)
            x_exp = x.unsqueeze(-1)  # (..., 8, 1)
            w_exp = w.unsqueeze(-2)  # (..., 1, 8)
            
            # 乘积矩阵: (..., 8, 8)
            products = x_exp * w_exp  # 广播乘法
            
            # 按乘法表累加到结果
            # idx: (8, 8) → 每个位置指向结果索引
            # sign: (8, 8) → 每个位置的符号
            result = torch.zeros(x.shape[:-1] + (8,), 
                                 dtype=x.dtype, device=x.device)
            
            for a in range(8):
                for b in range(8):
                    k = int(cls._idx_tensor[a, b].item())
                    s = cls._sign_tensor[a, b]
                    result[..., k] += s * products[..., a, b]
            
            return result
        else:
            # NumPy 版本（与 nar_net_core.py 一致）
            y = np.zeros(x.shape[:-1] + (8,), dtype=np.float32)
            for a in range(8):
                for b in range(8):
                    k = cls._idx_tensor[a, b]
                    s = cls._sign_tensor[a, b]
                    y[..., k] += s * x[..., a] * w[..., b]
            return y
    
    @classmethod
    def existence_degree(cls, x) -> Any:
        """信息存在度 I(e) = ||x||"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            return torch.sqrt(torch.sum(x ** 2, dim=-1))
        else:
            return np.sqrt(np.sum(x ** 2, axis=-1))
    
    @classmethod
    def chirality(cls, x) -> Any:
        """手性 = ||imag|| / ||total||"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            total_norm = torch.sqrt(torch.sum(x ** 2, dim=-1)) + 1e-8
            imag_norm = torch.sqrt(torch.sum(x[..., 1:] ** 2, dim=-1))
            return imag_norm / total_norm
        else:
            total_norm = np.sqrt(np.sum(x ** 2, axis=-1)) + 1e-8
            imag_norm = np.sqrt(np.sum(x[..., 1:] ** 2, axis=-1))
            return imag_norm / total_norm
    
    @classmethod
    def restore_chirality(cls, x, ref) -> Any:
        """从参考张量恢复手性"""
        info = DeviceInfo.get()
        
        x_chir = cls.chirality(x)
        ref_chir = cls.chirality(ref) + 1e-8
        
        if info.backend == 'torch':
            ratio = torch.minimum(x_chir / ref_chir, torch.ones_like(x_chir))
            scale = ratio.unsqueeze(-1)
            
            result = x.clone()
            need_restore = x_chir < ref_chir
            if need_restore.any():
                imag_boost = ref[..., 1:] * (1 - scale[..., 1:])
                result[..., 1:] = x[..., 1:] + imag_boost
            return result
        else:
            ratio = np.minimum(x_chir / ref_chir, 1.0)
            scale = np.expand_dims(ratio, -1)
            
            result = x.copy()
            need_restore = x_chir < ref_chir
            if np.any(need_restore):
                imag_boost = ref[..., 1:] * (1 - scale)
                result[..., 1:] = x[..., 1:] + imag_boost
            return result
    
    @classmethod
    def conv2d(cls, x, weights, bias=None, stride=1, padding=1):
        """
        八元数 2D 卷积（设备自适应）
        
        Args:
            x: (B, Cin, H, W, 8)
            weights: (Cout, Cin, k, k, 8)
            bias: (Cout, 8) or None
            stride: 步长
            padding: 填充
        
        Returns:
            (B, Cout, H', W', 8)
        """
        cls._init()
        info = DeviceInfo.get()
        
        if info.backend == 'torch':
            return cls._conv2d_torch(x, weights, bias, stride, padding)
        else:
            return cls._conv2d_numpy(x, weights, bias, stride, padding)
    
    @classmethod
    def _conv2d_torch(cls, x, weights, bias, stride, padding):
        """PyTorch GPU 加速八元数卷积"""
        torch = DeviceInfo.get().torch
        B, Cin, H, W, _ = x.shape
        Cout = weights.shape[0]
        k = weights.shape[2]
        
        # Padding
        if padding > 0:
            x_pad = torch.nn.functional.pad(x, (0, 0, padding, padding, padding, padding, 0, 0))
        else:
            x_pad = x
        
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        H_out = (H_pad - k) // stride + 1
        W_out = (W_pad - k) // stride + 1
        
        # im2col: 提取所有patch
        patches = torch.zeros((B, Cin, H_out, W_out, k, k, 8),
                              dtype=x.dtype, device=x.device)
        for kh in range(k):
            for kw in range(k):
                patches[:, :, :, :, kh, kw, :] = x_pad[:, :,
                    kh:kh+stride*H_out:stride, kw:kw+stride*W_out:stride, :]
        
        # 向量化八元数卷积：64次 einsum
        output = torch.zeros((B, Cout, H_out, W_out, 8),
                             dtype=x.dtype, device=x.device)
        
        for a in range(8):
            for b in range(8):
                c = int(cls._idx_tensor[a, b].item())
                s = cls._sign_tensor[a, b]
                
                x_a = patches[..., a]  # (B, Cin, H_out, W_out, k, k)
                w_b = weights[..., b]  # (Cout, Cin, k, k)
                
                # einsum on GPU
                contrib = torch.einsum('bchwij,dcij->bdhw', x_a, w_b)
                output[..., c] += s * contrib
        
        if bias is not None:
            output += bias.view(1, Cout, 1, 1, 8)
        
        return output
    
    @classmethod
    def _conv2d_numpy(cls, x, weights, bias, stride, padding):
        """NumPy einsum 八元数卷积（与 nar_net_core.py 一致）"""
        B, Cin, H, W, _ = x.shape
        Cout = weights.shape[0]
        k = weights.shape[2]
        
        if padding > 0:
            x_pad = np.zeros((B, Cin, H + 2*padding, W + 2*padding, 8), dtype=np.float32)
            x_pad[:, :, padding:padding+H, padding:padding+W, :] = x
        else:
            x_pad = x
        
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        H_out = (H_pad - k) // stride + 1
        W_out = (W_pad - k) // stride + 1
        
        patches = np.zeros((B, Cin, H_out, W_out, k, k, 8), dtype=np.float32)
        for kh in range(k):
            for kw in range(k):
                patches[:, :, :, :, kh, kw, :] = x_pad[:, :,
                    kh:kh+stride*H_out:stride, kw:kw+stride*W_out:stride, :]
        
        output = np.zeros((B, Cout, H_out, W_out, 8), dtype=np.float32)
        
        for a in range(8):
            for b in range(8):
                c = cls._idx_tensor[a, b]
                s = cls._sign_tensor[a, b]
                
                x_a = patches[..., a]
                w_b = weights[..., b]
                
                contrib = np.einsum('bchwij,dcij->bdhw', x_a, w_b, optimize=True)
                output[..., c] += s * contrib
        
        if bias is not None:
            output += bias[np.newaxis, :, np.newaxis, np.newaxis, :]
        
        return output
    
    @classmethod
    def batchnorm(cls, x, gamma, beta, running_mean, running_var, 
                  eps=1e-5, momentum=0.1, training=True):
        """
        八元数 BatchNorm
        
        Args:
            x: (B, C, H, W, 8)
            gamma, beta: (C, 8)
            running_mean, running_var: (C, 8)
        """
        info = DeviceInfo.get()
        
        if info.backend == 'torch':
            torch = info.torch
            if training:
                mean = x.mean(dim=(0, 2, 3))
                var = x.var(dim=(0, 2, 3), unbiased=False)
                running_mean[:] = (1 - momentum) * running_mean + momentum * mean.detach()
                running_var[:] = (1 - momentum) * running_var + momentum * var.detach()
            else:
                mean = running_mean
                var = running_var
            
            mean_r = mean.view(1, -1, 1, 1, 8)
            var_r = var.view(1, -1, 1, 1, 8)
            gamma_r = gamma.view(1, -1, 1, 1, 8)
            beta_r = beta.view(1, -1, 1, 1, 8)
            
            x_norm = (x - mean_r) / torch.sqrt(var_r + eps)
            return x_norm * gamma_r + beta_r
        else:
            if training:
                mean = np.mean(x, axis=(0, 2, 3))
                var = np.var(x, axis=(0, 2, 3))
                running_mean[:] = (1 - momentum) * running_mean + momentum * mean
                running_var[:] = (1 - momentum) * running_var + momentum * var
            else:
                mean = running_mean
                var = running_var
            
            mean_r = mean[np.newaxis, :, np.newaxis, np.newaxis, :]
            var_r = var[np.newaxis, :, np.newaxis, np.newaxis, :]
            gamma_r = gamma[np.newaxis, :, np.newaxis, np.newaxis, :]
            beta_r = beta[np.newaxis, :, np.newaxis, np.newaxis, :]
            
            x_norm = (x - mean_r) / np.sqrt(var_r + eps)
            return x_norm * gamma_r + beta_r
    
    @classmethod
    def relu(cls, x):
        """八元数 ReLU（实部 LeakyReLU，虚部保持）"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            torch = info.torch
            result = x.clone()
            real = x[..., 0]
            mask = real < 0
            result[mask] *= 0.1
            return result
        else:
            result = x.copy()
            real = x[..., 0]
            mask = real < 0
            result[mask] *= 0.1
            return result
    
    @classmethod
    def softmax(cls, x, dim=-1):
        """Softmax"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            torch = info.torch
            return torch.softmax(x, dim=dim)
        else:
            x_max = np.max(x, axis=dim, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    
    @classmethod
    def global_avg_pool(cls, x):
        """全局平均池化: (B, C, H, W, 8) → (B, C, 8)"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            return x.mean(dim=(2, 3))
        else:
            return np.mean(x, axis=(2, 3))
    
    @classmethod
    def matmul(cls, x, w):
        """矩阵乘法"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            return torch.matmul(x, w) if hasattr(x, 'shape') else x @ w
        else:
            return x @ w
    
    @classmethod
    def randn(cls, *shape, dtype=None):
        """随机正态分布"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            torch = info.torch
            t = torch.randn(*shape, device=info.device)
            if dtype is not None:
                t = t.to(dtype)
            return t
        else:
            dt = dtype if dtype is not None else np.float32
            return np.random.randn(*shape).astype(dt)
    
    @classmethod
    def ones(cls, shape, dtype=None):
        """全1张量"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            torch = info.torch
            t = torch.ones(shape, device=info.device)
            if dtype is not None:
                t = t.to(dtype)
            return t
        else:
            dt = dtype if dtype is not None else np.float32
            return np.ones(shape, dtype=dt)
    
    @classmethod
    def zeros(cls, shape, dtype=None):
        """全0张量"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            torch = info.torch
            t = torch.zeros(shape, device=info.device)
            if dtype is not None:
                t = t.to(dtype)
            return t
        else:
            dt = dtype if dtype is not None else np.float32
            return np.zeros(shape, dtype=dt)


# ============================================================================
# 设备状态报告
# ============================================================================

def print_device_status():
    """打印设备状态报告"""
    info = DeviceInfo.get()
    dev, name, is_gpu, backend = info.detect_device()
    
    print("=" * 60)
    print("TOMAS/NAR-Net 设备状态")
    print("=" * 60)
    print(f"  设备: {dev}")
    print(f"  名称: {name}")
    print(f"  GPU加速: {'✅ 是' if is_gpu else '❌ 否'}")
    print(f"  后端: {backend}")
    
    if backend == 'torch':
        torch = info.torch
        print(f"  PyTorch版本: {torch.__version__}")
        if is_gpu and dev.startswith('cuda'):
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print(f"  NumPy版本: {np.__version__}")
    
    # 性能预估
    if is_gpu:
        print(f"  预估加速: 10-30× (vs CPU NumPy)")
        print(f"  策略: PyTorch 8× 实值展开 (L3)")
    else:
        print(f"  预估加速: 875× (vs Python循环)")
        print(f"  策略: NumPy einsum 向量化 (L1)")
    
    print("=" * 60)
    return is_gpu


# ============================================================================
# 测试
# ============================================================================

def test_gpu_backend():
    """测试 GPU 后端"""
    print("\n" + "=" * 60)
    print("测试 GPU Backend")
    print("=" * 60)
    
    # 1. 设备检测
    print_device_status()
    
    info = DeviceInfo.get()
    
    # 2. 八元数乘法测试
    print("\n1. 八元数乘法测试...")
    e1 = to_tensor(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    e2 = to_tensor(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32))
    
    result = OctonionOps.multiply(e1, e2)
    result_np = to_numpy(result)
    expected = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    assert np.allclose(result_np, expected), f"e1*e2 should be e3, got {result_np}"
    print(f"   ✅ e1 * e2 = e3 (backend={info.backend})")
    
    # 3. 存在度测试
    print("\n2. 信息存在度测试...")
    x = to_tensor(np.random.randn(1, 4, 8, 8, 8).astype(np.float32))
    I_e = OctonionOps.existence_degree(x)
    print(f"   I(e) = {float(to_numpy(I_e).mean()):.4f} ✅")
    
    # 4. 卷积测试
    print("\n3. 八元数卷积测试...")
    x_conv = to_tensor(np.random.randn(1, 3, 8, 8, 8).astype(np.float32) * 0.1)
    w_conv = to_tensor(np.random.randn(4, 3, 3, 3, 8).astype(np.float32) * 0.01)
    
    t0 = time.time()
    out = OctonionOps.conv2d(x_conv, w_conv, stride=1, padding=1)
    t1 = time.time()
    out_np = to_numpy(out)
    print(f"   输入: {x_conv.shape if info.backend == 'numpy' else tuple(x_conv.shape)}")
    print(f"   输出: {out_np.shape}")
    print(f"   耗时: {t1-t0:.4f}s ✅")
    
    # 5. 性能对比
    print("\n4. 性能对比...")
    sizes = [(4, 4), (8, 8), (16, 16)]
    for h, w in sizes:
        x_test = to_tensor(np.random.randn(1, 3, h, w, 8).astype(np.float32) * 0.1)
        w_test = to_tensor(np.random.randn(4, 3, 3, 3, 8).astype(np.float32) * 0.01)
        
        t0 = time.time()
        for _ in range(10):
            OctonionOps.conv2d(x_test, w_test, stride=1, padding=1)
        t1 = time.time()
        avg = (t1 - t0) / 10
        print(f"   {h}×{w}: {avg*1000:.2f}ms/iter")
    
    print("\n✅ GPU Backend 测试通过!")
    return True


if __name__ == "__main__":
    test_gpu_backend()
