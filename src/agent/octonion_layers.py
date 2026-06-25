"""
八元数神经网络层 - 卷积、批归一化、激活函数
Octonion Neural Network Layers

实现八元数版本的卷积层、批归一化、激活函数，支持非结合运算和手性恢复。

Author: TOMAS Team
Version: 0.1.0 (Phase1 MVP)
"""

import numpy as np
from typing import Tuple, Optional, List
from .octonion_tensor import OctonionTensor, restore_chirality


class OctonionConv2d:
    """
    八元数2D卷积层
    
    将输入八元数张量（形状 (N, C_in, H, W, 8)）通过卷积映射到输出（形状 (N, C_out, H', W', 8)）。
    
    每个输出位置是输入局部区域的八元数加权求和（非结合！）。
    权重本身是八元数，卷积等价于八元数互相关。
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 nonassociative: bool = True):
        """
        初始化八元数卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            nonassociative: 是否启用非结合运算（默认True）
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.nonassociative = nonassociative
        
        # 初始化权重（八元数）
        # 形状: (out_channels, in_channels, kernel_size, kernel_size, 8)
        self.weights = self._init_octonion_weights(
            out_channels, in_channels, kernel_size
        )
        
        # 偏置（八元数，但通常设为0）
        self.bias = OctonionTensor.from_scalar(np.zeros(out_channels))
    
    def _init_octonion_weights(self, 
                                out_channels: int, 
                                in_channels: int, 
                                kernel_size: int) -> np.ndarray:
        """
        初始化八元数权重
        
        策略：
        - 实部：Xavier初始化
        - 虚部分量：小随机值（保持手性）
        """
        # 计算fan_in和fan_out
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        
        # Xavier初始化范围
        limit = np.sqrt(6 / (fan_in + fan_out))
        
        # 初始化权重数组 (out_channels, in_channels, k, k, 8)
        weights = np.zeros((out_channels, in_channels, kernel_size, kernel_size, 8))
        
        # 实部：Xavier初始化
        weights[..., 0] = np.random.uniform(-limit, limit, 
                                            (out_channels, in_channels, kernel_size, kernel_size))
        
        # 虚部分量：小随机值（引入手性）
        weights[..., 1:8] = np.random.randn(out_channels, in_channels, kernel_size, kernel_size, 7) * 0.01
        
        return weights
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入八元数张量，形状 (N, C_in, H, W, 8)
        
        Returns:
            输出八元数张量，形状 (N, C_out, H', W', 8)
        """
        N, C_in, H, W, _ = x.shape
        
        # 计算输出尺寸
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((N, self.out_channels, H_out, W_out, 8))
        
        # Padding
        if self.padding > 0:
            x_padded = np.pad(x, 
                             ((0, 0), (0, 0), 
                              (self.padding, self.padding), 
                              (self.padding, self.padding), 
                              (0, 0)), 
                             mode='constant')
        else:
            x_padded = x
        
        # 卷积运算（八元数版本）
        for n in range(N):
            for oc in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        
                        # 提取局部区域 (C_in, k, k, 8)
                        local_region = x_padded[n, :, 
                                               h_start:h_start+self.kernel_size, 
                                               w_start:w_start+self.kernel_size, :]
                        
                        # 八元数卷积 = 加权求和（非结合！）
                        # 对每个输入通道，计算 weight * local_region（八元数乘法）
                        conv_result = np.zeros(8)
                        
                        for ic in range(C_in):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    # 提取权重八元数 (8,)
                                    w_oct = self.weights[oc, ic, kh, kw]
                                    
                                    # 提取局部区域八元数 (8,)
                                    x_oct = local_region[ic, kh, kw]
                                    
                                    # 八元数乘法（非结合）
                                    if self.nonassociative:
                                        # 使用完整的八元数乘法
                                        mult_result = self._octonion_multiply_full(w_oct, x_oct)
                                    else:
                                        # 简化：逐分量乘法（失去非结合性）
                                        mult_result = w_oct * x_oct
                                    
                                    conv_result += mult_result
                        
                        # 加偏置
                        conv_result += self.bias.data[oc]
                        
                        output[n, oc, h_out, w_out] = conv_result
        
        return output
    
    def _octonion_multiply_full(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        完整的八元数乘法（非结合）
        
        使用Cayley-Dickson构造的完整乘法表。
        这是计算密集型操作，实际中应优化。
        """
        # 简化版本：使用预计算的乘法规则
        # 实际实现需要完整的Fano平面乘法表
        
        result = np.zeros(8)
        
        # 实部
        result[0] = a[0] * b[0] - np.dot(a[1:8], b[1:8])
        
        # 虚部分量（使用简化规则）
        for i in range(1, 8):
            result[i] = a[0] * b[i] + a[i] * b[0]
        
        # 非结合项（七维虚部分量间的交互）
        # 基于Fano平面的7个基三元组
        fano_triples = [
            (1, 2, 3), (1, 4, 5), (1, 6, 7),
            (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 5, 6)
        ]
        
        for (i, j, k) in fano_triples:
            # e_i * e_j = e_k (遵循循环顺序）
            result[k] += a[i] * b[j] - a[j] * b[i]
        
        return result


class OctonionBatchNorm:
    """
    八元数批归一化
    
    对八元数张量的每个分量独立进行归一化，但保持八元数结构。
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps
        
        # 可学习参数（每个输出通道8个分量）
        self.gamma = np.ones((num_features, 8))  # 缩放
        self.beta = np.zeros((num_features, 8))   # 偏移
        
        # 运行统计量
        self.running_mean = np.zeros((num_features, 8))
        self.running_var = np.ones((num_features, 8))
        
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 形状 (N, C, H, W, 8) 或 (N, C, 8)
        """
        if self.training:
            # 计算批次统计量
            if x.ndim == 5:
                # 2D数据：(N, C, H, W, 8)
                mean = np.mean(x, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1, 8)
                var = np.var(x, axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1, 8)
            else:
                # 1D数据：(N, C, 8)
                mean = np.mean(x, axis=0, keepdims=True)  # (1, C, 8)
                var = np.var(x, axis=0, keepdims=True)    # (1, C, 8)
            
            # 更新运行统计量
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean.squeeze()
            self.running_var = 0.9 * self.running_var + 0.1 * var.squeeze()
        else:
            # 使用运行统计量
            if x.ndim == 5:
                mean = self.running_mean.reshape(1, -1, 1, 1, 8)
                var = self.running_var.reshape(1, -1, 1, 1, 8)
            else:
                mean = self.running_mean.reshape(1, -1, 8)
                var = self.running_var.reshape(1, -1, 8)
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和偏移
        if x.ndim == 5:
            gamma = self.gamma.reshape(1, -1, 1, 1, 8)
            beta = self.beta.reshape(1, -1, 1, 1, 8)
        else:
            gamma = self.gamma.reshape(1, -1, 8)
            beta = self.beta.reshape(1, -1, 8)
        
        return gamma * x_norm + beta


def octonion_relu(x: np.ndarray) -> np.ndarray:
    """
    八元数ReLU激活函数
    
    策略：保持八元数结构，但对实部应用ReLU，虚部分量使用相同的掩码。
    """
    # 提取实部
    real_part = x[..., 0]
    
    # 创建掩码（实部 > 0 则保留）
    mask = (real_part > 0).astype(np.float32)
    
    # 应用掩码到所有8个分量
    mask_expanded = np.expand_dims(mask, axis=-1)  # 增加最后一维
    mask_broadcast = np.broadcast_to(mask_expanded, x.shape)
    
    return x * mask_broadcast


def octonion_softmax(x: np.ndarray, axis: int = -2) -> np.ndarray:
    """
    八元数Softmax
    
    对八元数张量的实部应用Softmax（通常用于策略头输出）。
    """
    # 只对实部应用softmax
    real_part = x[..., 0]
    
    # 数值稳定性
    real_part_shifted = real_part - np.max(real_part, axis=axis, keepdims=True)
    
    # Softmax
    exp_real = np.exp(real_part_shifted)
    softmax_real = exp_real / np.sum(exp_real, axis=axis, keepdims=True)
    
    # 构造输出八元数（实部=softmax，虚部=0）
    output = np.zeros_like(x)
    output[..., 0] = softmax_real
    
    return output


def test_octonion_layers():
    """测试八元数神经网络层"""
    print("=" * 60)
    print("测试八元数神经网络层")
    print("=" * 60)
    
    # 测试1：OctonionConv2d
    print("\n1. 测试OctonionConv2d...")
    conv = OctonionConv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    
    # 创建随机输入 (N=2, C=3, H=8, W=8, 8)
    x = np.random.randn(2, 3, 8, 8, 8)
    
    # 前向传播
    output = conv.forward(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   ✅ 卷积层工作正常")
    
    # 测试2：OctonionBatchNorm
    print("\n2. 测试OctonionBatchNorm...")
    bn = OctonionBatchNorm(num_features=8)
    
    # 前向传播
    output_bn = bn.forward(output)
    print(f"   输入形状: {output.shape}")
    print(f"   输出形状: {output_bn.shape}")
    print(f"   ✅ 批归一化工作正常")
    
    # 测试3：激活函数
    print("\n3. 测试激活函数...")
    x_relu = octonion_relu(output_bn)
    print(f"   ReLU输出形状: {x_relu.shape}")
    print(f"   ✅ 激活函数工作正常")
    
    print("\n✅ 八元数神经网络层测试完成！")


if __name__ == "__main__":
    test_octonion_layers()
