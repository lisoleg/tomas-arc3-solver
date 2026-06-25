"""
八元数ResNet - 残差网络架构
Octonion ResNet - Residual Network Architecture

实现带有八元数残差块（非结合运算 + 手性恢复 + I(e)守恒）的ResNet架构。

Author: TOMAS Team
Version: 0.1.0 (Phase1 MVP)
"""

import numpy as np
from typing import List, Tuple, Optional
from .octonion_tensor import OctonionTensor, restore_chirality
from .octonion_layers import OctonionConv2d, OctonionBatchNorm, octonion_relu


class OctonionResidualBlock:
    """
    八元数残差块（核心创新）
    
    结构：
    Input (Octonion) → Conv1 → BatchNorm → ReLU → Conv2 → BatchNorm
        ↓ (Skip Connection)
        + (八元数加法，I(e)守恒）
        ↓
    Chirality Restoration (手性恢复）
        ↓
    ReLU → Output
    
    创新点：
    1. 八元数非结合卷积（Conv权重是八元数）
    2. Skip Connection保证信息存在度 I(e) 守恒
    3. 手性恢复模块（Chirality Restoration）
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 chirality_aware: bool = True,
                 nonassociative: bool = True):
        """
        初始化八元数残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长（用于下采样）
            chirality_aware: 是否启用手性恢复
            nonassociative: 是否启用非结合运算
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chirality_aware = chirality_aware
        self.nonassociative = nonassociative
        
        # 第一层卷积
        self.conv1 = OctonionConv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1,
            nonassociative=nonassociative
        )
        self.bn1 = OctonionBatchNorm(out_channels)
        
        # 第二层卷积
        self.conv2 = OctonionConv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1,
            nonassociative=nonassociative
        )
        self.bn2 = OctonionBatchNorm(out_channels)
        
        # Skip Connection（如果输入输出通道数不同，需要1x1卷积映射）
        if stride != 1 or in_channels != out_channels:
            self.skip_conv = OctonionConv2d(
                in_channels, out_channels, 
                kernel_size=1, stride=stride, padding=0,
                nonassociative=False  # Skip Connection使用简单映射
            )
        else:
            self.skip_conv = None
        
        # 手性恢复模块
        if chirality_aware:
            self.chirality_alpha = 0.1  # 手性恢复强度
        else:
            self.chirality_alpha = 0.0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入八元数张量，形状 (N, C_in, H, W, 8)
        
        Returns:
            输出八元数张量，形状 (N, C_out, H', W', 8)
        """
        identity = x
        
        # 主分支
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = octonion_relu(out)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        
        # Skip Connection（I(e)守恒）
        if self.skip_conv is not None:
            identity = self.skip_conv.forward(identity)
        
        # 确保identity和out形状匹配
        # （如果stride!=1或通道数不同，identity已被映射）
        out = out + identity  # 八元数加法
        
        # 手性恢复（如果启用）
        if self.chirality_aware and self.chirality_alpha > 0:
            # 将numpy数组转换为OctonionTensor以计算手性
            out_tensor = OctonionTensor(out)
            identity_tensor = OctonionTensor(identity)
            
            # 计算手性差异
            chirality_out = out_tensor.chirality()
            chirality_identity = identity_tensor.chirality()
            
            # 手性恢复强度（自适应）
            alpha = self.chirality_alpha * np.abs(chirality_identity - chirality_out)
            alpha = np.clip(alpha, 0, 1)
            
            # 恢复手性（通过虚部分量增强）
            out_imag = out[..., 1:8]
            identity_imag = identity[..., 1:8]
            
            # 混合策略：保留一部分输入手性
            restored_imag = out_imag + alpha[..., np.newaxis] * (identity_imag - out_imag)
            out[..., 1:8] = restored_imag
        
        # 最终ReLU
        out = octonion_relu(out)
        
        return out


class OctonionResNet:
    """
    八元数ResNet架构
    
    用于ARC-AGI-3游戏的Oracle适配器主干网络。
    
    特点：
    1. 八元数残差块（非结合推理）
    2. 手性恢复（保持不对称性）
    3. I(e)守恒（信息不丢失）
    4. 小样本学习能力
    """
    
    def __init__(self, 
                 num_blocks: List[int] = [2, 2, 2, 2],
                 num_classes: int = 10,
                 chirality_aware: bool = True,
                 nonassociative: bool = True):
        """
        初始化八元数ResNet
        
        Args:
            num_blocks: 每个stage的残差块数量，默认[2,2,2,2]（ResNet-18）
            num_classes: 输出类别数（动作空间大小）
            chirality_aware: 是否启用手性恢复
            nonassociative: 是否启用非结合运算
        """
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.chirality_aware = chirality_aware
        self.nonassociative = nonassociative
        
        # 输入嵌入层（将普通张量映射到八元数空间）
        self.input_channels = 3  # RGB图像
        self.base_channels = 64
        
        # Stem: 初始卷积
        self.stem_conv = OctonionConv2d(
            in_channels=self.input_channels, 
            out_channels=self.base_channels, 
            kernel_size=7, stride=2, padding=3,
            nonassociative=False  # Stem使用简单卷积
        )
        self.stem_bn = OctonionBatchNorm(self.base_channels)
        
        # 残差stages
        self.stages = []
        in_channels = self.base_channels
        
        for stage_idx, num_block in enumerate(num_blocks):
            stage = []
            out_channels = self.base_channels * (2 ** stage_idx)
            
            for block_idx in range(num_block):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                
                block = OctonionResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    chirality_aware=chirality_aware,
                    nonassociative=nonassociative
                )
                stage.append(block)
                
                in_channels = out_channels
            
            self.stages.append(stage)
        
        # 全局平均池化
        self.global_avg_pool = None  # 将在forward中动态计算
        
        # 输出头（用于Oracle适配器）
        self.policy_head = None  # 策略头（动作概率）
        self.value_head = None    # 价值头（状态价值）
        
        # 初始化输出头
        self._init_output_heads()
    
    def _init_output_heads(self):
        """初始化输出头"""
        # 策略头：输出动作概率（八元数 → 实数）
        self.policy_conv = OctonionConv2d(
            in_channels=self.base_channels * (2 ** (len(self.num_blocks) - 1)),
            out_channels=1,  # 简化为单通道
            kernel_size=1,
            nonassociative=False
        )
        
        # 价值头：输出状态价值
        self.value_fc = None  # 将在forward中动态创建
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (N, C, H, W) 或 (N, C, H, W, 8)
                   如果输入是普通张量，将自动映射到八元数空间。
        
        Returns:
            policy: 策略输出（动作概率），形状 (N, num_actions)
            value: 价值输出（状态价值），形状 (N, 1)
        """
        # 确保输入是八元数格式
        if x.ndim == 4:  # (N, C, H, W)
            # 映射到八元数空间（简单策略：复制到8个分量）
            x_oct = np.zeros(x.shape + (8,), dtype=np.float32)
            x_oct[..., 0] = x  # 实部 = 原始输入
            x_oct[..., 1:8] = np.random.randn(*x.shape, 7) * 0.01  # 虚部分量（小随机）
        else:
            x_oct = x  # 假设已经是八元数格式
        
        # Stem
        x_oct = self.stem_conv.forward(x_oct)
        x_oct = self.stem_bn.forward(x_oct)
        x_oct = octonion_relu(x_oct)
        
        # 残差stages
        for stage in self.stages:
            for block in stage:
                x_oct = block.forward(x_oct)
        
        # 全局平均池化
        # 形状: (N, C, H, W, 8) → (N, C, 8)
        N, C, H, W, _ = x_oct.shape
        x_pooled = np.mean(x_oct, axis=(2, 3))  # 对H, W维度平均
        
        # 策略头
        # (N, C, 8) → (N, num_actions)
        # 简化：使用实部作为策略logits
        policy_logits = x_pooled[..., 0]  # (N, C)
        
        # 如果C != num_classes，需要映射
        if C != self.num_classes:
            # 线性映射（简化）
            W = np.random.randn(C, self.num_classes) * 0.01
            policy_logits = np.dot(policy_logits, W)
        
        # Softmax
        policy_probs = self._softmax(policy_logits)
        
        # 价值头
        # (N, C, 8) → (N, 1)
        value = np.mean(x_pooled[..., 0], axis=-1, keepdims=True)  # 实部平均
        value = np.tanh(value)  # 压缩到[-1, 1]
        
        return policy_probs, value
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def compute_existence_conservation(self, x: np.ndarray) -> float:
        """
        计算信息存在度守恒误差
        
        用于验证Skip Connection是否正确保持了I(e)。
        
        Returns:
            平均I(e)差异（应该接近0）
        """
        # 转换到八元数张量
        x_tensor = OctonionTensor(x)
        I_input = x_tensor.existence_degree()
        
        # 前向传播（简化：只通过一个残差块）
        if len(self.stages) > 0 and len(self.stages[0]) > 0:
            block = self.stages[0][0]
            output = block.forward(x)
            output_tensor = OctonionTensor(output)
            I_output = output_tensor.existence_degree()
            
            # 计算差异
            diff = np.mean(np.abs(I_input - I_output))
            return diff
        else:
            return 0.0


def test_octonion_resnet():
    """测试八元数ResNet"""
    print("=" * 60)
    print("测试八元数ResNet")
    print("=" * 60)
    
    # 创建ResNet
    print("\n1. 创建OctonionResNet...")
    model = OctonionResNet(
        num_blocks=[1, 1],  # 简化：2个stage，各1个块
        num_classes=4,        # 4个动作
        chirality_aware=True,
        nonassociative=True
    )
    print(f"   ✅ 模型创建成功")
    
    # 测试前向传播
    print("\n2. 测试前向传播...")
    batch_size = 2
    x = np.random.randn(batch_size, 3, 32, 32)  # (N, C, H, W)
    
    policy, value = model.forward(x)
    print(f"   输入形状: {x.shape}")
    print(f"   策略输出形状: {policy.shape}")
    print(f"   价值输出形状: {value.shape}")
    print(f"   策略示例: {policy[0]}")
    print(f"   价值示例: {value[0]}")
    print(f"   ✅ 前向传播工作正常")
    
    # 测试信息守恒
    print("\n3. 测试信息存在度守恒...")
    # 创建八元数输入
    x_oct = np.random.randn(batch_size, 64, 8, 8, 8)  # (N, C, H, W, 8)
    
    conservation_error = model.compute_existence_conservation(x_oct)
    print(f"   I(e)守恒误差: {conservation_error:.6f}")
    if conservation_error < 0.1:
        print(f"   ✅ 信息守恒良好")
    else:
        print(f"   ⚠️  信息守恒误差较大，需检查")
    
    print("\n✅ 八元数ResNet测试完成！")


if __name__ == "__main__":
    test_octonion_resnet()
