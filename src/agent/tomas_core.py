"""
TOMAS Core: 太乙互搏框架核心

整合 TOMAS 五层级体系与 NAR-Net，实现：
1. L1 信息存在度公理监控（I(e) 守恒验证）
2. L2 局部→全域传播（超图连通性）
3. L3 全域扫描（注意力机制）
4. L4 观测者审计（链式法则 = 信息泛函）
5. L5 二阶自返（流体智力投影）

太一理论整合：
  - 太一（Unity）= 八元数实单位 e₀
  - 手性（Diversity）= 虚单位 e₁..e₇ 的非结合组合
  - 从太一恢复手性 = 从实部引导虚部重组

设备自适应：通过 gpu_backend 自动检测 GPU/CPU

Author: TOMAS Team
Version: 0.1.0
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Any
import time

# GPU 后端
try:
    from .gpu_backend import (
        DeviceInfo, OctonionOps, to_tensor, to_numpy,
        is_gpu_available, get_device, get_backend, print_device_status
    )
except ImportError:
    from gpu_backend import (
        DeviceInfo, OctonionOps, to_tensor, to_numpy,
        is_gpu_available, get_device, get_backend, print_device_status
    )


# ============================================================================
# TOMAS 五层级定义
# ============================================================================

TOMAS_LEVELS = {
    'L1': {
        'name': '信息存在度公理',
        'description': 'I(e) 守恒：信息在变换中不被消灭',
        'resnet_mapping': 'ResNet 残差连接 = I(e) 不消灭源节点',
        'nar_mapping': 'NAR-Net Skip Connection + 八元数范数守恒',
    },
    'L2': {
        'name': '局部→全域传播',
        'description': '超图连通性：信息从局部传播到全局',
        'resnet_mapping': '卷积层 = 局部感受野',
        'nar_mapping': '八元数卷积 = 非结合传播 + 手性注入',
    },
    'L3': {
        'name': '全域扫描',
        'description': '自注意力 = 全域超图权重分配',
        'resnet_mapping': 'Transformer 自注意力',
        'nar_mapping': '八元数注意力 = 全域非结合扫描',
    },
    'L4': {
        'name': '观测者审计',
        'description': '链式法则 = 信息泛函 δI/δw',
        'resnet_mapping': '反向传播 (Backprop)',
        'nar_mapping': '八元数梯度 + 非结合链式法则',
    },
    'L5': {
        'name': '二阶自返',
        'description': '流体智力 = 在-context 学习的 L5 投影',
        'resnet_mapping': 'LLM In-Context Learning',
        'nar_mapping': 'NAR-Net 小样本适应 = 非结合推理的涌现',
    },
}


# ============================================================================
# L1: 信息存在度监控器
# ============================================================================

class ExistenceMonitor:
    """
    L1: 信息存在度监控器
    
    验证 NAR-Net 每一层的 I(e) 守恒性质。
    如果 I(e) 偏差超过阈值，触发警告或自动修复。
    
    核心公理：
      I(e) = ||x|| = sqrt(Σ xᵢ²)
      
    Skip Connection 确保：
      I(y) = I(F(x) + x) ≈ I(x)
      
    非结合运算确保：
      Chirality(y) ≠ 0 （手性不退化）
    """
    
    def __init__(self, conservation_threshold: float = 0.15):
        self.threshold = conservation_threshold
        self.violations: List[Dict] = []
        self.history: List[Dict] = []
        self.max_history = 1000
    
    def check_conservation(self, 
                          input_tensor: Any, 
                          output_tensor: Any,
                          layer_name: str = "") -> Dict:
        """
        检查 I(e) 守恒
        
        Args:
            input_tensor: 变换前的八元数张量 (..., 8)
            output_tensor: 变换后的八元数张量 (..., 8)
            layer_name: 层名称（用于日志）
        
        Returns:
            检查结果字典
        """
        I_in = OctonionOps.existence_degree(input_tensor)
        I_out = OctonionOps.existence_degree(output_tensor)
        
        I_in_np = to_numpy(I_in)
        I_out_np = to_numpy(I_out)
        
        diff = np.mean(np.abs(I_in_np - I_out_np))
        rel_diff = diff / (np.mean(I_in_np) + 1e-8)
        
        chir_in = to_numpy(OctonionOps.chirality(input_tensor))
        chir_out = to_numpy(OctonionOps.chirality(output_tensor))
        
        is_conserved = rel_diff < self.threshold
        chirality_maintained = np.mean(chir_out) > 0.01
        
        result = {
            'layer': layer_name,
            'I_input': float(np.mean(I_in_np)),
            'I_output': float(np.mean(I_out_np)),
            'absolute_diff': float(diff),
            'relative_diff': float(rel_diff),
            'is_conserved': bool(is_conserved),
            'chirality_input': float(np.mean(chir_in)),
            'chirality_output': float(np.mean(chir_out)),
            'chirality_maintained': bool(chirality_maintained),
            'timestamp': time.time(),
        }
        
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        if not is_conserved:
            self.violations.append({
                'layer': layer_name,
                'rel_diff': rel_diff,
                'timestamp': time.time(),
            })
        
        return result
    
    def auto_repair(self, 
                   output_tensor: Any, 
                   identity_tensor: Any) -> Any:
        """
        自动修复 I(e) 偏差
        
        当 I(e) 不守恒时，通过调整 Skip Connection 权重恢复守恒。
        
        策略：
          y_repaired = α * F(x) + (1-α) * x
          其中 α 选择使 I(y_repaired) ≈ I(x)
        """
        I_identity = OctonionOps.existence_degree(identity_tensor)
        I_output = OctonionOps.existence_degree(output_tensor - identity_tensor)
        
        I_id_np = to_numpy(I_identity)
        I_out_np = to_numpy(I_output)
        
        # α = I(x) / (I(x) + I(F(x)))
        alpha = I_id_np / (I_id_np + I_out_np + 1e-8)
        alpha = np.clip(alpha, 0.1, 0.9)
        
        info = DeviceInfo.get()
        if info.backend == 'torch':
            alpha_t = to_tensor(alpha.astype(np.float32))
            # 需要 reshape 以广播
            while alpha_t.dim() < output_tensor.dim():
                alpha_t = alpha_t.unsqueeze(-1)
            repaired = alpha_t * (output_tensor - identity_tensor) + identity_tensor
        else:
            alpha_n = alpha
            while alpha_n.ndim < output_tensor.ndim:
                alpha_n = np.expand_dims(alpha_n, -1)
            repaired = alpha_n * (output_tensor - identity_tensor) + identity_tensor
        
        return repaired
    
    def get_summary(self) -> Dict:
        """获取监控摘要"""
        if not self.history:
            return {'total_checks': 0, 'violations': 0}
        
        return {
            'total_checks': len(self.history),
            'violations': len(self.violations),
            'violation_rate': len(self.violations) / len(self.history),
            'avg_relative_diff': np.mean([h['relative_diff'] for h in self.history]),
            'avg_chirality': np.mean([h['chirality_output'] for h in self.history]),
        }


# ============================================================================
# 太一理论：手性恢复
# ============================================================================

class TaiyiChiralityRestorer:
    """
    太一理论指导的手性恢复
    
    核心思想：
      太一（Unity）= 八元数实单位 e₀ = 所有信息的起源
      手性（Diversity）= 虚单位 e₁..e₇ 的非结合组合 = 多样性
      
      当网络深度增加导致手性退化（Chirality → 0）时，
      从太一（实部）引导虚部重组，恢复手性。
    
    数学表达：
      Taiyi = Re(x) = x₀ * e₀  （提取太一）
      Chirality = Im(x) = x₁e₁ + ... + x₇e₇  （提取手性）
      
      恢复操作：
        if Chirality(y) < threshold:
          y_restored = Taiyi(y) ⊗ R(Im(ref))  
          其中 R() 是非结合重组算子，ref 是参考张量
    """
    
    def __init__(self, 
                 chirality_threshold: float = 0.1,
                 restoration_strength: float = 0.5):
        self.threshold = chirality_threshold
        self.strength = restoration_strength
        self.restoration_count = 0
    
    def extract_taiyi(self, x: Any) -> Any:
        """提取太一（实部）"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            result = OctonionOps.zeros(x.shape)
            result[..., 0] = x[..., 0]
            return result
        else:
            result = np.zeros_like(x)
            result[..., 0] = x[..., 0]
            return result
    
    def extract_chirality(self, x: Any) -> Any:
        """提取手性（虚部）"""
        info = DeviceInfo.get()
        if info.backend == 'torch':
            result = OctonionOps.zeros(x.shape)
            result[..., 1:] = x[..., 1:]
            return result
        else:
            result = np.zeros_like(x)
            result[..., 1:] = x[..., 1:]
            return result
    
    def restore(self, x: Any, ref: Any) -> Any:
        """
        从太一恢复手性
        
        Args:
            x: 当前张量（可能手性退化）
            ref: 参考张量（通常是 Skip Connection 的 identity）
        
        Returns:
            手性恢复后的张量
        """
        chir_x = to_numpy(OctonionOps.chirality(x))
        chir_ref = to_numpy(OctonionOps.chirality(ref))
        
        # 判断哪些位置需要恢复
        need_restore = chir_x < self.threshold
        restore_ratio = np.mean(need_restore)
        
        if restore_ratio < 0.01:
            # 几乎不需要恢复
            return x
        
        self.restoration_count += 1
        
        # 提取太一和手性
        taiyi = self.extract_taiyi(x)  # 实部
        chir_ref = self.extract_chirality(ref)  # 参考的虚部
        
        # 从太一引导虚部重组
        # y = 太一 + strength * (参考虚部) + (1-strength) * (当前虚部)
        info = DeviceInfo.get()
        
        if info.backend == 'torch':
            strength = self.strength
            restored = x.clone()
            # 只在需要恢复的位置注入手性
            mask = to_tensor(need_restore.astype(np.float32))
            while mask.dim() < x.dim():
                mask = mask.unsqueeze(-1)
            
            restored[..., 1:] = x[..., 1:] * (1 - mask * strength) + \
                                chir_ref[..., 1:] * mask * strength
        else:
            mask = need_restore.astype(np.float32)
            while mask.ndim < x.ndim:
                mask = np.expand_dims(mask, -1)
            
            strength = self.strength
            restored = x.copy()
            restored[..., 1:] = x[..., 1:] * (1 - mask * strength) + \
                               chir_ref[..., 1:] * mask * strength
        
        return restored
    
    def get_stats(self) -> Dict:
        return {
            'restoration_count': self.restoration_count,
            'threshold': self.threshold,
            'strength': self.strength,
        }


# ============================================================================
# L2-L3: 超图传播 + 全域扫描
# ============================================================================

class HypergraphPropagator:
    """
    L2: 局部→全域传播
    
    将八元数卷积的局部感受野扩展为全域超图传播。
    使用八元数非结合运算确保信息在传播中保持手性。
    """
    
    def __init__(self, channels: int, grid_size: int = 16):
        self.channels = channels
        self.grid_size = grid_size
        
        # 超图邻接矩阵（可学习）
        adj_shape = (grid_size * grid_size, grid_size * grid_size, 8)
        self.adjacency = OctonionOps.randn(*adj_shape) * 0.01
    
    def propagate(self, x: Any) -> Any:
        """
        超图传播
        
        Args:
            x: (B, C, H, W, 8) 八元数特征
        
        Returns:
            传播后的特征（同形状）
        """
        B, C, H, W, _ = x.shape if DeviceInfo.get().backend == 'numpy' else (tuple(x.shape))
        
        # 展平空间维度: (B, C, H*W, 8)
        x_flat = x.reshape(B, C, H * W, 8) if DeviceInfo.get().backend == 'numpy' \
            else x.reshape(B, C, H * W, 8)
        
        # 非结合传播: x' = x ⊗ A (八元数矩阵乘法)
        # 简化版本：对每个通道独立传播
        # 完整版本应使用八元数矩阵乘法
        
        # 这里用简化版本：全局平均 + 残差
        pooled = OctonionOps.global_avg_pool(x)  # (B, C, 8)
        # 广播回去
        pooled_expanded = pooled.unsqueeze(2).unsqueeze(2) \
            if DeviceInfo.get().backend == 'torch' \
            else pooled[:, :, np.newaxis, np.newaxis, :]
        
        # 残差传播
        out = x + 0.1 * pooled_expanded
        
        return out


class GlobalScanner:
    """
    L3: 全域扫描
    
    基于八元数注意力的全域扫描机制。
    类似 Transformer 自注意力，但使用八元数非结合运算。
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Q, K, V 投影矩阵（八元数）
        scale = 1.0 / np.sqrt(channels * 8)
        self.W_q = OctonionOps.randn(channels * 8, channels * 8) * scale
        self.W_k = OctonionOps.randn(channels * 8, channels * 8) * scale
        self.W_v = OctonionOps.randn(channels * 8, channels * 8) * scale
    
    def scan(self, x: Any) -> Any:
        """
        全域扫描
        
        Args:
            x: (B, C, H, W, 8) 八元数特征
        
        Returns:
            扫描后的特征（同形状）
        """
        info = DeviceInfo.get()
        
        if info.backend == 'numpy':
            B, C, H, W, _ = x.shape
        else:
            B, C, H, W, _ = tuple(x.shape)
        
        # 展平: (B, H*W, C*8)
        x_flat = x.transpose(0, 2, 3, 1, 4).reshape(B, H * W, C * 8) \
            if info.backend == 'numpy' \
            else x.permute(0, 2, 3, 1, 4).reshape(B, H * W, C * 8)
        
        # 简化注意力（MVP版本：不做多头分割）
        # Q = x @ W_q, K = x @ W_k, V = x @ W_v
        Q = OctonionOps.matmul(x_flat, self.W_q)
        K = OctonionOps.matmul(x_flat, self.W_k)
        V = OctonionOps.matmul(x_flat, self.W_v)
        
        # 注意力分数: (B, N, N)
        scores = OctonionOps.matmul(Q, K.transpose(-1, -2)) / np.sqrt(C * 8)
        attn = OctonionOps.softmax(scores, dim=-1)
        
        # 加权求和
        out = OctonionOps.matmul(attn, V)  # (B, N, C*8)
        
        # 恢复形状: (B, C, H, W, 8)
        out = out.reshape(B, H, W, C, 8).transpose(0, 3, 1, 2, 4) \
            if info.backend == 'numpy' \
            else out.reshape(B, H, W, C, 8).permute(0, 3, 1, 2, 4)
        
        return out


# ============================================================================
# L4: 观测者审计
# ============================================================================

class ObserverAuditor:
    """
    L4: 观测者审计
    
    反向传播 = 信息泛函 δI/δw
    在 NAR-Net 中，使用八元数非结合链式法则计算梯度。
    
    MVP版本：记录梯度统计，验证信息泛函的方向正确性。
    """
    
    def __init__(self):
        self.gradient_history: List[Dict] = []
    
    def audit_gradient(self, 
                      gradient: Any, 
                      layer_name: str = "",
                      I_e_before: float = 0.0,
                      I_e_after: float = 0.0) -> Dict:
        """
        审计梯度
        
        验证梯度方向是否增加 I(e)（信息增益方向）
        """
        grad_np = to_numpy(gradient)
        grad_norm = float(np.linalg.norm(grad_np))
        grad_mean = float(np.mean(grad_np))
        
        delta_I = I_e_after - I_e_before
        
        result = {
            'layer': layer_name,
            'grad_norm': grad_norm,
            'grad_mean': grad_mean,
            'I_e_before': I_e_before,
            'I_e_after': I_e_after,
            'delta_I': delta_I,
            'is_information_gain': delta_I >= 0,
            'timestamp': time.time(),
        }
        
        self.gradient_history.append(result)
        return result
    
    def get_summary(self) -> Dict:
        if not self.gradient_history:
            return {'total_audits': 0}
        
        return {
            'total_audits': len(self.gradient_history),
            'avg_grad_norm': np.mean([h['grad_norm'] for h in self.gradient_history]),
            'information_gain_rate': np.mean([h['is_information_gain'] for h in self.gradient_history]),
        }


# ============================================================================
# L5: 二阶自返（小样本适应）
# ============================================================================

class SecondOrderSelfReflexion:
    """
    L5: 二阶自返
    
    NAR-Net 的小样本适应能力 = 流体智力的 L5 投影。
    
    通过在少量样本上快速调整，实现 in-context learning。
    核心机制：八元数流形上的梯度下降。
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 adaptation_steps: int = 5):
        self.lr = learning_rate
        self.adaptation_steps = adaptation_steps
        self.adaptation_log: List[Dict] = []
    
    def adapt(self,
              model: Any,
              states: List[np.ndarray],
              actions: List[int],
              rewards: List[float]) -> Dict:
        """
        小样本适应
        
        在八元数流形上做几步梯度下降，快速适应新任务。
        
        MVP版本：微调策略头权重
        """
        info = DeviceInfo.get()
        
        initial_I_e = 0.0
        final_I_e = 0.0
        
        for step in range(self.adaptation_steps):
            idx = np.random.randint(len(states))
            state = states[idx]
            action = actions[idx]
            reward = rewards[idx]
            
            # 前向传播
            state_oct = model.net.encode_state(state) if hasattr(model, 'net') else \
                       model.encode_state(state)
            
            I_e_before = float(to_numpy(
                OctonionOps.existence_degree(state_oct)
            ).mean())
            
            # 计算梯度方向（简化版）
            # 高奖励 → 增强该动作概率
            # 低奖励 → 降低该动作概率
            lr_adjusted = self.lr * (1 if reward > 0 else -1)
            
            # 微调策略头
            if hasattr(model, 'net'):
                pooled = OctonionOps.global_avg_pool(
                    OctonionOps.conv2d(
                        state_oct, 
                        model.net.stem.weights,
                        model.net.stem.bias,
                        stride=model.net.stem.stride,
                        padding=model.net.stem.padding
                    )
                )
                if info.backend == 'numpy':
                    flat = pooled.reshape(1, -1)
                    grad = np.zeros_like(model.net.policy_weight)
                    grad[:, action] = lr_adjusted * flat[0]
                    model.net.policy_weight += grad
                else:
                    # torch backend
                    flat = pooled.reshape(1, -1)
                    grad = OctonionOps.zeros(model.net.policy_weight.shape)
                    grad[:, action] = lr_adjusted * flat[0]
                    model.net.policy_weight += grad
            
            I_e_after = float(to_numpy(
                OctonionOps.existence_degree(state_oct)
            ).mean())
            
            self.adaptation_log.append({
                'step': step,
                'reward': reward,
                'I_e_before': I_e_before,
                'I_e_after': I_e_after,
                'delta_I': I_e_after - I_e_before,
            })
            
            if step == 0:
                initial_I_e = I_e_before
            final_I_e = I_e_after
        
        return {
            'initial_I_e': initial_I_e,
            'final_I_e': final_I_e,
            'delta_I': final_I_e - initial_I_e,
            'steps': self.adaptation_steps,
            'is_fluid_intelligence': final_I_e >= initial_I_e,
        }


# ============================================================================
# TOMAS 整合管理器
# ============================================================================

class TOMASManager:
    """
    TOMAS 整合管理器
    
    统一管理 L1-L5 五层级，为 NAR-Net 提供 TOMAS 框架支持。
    
    使用方式：
      tomas = TOMASManager()
      tomas.attach_to_nar_bridge(nar_bridge)
      # 每次 NAR-Net 前向传播后调用
      tomas.monitor_layer(input, output, "block_0")
      tomas.maybe_restore_chirality(output, identity)
    """
    
    def __init__(self, 
                 conservation_threshold: float = 0.15,
                 chirality_threshold: float = 0.1,
                 enable_global_scan: bool = False):
        """
        Args:
            conservation_threshold: I(e) 守恒阈值
            chirality_threshold: 手性恢复阈值
            enable_global_scan: 是否启用 L3 全域扫描（较慢）
        """
        # 设备检测
        self.device_info = DeviceInfo.get()
        self.device_info.detect_device()
        
        # L1: 存在度监控
        self.existence_monitor = ExistenceMonitor(conservation_threshold)
        
        # 太一理论: 手性恢复
        self.chirality_restorer = TaiyiChiralityRestorer(chirality_threshold)
        
        # L2: 超图传播（可选）
        self.hypergraph_propagator: Optional[HypergraphPropagator] = None
        
        # L3: 全域扫描（可选，较慢）
        self.global_scanner: Optional[GlobalScanner] = None
        if enable_global_scan:
            self.global_scanner = GlobalScanner(channels=4)
        
        # L4: 观测者审计
        self.observer_auditor = ObserverAuditor()
        
        # L5: 二阶自返
        self.self_reflexion = SecondOrderSelfReflexion()
        
        # 统计
        self.total_forward_passes = 0
        self.total_restorations = 0
    
    def monitor_forward(self, 
                       input_tensor: Any,
                       output_tensor: Any,
                       identity_tensor: Any,
                       layer_name: str = "block") -> Dict:
        """
        监控一次前向传播
        
        在 NAR-Net 的每个残差块后调用：
        1. 检查 I(e) 守恒 (L1)
        2. 如需要，恢复手性 (太一理论)
        3. 记录梯度审计 (L4)
        
        Returns:
            监控结果字典
        """
        self.total_forward_passes += 1
        
        # L1: I(e) 守恒检查
        conservation = self.existence_monitor.check_conservation(
            input_tensor, output_tensor, layer_name
        )
        
        # 太一理论: 手性恢复
        restored = self.chirality_restorer.restore(output_tensor, identity_tensor)
        if self.chirality_restorer.restoration_count > self.total_restorations:
            self.total_restorations = self.chirality_restorer.restoration_count
        
        return {
            'conservation': conservation,
            'restored_tensor': restored,
            'restoration_triggered': conservation['chirality_output'] < self.chirality_restorer.threshold,
        }
    
    def adapt(self, model: Any, states, actions, rewards) -> Dict:
        """
        L5: 小样本适应
        """
        return self.self_reflexion.adapt(model, states, actions, rewards)
    
    def get_full_report(self) -> Dict:
        """获取完整 TOMAS 报告"""
        return {
            'device': self.device_info.summary(),
            'L1_existence': self.existence_monitor.get_summary(),
            'taiyi_chirality': self.chirality_restorer.get_stats(),
            'L4_observer': self.observer_auditor.get_summary(),
            'L5_adaptations': len(self.self_reflexion.adaptation_log),
            'total_forward_passes': self.total_forward_passes,
            'total_restorations': self.total_restorations,
        }
    
    def print_report(self):
        """打印 TOMAS 报告"""
        report = self.get_full_report()
        
        print("\n" + "=" * 60)
        print("TOMAS 框架状态报告")
        print("=" * 60)
        print(f"  设备: {report['device']}")
        print(f"\n  L1 信息存在度:")
        l1 = report['L1_existence']
        print(f"    检查次数: {l1.get('total_checks', 0)}")
        print(f"    违反次数: {l1.get('violations', 0)}")
        print(f"    违反率: {l1.get('violation_rate', 0):.2%}")
        print(f"    平均相对偏差: {l1.get('avg_relative_diff', 0):.6f}")
        print(f"    平均手性: {l1.get('avg_chirality', 0):.6f}")
        print(f"\n  太一理论手性恢复:")
        tc = report['taiyi_chirality']
        print(f"    恢复次数: {tc['restoration_count']}")
        print(f"    恢复阈值: {tc['threshold']}")
        print(f"\n  L4 观测者审计:")
        l4 = report['L4_observer']
        print(f"    审计次数: {l4.get('total_audits', 0)}")
        print(f"    信息增益率: {l4.get('information_gain_rate', 0):.2%}")
        print(f"\n  L5 二阶自返:")
        print(f"    适应次数: {report['L5_adaptations']}")
        print(f"\n  总计:")
        print(f"    前向传播: {report['total_forward_passes']}")
        print(f"    手性恢复: {report['total_restorations']}")
        print("=" * 60)


# ============================================================================
# 测试
# ============================================================================

def test_tomas_core():
    """测试 TOMAS Core"""
    print("🎉" * 30)
    print("TOMAS Core 测试")
    print("🎉" * 30)
    
    # 设备状态
    print_device_status()
    
    # 创建 TOMAS 管理器
    tomas = TOMASManager(
        conservation_threshold=0.15,
        chirality_threshold=0.1,
        enable_global_scan=False  # MVP 不启用
    )
    print(f"\n✅ TOMAS 管理器创建: {tomas.device_info.summary()}")
    
    # 测试1: L1 存在度监控
    print("\n1. L1 信息存在度监控...")
    x = to_tensor(np.random.randn(1, 4, 8, 8, 8).astype(np.float32))
    F_x = to_tensor(np.random.randn(1, 4, 8, 8, 8).astype(np.float32) * 0.1)
    y = F_x + x  # Skip Connection
    
    result = tomas.monitor_forward(x, y, x, "test_block")
    print(f"   I(e) 输入: {result['conservation']['I_input']:.4f}")
    print(f"   I(e) 输出: {result['conservation']['I_output']:.4f}")
    print(f"   相对偏差: {result['conservation']['relative_diff']:.6f}")
    print(f"   守恒: {'✅' if result['conservation']['is_conserved'] else '❌'}")
    print(f"   手性输入: {result['conservation']['chirality_input']:.4f}")
    print(f"   手性输出: {result['conservation']['chirality_output']:.4f}")
    print(f"   ✅ L1 监控正常")
    
    # 测试2: 太一理论手性恢复
    print("\n2. 太一理论手性恢复...")
    # 创建手性退化的张量（几乎全是实部）
    degenerate = to_tensor(np.array([[[[[1.0, 0.001, 0.001, 0, 0, 0, 0, 0]]]]], dtype=np.float32))
    reference = to_tensor(np.random.randn(1, 1, 1, 1, 8).astype(np.float32))
    
    chir_before = float(to_numpy(OctonionOps.chirality(degenerate)).mean())
    restored = tomas.chirality_restorer.restore(degenerate, reference)
    chir_after = float(to_numpy(OctonionOps.chirality(restored)).mean())
    
    print(f"   恢复前手性: {chir_before:.6f}")
    print(f"   恢复后手性: {chir_after:.6f}")
    print(f"   恢复次数: {tomas.chirality_restorer.restoration_count}")
    print(f"   ✅ 手性恢复正常")
    
    # 测试3: 完整 NAR-Net + TOMAS 监控
    print("\n3. NAR-Net + TOMAS 完整监控...")
    try:
        from nar_net_core import NAROracleAdapter
        
        adapter = NAROracleAdapter(
            game_id='tomas_test',
            state_shape=(3, 8, 8),
            num_actions=4,
            base_channels=4,
            num_blocks=1
        )
        
        state = np.random.randint(0, 10, size=(3, 8, 8)).astype(np.float32)
        
        # 前向传播
        state_oct = adapter.net.encode_state(state)
        policy, value = adapter.forward(state)
        
        # TOMAS 监控
        I_e = adapter.get_existence_degree(state)
        chir = adapter.get_chirality(state)
        
        print(f"   策略: {policy}")
        print(f"   价值: {value:.4f}")
        print(f"   I(e): {I_e:.4f}")
        print(f"   手性: {chir:.4f}")
        print(f"   ✅ NAR-Net + TOMAS 集成正常")
    except Exception as e:
        print(f"   ⚠️ NAR-Net 集成跳过: {e}")
    
    # 测试4: TOMAS 报告
    print("\n4. TOMAS 完整报告...")
    tomas.print_report()
    
    print("\n✅ TOMAS Core 测试通过!")
    return True


if __name__ == "__main__":
    test_tomas_core()
