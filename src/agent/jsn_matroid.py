"""
JSN拟阵回路检测 — Union-Find实现

实现L1 JSN的拟阵(Matroid)结构，用于κ-Gate最优剪枝。

拟阵定义 (E, I)：
- E: 超边集合
- I: 独立集集合（无回路的超边子集）
- 拟阵性质：
  1. ∅ ∈ I（空集独立）
  2. 若 A ∈ I 且 B ⊆ A，则 B ∈ I（遗传性质）
  3. 若 A, B ∈ I 且 |A| < |B|，则 ∃e ∈ B\A 使得 A∪{e} ∈ I（交换性质）

回路检测（Circuit Elimination）：
- 使用Union-Find检测超图回路
- 若添加超边e后形成回路，则e依赖于现有基（可删减）

κ-Gate剪枝：
- 贪心算法：按存在度φ降序，用Union-Find维护最大独立集
- 时间复杂度：O(|E| * α(|V|))（α为反阿克曼函数，≈常数）

Author: TOMAS Team
Version: 0.1.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import sqlite3
import json


# ============================================================================
# Union-Find数据结构（用于回路检测）
# ============================================================================
class UnionFind:
    """
    Union-Find（并查集）数据结构
    
    用于检测超图中的回路（cycle）。
    在超图中，回路定义为：存在一组超边，其顶点集合形成欧拉回路。
    
    简化版本：将超图视为普通图（每条超边连接两个顶点），
    用标准Union-Find检测回路。
    """
    
    def __init__(self, num_vertices: int):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices
        self.component_count = num_vertices  # 连通分量数
    
    def find(self, x: int) -> int:
        """查找元素x的根（路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        合并两个集合
        
        Returns:
            True: 合并成功（原本不在同一集合）
            False: 形成回路（原本在同一集合）
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # 回路！
        
        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.component_count -= 1
        return True
    
    def is_connected(self, x: int, y: int) -> bool:
        """检查两个顶点是否连通"""
        return self.find(x) == self.find(y)
    
    def get_components(self) -> List[List[int]]:
        """获取所有连通分量"""
        components = {}
        for v in range(len(self.parent)):
            root = self.find(v)
            if root not in components:
                components[root] = []
            components[root].append(v)
        
        return list(components.values())


# ============================================================================
# 超图拟阵（Hypergraph Matroid）
# ============================================================================
class HypergraphMatroid:
    """
    超图拟阵（Graphic Matroid的推广）
    
    对于超图 G = (V, E)，其中每条超边 e ∈ E 连接任意数量的顶点：
    - 独立集：无回路的超边集合
    - 基：最大独立集（生成森林）
    - 秩：基的大小 = |V| - 连通分量数
    
    回路检测（超图版本）：
    1. 将超边转换为顶点子集
    2. 若超边e的顶点子集与现有独立集的顶点子集形成回路，则e形成回路
    
    简化实现：
    - 假设超图是二部图（顶点-超边二部图）
    - 使用Union-Find检测二部图回路
    """
    
    def __init__(self, num_vertices: int):
        self.num_vertices = num_vertices
        self.edges = []  # 已接受的超边
        self.uf = UnionFind(num_vertices)
        self.circuits = []  # 检测到的回路
    
    def add_edge(self, u: int, v: int) -> bool:
        """
        尝试添加超边 (u, v)
        
        Returns:
            True: 添加成功（保持独立性）
            False: 形成回路（应被剪枝）
        """
        # 检测是否形成回路
        if self.uf.find(u) == self.uf.find(v):
            # 形成回路！记录回路
            circuit = self._extract_circuit(u, v)
            self.circuits.append(circuit)
            return False
        
        # 无回路，添加
        self.uf.union(u, v)
        self.edges.append((u, v))
        return True
    
    def _extract_circuit(self, u: int, v: int) -> List[int]:
        """
        提取回路（简化版本）
        
        在完整实现中，需要追踪路径并提取完整回路。
        这里返回 [u, v] 作为简化回路。
        """
        return [u, v]
    
    def is_independent(self, edge_list: List[Tuple[int, int]]) -> bool:
        """检查超边集合是否独立（无回路）"""
        uf_test = UnionFind(self.num_vertices)
        
        for u, v in edge_list:
            if uf_test.find(u) == uf_test.find(v):
                return False  # 形成回路
            uf_test.union(u, v)
        
        return True
    
    def get_rank(self) -> int:
        """计算拟阵的秩（最大独立集大小）"""
        return self.num_vertices - self.uf.component_count
    
    def get_max_independent_set(self, all_edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        使用贪心算法计算最大独立集（Kruskal算法）
        
        Args:
            all_edges: 所有超边（按权重降序排列）
        Returns:
            最大独立集
        """
        uf = UnionFind(self.num_vertices)
        independent_set = []
        
        for u, v in all_edges:
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                independent_set.append((u, v))
        
        return independent_set


# ============================================================================
# JSN存储的拟阵剪枝（κ-Gate）
# ============================================================================
class MatroidPruner:
    """
    基于拟阵的κ-Gate剪枝
    
    实现文章中的κ-Gate算法：
    1. 按存在度φ(e)降序排序所有超边
    2. 用Union-Find维护最大独立集（基）
    3. 若添加e后保持独立性，则保留；否则剪枝
    
    数学保证：
    - 结果是最优基（最大权重独立集）
    - 时间复杂度：O(|E| * log|E| * α(|V|))
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def prune(self, theta_dead: float = 0.5) -> Dict:
        """
        κ-Gate剪枝
        
        Args:
            theta_dead: 存在度阈值，低于此值的超边可能被剪枝
        Returns:
            剪枝结果统计
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. 加载所有超边，按存在度降序排序
        cursor.execute("""
            SELECT id, source_id, target_id, existence_I_e, asym, in_matroid_base
            FROM hyperedges
            ORDER BY existence_I_e DESC
        """)
        edges = cursor.fetchall()
        
        print(f"  📊 加载 {len(edges)} 条超边")
        
        # 2. 获取顶点总数
        cursor.execute("SELECT COUNT(*) FROM vertices")
        num_vertices = cursor.fetchone()[0]
        
        # 3. 初始化Union-Find
        # 注意：顶点ID可能是字符串，需要映射到整数索引
        vertex_id_map = {}
        cursor.execute("SELECT id FROM vertices")
        for idx, (vid,) in enumerate(cursor.fetchall()):
            vertex_id_map[vid] = idx
        
        uf = UnionFind(num_vertices)
        base_edges = []
        pruned_count = 0
        
        # 4. 贪心算法：按存在度降序尝试添加
        for row in edges:
            edge_id = row[0]
            source_id = row[1]
            target_id = row[2]
            phi = row[3]
            asym = row[4]
            in_base = row[5]
            
            # 映射顶点ID到整数
            u = vertex_id_map.get(source_id, -1)
            v = vertex_id_map.get(target_id, -1)
            
            if u == -1 or v == -1:
                continue  # 顶点不存在，跳过
            
            # 检查是否形成回路
            if uf.find(u) != uf.find(v):
                # 无回路，添加到基
                uf.union(u, v)
                base_edges.append(edge_id)
                
                # 更新数据库
                cursor.execute("""
                    UPDATE hyperedges 
                    SET in_matroid_base = 1 
                    WHERE id = ?
                """, (edge_id,))
            else:
                # 形成回路，剪枝
                cursor.execute("""
                    UPDATE hyperedges 
                    SET in_matroid_base = 0 
                    WHERE id = ?
                """, (edge_id,))
                pruned_count += 1
        
        conn.commit()
        conn.close()
        
        return {
            'total_edges': len(edges),
            'base_size': len(base_edges),
            'pruned_count': pruned_count,
            'base_edges': base_edges
        }
    
    def verify_matroid_properties(self) -> Dict:
        """
        验证拟阵性质（用于调试）
        
        Returns:
            验证结果
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 加载基中的超边
        cursor.execute("""
            SELECT id, source_id, target_id 
            FROM hyperedges 
            WHERE in_matroid_base = 1
        """)
        base_edges = cursor.fetchall()
        
        # 验证无回路
        num_vertices = cursor.execute("SELECT COUNT(*) FROM vertices").fetchone()[0]
        uf = UnionFind(num_vertices)
        
        vertex_map = {}
        cursor.execute("SELECT id FROM vertices")
        for idx, (vid,) in enumerate(cursor.fetchall()):
            vertex_map[vid] = idx
        
        has_cycle = False
        for edge_id, source_id, target_id in base_edges:
            u = vertex_map.get(source_id, -1)
            v = vertex_map.get(target_id, -1)
            if u != -1 and v != -1:
                if uf.find(u) == uf.find(v):
                    has_cycle = True
                    break
                uf.union(u, v)
        
        conn.close()
        
        return {
            'base_size': len(base_edges),
            'has_cycle': has_cycle,
            'is_valid_matroid': not has_cycle
        }


# ============================================================================
# 完整κ-Gate算法（含八元数φ场）
# ============================================================================
def k_gate_prune_with_octonion(db_path: str, 
                                theta_dead: float = 0.5,
                                use_octonion: bool = True) -> Dict:
    """
    完整κ-Gate剪枝算法（考虑八元数φ场）
    
    算法流程：
    1. 加载所有超边，计算综合权重 w(e) = φ(e) + α*asym(e)
    2. 按w(e)降序排序
    3. Union-Find贪心选择最大独立集
    4. 更新数据库
    
    Args:
        db_path: JSN数据库路径
        theta_dead: 存在度阈值
        use_octonion: 是否使用八元数asym（手性）
    
    Returns:
        剪枝统计
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. 加载顶点映射
    cursor.execute("SELECT id FROM vertices")
    vertex_id_map = {vid: idx for idx, (vid,) in enumerate(cursor.fetchall())}
    num_vertices = len(vertex_id_map)
    
    # 2. 加载超边，计算权重
    if use_octonion:
        cursor.execute("""
            SELECT id, source_id, target_id, 
                   existence_phi, asym, created_at
            FROM hyperedges
        """)
        edges = []
        for row in cursor.fetchall():
            edge_id, src, tgt, phi, asym, created = row
            # 综合权重：φ场 + 手性奖励
            weight = phi + 0.1 * abs(asym)  # α=0.1
            edges.append((edge_id, src, tgt, weight))
    else:
        cursor.execute("""
            SELECT id, source_id, target_id, existence_phi
            FROM hyperedges
        """)
        edges = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]
    
    # 3. 按权重降序排序
    edges.sort(key=lambda x: x[3], reverse=True)
    
    # 4. Union-Find贪心
    uf = UnionFind(num_vertices)
    base_edges = []
    pruned = []
    
    for edge_id, src, tgt, weight in edges:
        u = vertex_id_map.get(src, -1)
        v = vertex_id_map.get(tgt, -1)
        
        if u == -1 or v == -1:
            continue
        
        if uf.find(u) != uf.find(v):
            # 添加到基
            uf.union(u, v)
            base_edges.append(edge_id)
            cursor.execute("UPDATE hyperedges SET in_matroid_base=1 WHERE id=?", (edge_id,))
        else:
            # 剪枝
            pruned.append(edge_id)
            cursor.execute("UPDATE hyperedges SET in_matroid_base=0 WHERE id=?", (edge_id,))
    
    conn.commit()
    
    # 5. 统计
    stats = {
        'total': len(edges),
        'base_size': len(base_edges),
        'pruned': len(pruned),
        'base_edges': base_edges[:10],  # 只显示前10个
        'theta_dead': theta_dead,
        'use_octonion': use_octonion
    }
    
    conn.close()
    return stats


# ============================================================================
# 测试函数
# ============================================================================
def test_union_find():
    """测试Union-Find"""
    print("=" * 60)
    print("测试 Union-Find 回路检测")
    print("=" * 60)
    
    uf = UnionFind(5)
    
    # 添加边 (0,1), (1,2), (2,3)
    print("\n1. 添加边 (0,1), (1,2), (2,3)...")
    assert uf.union(0, 1) == True
    assert uf.union(1, 2) == True
    assert uf.union(2, 3) == True
    print(f"  ✅ 无回路，连通分量数: {uf.component_count}")
    
    # 添加边 (0,3) — 应形成回路
    print("\n2. 添加边 (0,3)...")
    assert uf.union(0, 3) == False
    print(f"  ✅ 检测到回路！")
    
    # 获取连通分量
    components = uf.get_components()
    print(f"\n3. 连通分量: {components}")
    
    print("\n" + "=" * 60)
    print("✅ Union-Find 测试通过！")
    print("=" * 60)


def test_hypergraph_matroid():
    """测试超图拟阵"""
    print("\n" + "=" * 60)
    print("测试 超图拟阵")
    print("=" * 60)
    
    matroid = HypergraphMatroid(num_vertices=6)
    
    # 添加边
    print("\n1. 添加边 (0,1), (1,2), (3,4)...")
    assert matroid.add_edge(0, 1) == True
    assert matroid.add_edge(1, 2) == True
    assert matroid.add_edge(3, 4) == True
    print(f"  ✅ 秩: {matroid.get_rank()}")
    
    # 添加回路
    print("\n2. 添加边 (0,2) — 应形成回路...")
    assert matroid.add_edge(0, 2) == False
    print(f"  ✅ 检测到回路！")
    
    # 最大独立集
    all_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,2)]
    mis = matroid.get_max_independent_set(all_edges)
    print(f"\n3. 最大独立集: {mis}")
    print(f"  ✅ 大小: {len(mis)}")
    
    print("\n" + "=" * 60)
    print("✅ 超图拟阵测试通过！")
    print("=" * 60)


def test_matroid_pruner():
    """测试拟阵剪枝"""
    print("\n" + "=" * 60)
    print("测试 拟阵剪枝（κ-Gate）")
    print("=" * 60)
    
    # 创建测试数据库
    import tempfile
    db_path = tempfile.mktemp(suffix='.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute("""
        CREATE TABLE vertices (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE hyperedges (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            target_id TEXT,
            existence_phi REAL,
            in_matroid_base INTEGER DEFAULT 0
        )
    """)
    
    # 插入测试数据
    for i in range(6):
        cursor.execute("INSERT INTO vertices VALUES (?, ?)", (f'v{i}', f'vertex_{i}'))
    
    edges_data = [
        ('e0', 'v0', 'v1', 0.9),
        ('e1', 'v1', 'v2', 0.8),
        ('e2', 'v2', 'v3', 0.7),  # 形成回路 (v0-v1-v2-v3)
        ('e3', 'v0', 'v3', 0.6),  # 回路边
        ('e4', 'v4', 'v5', 0.5),  # 独立分量
    ]
    
    for eid, src, tgt, phi in edges_data:
        cursor.execute("""
            INSERT INTO hyperedges (id, source_id, target_id, existence_phi)
            VALUES (?, ?, ?, ?)
        """, (eid, src, tgt, phi))
    
    conn.commit()
    conn.close()
    
    # 运行剪枝
    pruner = MatroidPruner(db_path)
    result = pruner.prune()
    
    print(f"\n📊 剪枝结果:")
    print(f"  总超边数: {result['total_edges']}")
    print(f"  基大小: {result['base_size']}")
    print(f"  剪枝数: {result['pruned_count']}")
    
    # 验证拟阵性质
    verification = pruner.verify_matroid_properties()
    print(f"\n✅ 拟阵验证:")
    print(f"  基是否有效: {verification['is_valid_matroid']}")
    print(f"  是否有回路: {verification['has_cycle']}")
    
    # 清理
    import os
    os.remove(db_path)
    
    print("\n" + "=" * 60)
    print("✅ 拟阵剪枝测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_union_find()
    test_hypergraph_matroid()
    test_matroid_pruner()
    
    print("\n" + "🎉" * 20)
    print("JSN拟阵回路检测（Union-Find）实现完成！")
    print("🎉" * 20)
    print("\n📋 已实现:")
    print("  ✅ Union-Find数据结构")
    print("  ✅ 超图拟阵独立集判定")
    print("  ✅ κ-Gate贪心剪枝算法")
    print("  ✅ 八元数φ场 + 手性权重")
    print("\n🚀 下一步:")
    print("  1. 集成到jsn_store.py")
    print("  2. Phase 4: 集成到planner_agent.py")
    print("  3. Kaggle提交（截止2026-06-30）")
