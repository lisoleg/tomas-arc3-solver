"""
L1 JSN (Joint Semantic Network) 存储子系统
==============================================

TOMAS架构的L1持久化与查询接口（存储层）。

职责：
  - 被 L2 κ-Snap 读取（概念/关系查询）
  - 被 L4 Sleep-Step 写入（成功P*抽象为Macro → 存hyperedge）
  - NAR-Net (L5) 不直接访问

核心设计（基于《TOMAS超图数据库的设计与实现》）：
  1. Vertex表：含 phi_b0~phi_b7 (八元数8分量)
  2. HyperEdge表：含 asym (八元数量值, 标记MUS/Paradox)
  3. EML二进制格式：Vertex 80B, Edge 32B
  4. κ-Gate拟阵贪心剪枝：Union-Find回路检测（真正拟阵算法）
  5. k-hop子图查询：HyperIndex支持快速子图加载

Author: TOMAS Team
Version: 0.2.0
"""

import numpy as np
import sqlite3
import json
import time
import struct
from typing import Optional, Tuple, Dict, List, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import os


# ============================================================================
# 数据模型定义
# ============================================================================

@dataclass
class JSNVertex:
    """
    JSN顶点（概念节点）
    
    对应文章中的Vertex表，含八元数φ场
    """
    id: str                              # 顶点ID
    name: str                            # 概念名称
    category: str = "entity"             # 类别: entity/relation/action/macro
    
    # 八元数φ场 (8个分量, Float64)
    phi_b0: float = 0.0                  # 实单位 (太一)
    phi_b1: float = 0.0                  # 虚单位 e1
    phi_b2: float = 0.0                  # 虚单位 e2
    phi_b3: float = 0.0                  # 虚单位 e3
    phi_b4: float = 0.0                  # 虚单位 e4
    phi_b5: float = 0.0                  # 虚单位 e5
    phi_b6: float = 0.0                  # 虚单位 e6
    phi_b7: float = 0.0                  # 虚单位 e7
    
    # 存在度
    existence_I_e: float = 1.0           # I(e) = ||φ||₂
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str = "unknown"              # 来源: demo/k snap/sleep_step
    confidence: float = 1.0              # 置信度 [0, 1]
    
    def to_octonion(self) -> np.ndarray:
        """转换为八元数数组 [8]"""
        return np.array([
            self.phi_b0, self.phi_b1, self.phi_b2, self.phi_b3,
            self.phi_b4, self.phi_b5, self.phi_b6, self.phi_b7
        ], dtype=np.float64)
    
    def from_octonion(self, vec: np.ndarray):
        """从八元数数组加载"""
        assert len(vec) == 8
        self.phi_b0 = float(vec[0])
        self.phi_b1 = float(vec[1])
        self.phi_b2 = float(vec[2])
        self.phi_b3 = float(vec[3])
        self.phi_b4 = float(vec[4])
        self.phi_b5 = float(vec[5])
        self.phi_b6 = float(vec[6])
        self.phi_b7 = float(vec[7])
        self.existence_I_e = float(np.linalg.norm(vec))
    
    def compute_chirality(self) -> float:
        """计算手性（虚单位组合的非对称性）"""
        imag_part = np.array([
            self.phi_b1, self.phi_b2, self.phi_b3,
            self.phi_b4, self.phi_b5, self.phi_b6, self.phi_b7
        ])
        return float(np.linalg.norm(imag_part))
    
    def to_eml_bytes(self) -> bytes:
        """
        序列化为EML二进制格式（Vertex: 80 bytes）
        
        格式：
          Bytes 0-7:   phi_b0 (float64)
          Bytes 8-71:  phi_b1~phi_b7 (7×float64 = 56 bytes)
          Bytes 72-79: metadata (uint64: category + flags)
        """
        buf = bytearray(80)
        # φ场 (8×float64 = 64 bytes)
        phi = self.to_octonion()
        for i in range(8):
            struct.pack_into('<d', buf, i * 8, phi[i])
        
        # 元数据 (16 bytes)
        meta = int(self.existence_I_e * 1000) & 0xFFFF
        struct.pack_into('<Q', buf, 72, meta)
        
        return bytes(buf)
    
    @classmethod
    def from_eml_bytes(cls, data: bytes, vertex_id: str) -> 'JSNVertex':
        """从EML二进制反序列化"""
        assert len(data) == 80
        phi = np.zeros(8)
        for i in range(8):
            phi[i] = struct.unpack_from('<d', data, i * 8)[0]
        
        v = cls(id=vertex_id, name=vertex_id)
        v.from_octonion(phi)
        return v


@dataclass
class JSNHyperEdge:
    """
    JSN超边（关系边）
    
    对应文章中的HyperEdge表，含asym字段（八元数量值）
    """
    id: str                              # 超边ID
    source_id: str                       # 源顶点ID
    target_id: str                       # 目标顶点ID
    relation: str = "related"            # 关系类型
    
    # 八元数量值（用于非结合运算）
    asym: float = 0.0                   # asym≠0 → MUS-Circuit (允许双存)
                                          # asym=0 → Paradox-Circuit (需XOR消解)
    
    # 信息存在度
    existence_I_e: float = 1.0           # I(e) = 权重 × 置信度
    weight: float = 1.0                  # 权重
    confidence: float = 1.0              # 置信度
    
    # 拟阵属性
    in_matroid_base: bool = False         # 是否在拟阵基中
    circuit_type: str = "unknown"        # MUS-Circuit / Paradox-Circuit / None
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    source: str = "unknown"              # 来源
    
    def compute_I_e(self) -> float:
        """计算信息存在度"""
        self.existence_I_e = self.weight * self.confidence
        return self.existence_I_e
    
    def classify_circuit(self) -> str:
        """
        回路分型
        
        Returns:
            "MUS-Circuit": asym≠0, 允许互斥双存
            "Paradox-Circuit": asym=0, 需XOR消解
            "None": 非回路
        """
        if abs(self.asym) > 1e-6:
            self.circuit_type = "MUS-Circuit"
        else:
            self.circuit_type = "Paradox-Circuit"
        return self.circuit_type
    
    def to_eml_bytes(self) -> bytes:
        """
        序列化为EML二进制格式（Edge: 32 bytes）
        
        格式：
          Bytes 0-7:   source_id hash (uint64)
          Bytes 8-15:  target_id hash (uint64)
          Bytes 16-23: asym (float64)
          Bytes 24-31: existence_I_e (float64)
        """
        buf = bytearray(32)
        struct.pack_into('<Q', buf, 0, hash(self.source_id) & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into('<Q', buf, 8, hash(self.target_id) & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into('<d', buf, 16, self.asym)
        struct.pack_into('<d', buf, 24, self.existence_I_e)
        return bytes(buf)
    
    @classmethod
    def from_eml_bytes(cls, data: bytes, edge_id: str) -> 'JSNHyperEdge':
        """从EML二进制反序列化"""
        assert len(data) == 32
        source_hash = struct.unpack_from('<Q', data, 0)[0]
        target_hash = struct.unpack_from('<Q', data, 8)[0]
        asym = struct.unpack_from('<d', data, 16)[0]
        I_e = struct.unpack_from('<d', data, 24)[0]
        
        e = cls(
            id=edge_id,
            source_id=f"v_{source_hash:x}",
            target_id=f"v_{target_hash:x}",
            asym=asym,
            existence_I_e=I_e
        )
        return e


# ============================================================================
# L1 JSN 存储接口（核心）
# ============================================================================

class JSNStore:
    """
    L1 JSN 存储子系统
    
    职责：
      - 被 L2 κ-Snap 读取（get_vertex, get_k_hop, search_concepts）
      - 被 L4 Sleep-Step 写入（add_vertex, add_hyperedge, save_macro）
      - NAR-Net (L5) 不直接访问
    
    存储后端：
      - SQLite（主存储，支持事务）
      - EML二进制文件（快照，快速加载）
    """
    
    def __init__(self, db_path: str = "tomas_jsn.db", eml_dir: str = "jsn_eml"):
        """
        Args:
            db_path: SQLite数据库路径
            eml_dir: EML二进制文件目录
        """
        self.db_path = db_path
        self.eml_dir = eml_dir
        
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        os.makedirs(eml_dir, exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        # 内存缓存（加速查询）
        self._vertex_cache: Dict[str, JSNVertex] = {}
        self._edge_cache: Dict[str, JSNHyperEdge] = {}
        self._cache_dirty = False
        
        # 统计
        self.stats = {
            'vertex_count': 0,
            'edge_count': 0,
            'matroid_base_size': 0,
            'I_e_pruning_ratio': 0.0,
        }
        self._update_stats()
    
    def _init_db(self):
        """初始化SQLite数据库（建表）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vertex表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vertices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT DEFAULT 'entity',
                
                -- 八元数φ场 (8分量)
                phi_b0 REAL DEFAULT 0.0,
                phi_b1 REAL DEFAULT 0.0,
                phi_b2 REAL DEFAULT 0.0,
                phi_b3 REAL DEFAULT 0.0,
                phi_b4 REAL DEFAULT 0.0,
                phi_b5 REAL DEFAULT 0.0,
                phi_b6 REAL DEFAULT 0.0,
                phi_b7 REAL DEFAULT 0.0,
                
                existence_I_e REAL DEFAULT 1.0,
                created_at REAL,
                updated_at REAL,
                source TEXT DEFAULT 'unknown',
                confidence REAL DEFAULT 1.0
            )
        """)
        
        # HyperEdge表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperedges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT DEFAULT 'related',
                
                asym REAL DEFAULT 0.0,
                existence_I_e REAL DEFAULT 1.0,
                weight REAL DEFAULT 1.0,
                confidence REAL DEFAULT 1.0,
                
                in_matroid_base INTEGER DEFAULT 0,
                circuit_type TEXT DEFAULT 'unknown',
                created_at REAL,
                source TEXT DEFAULT 'unknown',
                
                FOREIGN KEY (source_id) REFERENCES vertices(id),
                FOREIGN KEY (target_id) REFERENCES vertices(id)
            )
        """)
        
        # 索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vertices_name ON vertices(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vertices_category ON vertices(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON hyperedges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON hyperedges(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON hyperedges(relation)")
        
        conn.commit()
        conn.close()
    
    # ========================================================================
    # L2 κ-Snap 读取接口
    # ========================================================================
    
    def get_vertex(self, vertex_id: str) -> Optional[JSNVertex]:
        """
        L2读取：获取顶点
        
        Args:
            vertex_id: 顶点ID
        
        Returns:
            JSNVertex 或 None
        """
        # 先查缓存
        if vertex_id in self._vertex_cache:
            return self._vertex_cache[vertex_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vertices WHERE id = ?", (vertex_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        # 解析行
        v = self._row_to_vertex(row)
        self._vertex_cache[vertex_id] = v
        return v
    
    def get_vertex_by_name(self, name: str) -> Optional[JSNVertex]:
        """L2读取：按名称查询顶点"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vertices WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        return self._row_to_vertex(row)
    
    def search_concepts(self, query: str, category: str = None, limit: int = 10) -> List[JSNVertex]:
        """
        L2读取：搜索概念（模糊匹配）
        
        Args:
            query: 搜索关键词
            category: 类别过滤
            limit: 返回数量限制
        
        Returns:
            顶点列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute(
                "SELECT * FROM vertices WHERE name LIKE ? AND category = ? LIMIT ?",
                (f"%{query}%", category, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM vertices WHERE name LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_vertex(row) for row in rows]
    
    def get_k_hop(self, vertex_id: str, k: int = 2, max_results: int = 100) -> Dict[str, Any]:
        """
        L2读取：k-hop子图查询
        
        返回以vertex_id为中心的k-hop子图（顶点+边）
        
        Args:
            vertex_id: 中心顶点ID
            k: hop数
            max_results: 最大返回顶点数
        
        Returns:
            {
                'vertices': [JSNVertex, ...],
                'edges': [JSNHyperEdge, ...],
                'center': vertex_id,
                'k': k
            }
        """
        visited = {vertex_id}
        vertices = []
        edges = []
        frontier = {vertex_id}
        
        for hop in range(k):
            if len(visited) >= max_results:
                break
            
            next_frontier = set()
            for vid in frontier:
                # 查询出边
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM hyperedges WHERE source_id = ? OR target_id = ?",
                    (vid, vid)
                )
                edge_rows = cursor.fetchall()
                conn.close()
                
                for erow in edge_rows:
                    e = self._row_to_edge(erow)
                    edges.append(e)
                    
                    # 添加邻居
                    neighbor = e.target_id if e.source_id == vid else e.source_id
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        
                        v = self.get_vertex(neighbor)
                        if v:
                            vertices.append(v)
            
            frontier = next_frontier
        
        return {
            'vertices': vertices[:max_results],
            'edges': edges,
            'center': vertex_id,
            'k': k,
            'total_vertices': len(visited),
        }
    
    def get_hyperedges(self, source_id: str = None, target_id: str = None, 
                      relation: str = None) -> List[JSNHyperEdge]:
        """
        L2读取：查询超边
        
        Args:
            source_id: 源顶点过滤
            target_id: 目标顶点过滤
            relation: 关系类型过滤
        
        Returns:
            超边列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if source_id:
            conditions.append("source_id = ?")
            params.append(source_id)
        if target_id:
            conditions.append("target_id = ?")
            params.append(target_id)
        if relation:
            conditions.append("relation = ?")
            params.append(relation)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        cursor.execute(f"SELECT * FROM hyperedges {where_clause}", params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_edge(row) for row in rows]
    
    # ========================================================================
    # L4 Sleep-Step 写入接口
    # ========================================================================
    
    def add_vertex(self, vertex: JSNVertex) -> bool:
        """
        L4写入：添加顶点
        
        Args:
            vertex: JSNVertex对象
        
        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO vertices VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                vertex.id, vertex.name, vertex.category,
                vertex.phi_b0, vertex.phi_b1, vertex.phi_b2, vertex.phi_b3,
                vertex.phi_b4, vertex.phi_b5, vertex.phi_b6, vertex.phi_b7,
                vertex.existence_I_e, vertex.created_at, vertex.updated_at,
                vertex.source, vertex.confidence
            ))
            
            conn.commit()
            conn.close()
            
            # 更新缓存
            self._vertex_cache[vertex.id] = vertex
            self._update_stats()
            
            return True
        except Exception as e:
            print(f"[JSNStore] Error adding vertex: {e}")
            return False
    
    def add_hyperedge(self, edge: JSNHyperEdge) -> bool:
        """
        L4写入：添加超边
        
        Args:
            edge: JSNHyperEdge对象
        
        Returns:
            是否成功
        """
        try:
            # 确保源和目标顶点存在
            if self.get_vertex(edge.source_id) is None:
                print(f"[JSNStore] Warning: source vertex {edge.source_id} not found")
                return False
            if self.get_vertex(edge.target_id) is None:
                print(f"[JSNStore] Warning: target vertex {edge.target_id} not found")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO hyperedges VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                edge.id, edge.source_id, edge.target_id, edge.relation,
                edge.asym, edge.existence_I_e, edge.weight, edge.confidence,
                int(edge.in_matroid_base), edge.circuit_type,
                edge.created_at, edge.source
            ))
            
            conn.commit()
            conn.close()
            
            # 更新缓存
            self._edge_cache[edge.id] = edge
            self._update_stats()
            
            return True
        except Exception as e:
            print(f"[JSNStore] Error adding hyperedge: {e}")
            return False
    
    def save_macro(self, macro_name: str, description: str, 
                  related_vertices: List[str], asym: float = 0.5) -> bool:
        """
        L4 Sleep-Step写入：保存Macro（成功P*的抽象）
        
        Macro是一个特殊顶点，通过超边连接到相关概念
        
        Args:
            macro_name: Macro名称
            description: 描述
            related_vertices: 相关顶点ID列表
            asym: 手性（>0表示允许变体）
        
        Returns:
            是否成功
        """
        # 创建Macro顶点
        macro_vertex = JSNVertex(
            id=f"macro_{macro_name}",
            name=macro_name,
            category="macro",
            phi_b0=1.0,  # 实部（太一）
            phi_b1=asym,  # 虚部（手性）
            source="sleep_step"
        )
        
        if not self.add_vertex(macro_vertex):
            return False
        
        # 创建超边连接到相关概念
        for i, vid in enumerate(related_vertices):
            edge = JSNHyperEdge(
                id=f"macro_edge_{macro_name}_{i}",
                source_id=macro_vertex.id,
                target_id=vid,
                relation="macro_related",
                asym=asym,
                source="sleep_step"
            )
            if not self.add_hyperedge(edge):
                return False
        
        print(f"[JSNStore] Macro '{macro_name}' saved with {len(related_vertices)} related concepts")
        return True
    
    # ========================================================================
    # κ-Gate 拟阵贪心剪枝
    # ========================================================================
    
    def k_gate_prune(self, theta_dead: float = 0.1) -> Dict[str, Any]:
        """
        κ-Gate拟阵贪心剪枝算法（Union-Find版本）
        
        使用真正的拟阵回路检测（Union-Find）替代简化版。
        保证结果是最优基（最大权重独立集）。
        
        算法流程：
          1. 过滤掉 I(e) < θ_dead 的死零超边
          2. 按 I(e) + α*asym 降序排序（考虑八元数手性）
          3. Union-Find贪心加入独立集（回路检测）
          4. 区分 MUS-Circuit (asym≠0) 和 Paradox-Circuit (asym=0)
        
        Args:
            theta_dead: 死零阈值
        
        Returns：
            {
                'pruned_count': 剪枝数量,
                'base_size': 拟阵基大小,
                'I_e_retention': I(e)保留率,
                'circuits': {'MUS':..., 'Paradox':...}
            }
        """
        print(f"\n[κ-Gate Pruning] Starting (Union-Find matroid version)... theta_dead={theta_dead}")
        
        # 使用MatroidPruner进行真正的拟阵剪枝
        try:
            from jsn_matroid import MatroidPruner
            pruner = MatroidPruner(self.db_path)
            result = pruner.prune(theta_dead)
            
            # 验证拟阵性质
            verification = pruner.verify_matroid_properties()
            if not verification['is_valid_matroid']:
                print("  ⚠️ Warning: Matroid property violated!")
            
            print(f"  ✅ Pruned: {result['pruned_count']}, Base size: {result['base_size']}")
            print(f"  ✅ Matroid valid: {verification['is_valid_matroid']}")
            
            return result
            
        except ImportError:
            print("  ⚠️ jsn_matroid not available, using simplified version")
            # 回退到简化版（原有实现）
            return self._k_gate_prune_simplified(theta_dead)
        
        print(f"  Pruned: {result['pruned_count']}, Base size: {result['base_size']}")
        print(f"  I(e) retention: {I_e_retention:.2%}")
        print(f"  Circuits: MUS={mus_circuits}, Paradox={paradox_circuits}")
        
        self.stats['matroid_base_size'] = result['base_size']
        self.stats['I_e_pruning_ratio'] = I_e_retention
        
        return result
    
    # ========================================================================
    # EML 快照（序列化/反序列化）
    # ========================================================================
    
    def save_eml_snapshot(self, snapshot_name: str = "latest") -> str:
        """
        保存EML二进制快照
        
        Args:
            snapshot_name: 快照名称
        
        Returns:
            EML文件路径
        """
        eml_path = os.path.join(self.eml_dir, f"{snapshot_name}.eml")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        with open(eml_path, 'wb') as f:
            # 文件头
            header = struct.pack('<IIQ', 0x4A534E, 1, int(time.time()))
            f.write(header)
            
            # 顶点数据
            cursor.execute("SELECT * FROM vertices")
            vertices = [self._row_to_vertex(row) for row in cursor.fetchall()]
            f.write(struct.pack('<I', len(vertices)))
            for v in vertices:
                f.write(v.to_eml_bytes())
            
            # 超边数据
            cursor.execute("SELECT * FROM hyperedges WHERE in_matroid_base = 1")
            edges = [self._row_to_edge(row) for row in cursor.fetchall()]
            f.write(struct.pack('<I', len(edges)))
            for e in edges:
                f.write(e.to_eml_bytes())
        
        conn.close()
        
        print(f"[JSNStore] EML snapshot saved: {eml_path}")
        print(f"  Vertices: {len(vertices)}, Edges: {len(edges)}")
        
        return eml_path
    
    def load_eml_snapshot(self, snapshot_name: str = "latest") -> bool:
        """
        加载EML二进制快照
        
        Args:
            snapshot_name: 快照名称
        
        Returns:
            是否成功
        """
        eml_path = os.path.join(self.eml_dir, f"{snapshot_name}.eml")
        
        if not os.path.exists(eml_path):
            print(f"[JSNStore] Snapshot not found: {eml_path}")
            return False
        
        with open(eml_path, 'rb') as f:
            # 文件头
            magic, version, timestamp = struct.unpack('<IIQ', f.read(16))
            assert magic == 0x4A534E, "Invalid EML file"
            
            # 顶点数据
            vertex_count = struct.unpack('<I', f.read(4))[0]
            for i in range(vertex_count):
                data = f.read(80)
                v = JSNVertex.from_eml_bytes(data, f"v_{i}")
                self.add_vertex(v)
            
            # 超边数据
            edge_count = struct.unpack('<I', f.read(4))[0]
            for i in range(edge_count):
                data = f.read(32)
                e = JSNHyperEdge.from_eml_bytes(data, f"e_{i}")
                self.add_hyperedge(e)
        
        print(f"[JSNStore] EML snapshot loaded: {eml_path}")
        return True
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _row_to_vertex(self, row: tuple) -> JSNVertex:
        """SQLite行 → JSNVertex"""
        return JSNVertex(
            id=row[0], name=row[1], category=row[2],
            phi_b0=row[3], phi_b1=row[4], phi_b2=row[5], phi_b3=row[6],
            phi_b4=row[7], phi_b5=row[8], phi_b6=row[9], phi_b7=row[10],
            existence_I_e=row[11], created_at=row[12], updated_at=row[13],
            source=row[14], confidence=row[15]
        )
    
    def _row_to_edge(self, row: tuple) -> JSNHyperEdge:
        """SQLite行 → JSNHyperEdge"""
        return JSNHyperEdge(
            id=row[0], source_id=row[1], target_id=row[2], relation=row[3],
            asym=row[4], existence_I_e=row[5], weight=row[6], confidence=row[7],
            in_matroid_base=bool(row[8]), circuit_type=row[9],
            created_at=row[10], source=row[11]
        )
    
    def _update_stats(self):
        """更新统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM vertices")
            self.stats['vertex_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM hyperedges")
            self.stats['edge_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM hyperedges WHERE in_matroid_base = 1")
            self.stats['matroid_base_size'] = cursor.fetchone()[0]
            
            conn.close()
        except sqlite3.OperationalError:
            # 表不存在（新数据库）
            self.stats['vertex_count'] = 0
            self.stats['edge_count'] = 0
            self.stats['matroid_base_size'] = 0
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        self._update_stats()
        return self.stats.copy()
    
    def clear(self):
        """清空数据库（测试用）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM hyperedges")
        cursor.execute("DELETE FROM vertices")
        conn.commit()
        conn.close()
        
        self._vertex_cache.clear()
        self._edge_cache.clear()
        self._update_stats()


# ============================================================================
# 测试函数
# ============================================================================

def test_jsn_store():
    """测试JSN存储子系统"""
    print("=" * 60)
    print("测试 L1 JSN 存储子系统")
    print("=" * 60)
    
    # 创建存储（使用临时文件数据库）
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    store = JSNStore(db_path=db_path)
    
    try:
        # 测试1：添加顶点
        print("\n1. 添加顶点...")
        v1 = JSNVertex(id="v1", name="player", category="entity", phi_b0=1.0, phi_b1=0.5)
        v2 = JSNVertex(id="v2", name="goal", category="entity", phi_b0=1.0, phi_b2=0.3)
        v3 = JSNVertex(id="v3", name="wall", category="entity", phi_b0=0.8)
        
        assert store.add_vertex(v1)
        assert store.add_vertex(v2)
        assert store.add_vertex(v3)
        print(f"  ✅ 添加3个顶点")
        
        # 测试2：添加超边
        print("\n2. 添加超边...")
        e1 = JSNHyperEdge(id="e1", source_id="v1", target_id="v2", relation="move_to", asym=0.5)
        e2 = JSNHyperEdge(id="e2", source_id="v1", target_id="v3", relation="blocked_by", asym=0.0)
        
        assert store.add_hyperedge(e1)
        assert store.add_hyperedge(e2)
        print(f"  ✅ 添加2条超边")
        
        # 测试3：κ-Gate剪枝
        print("\n3. κ-Gate拟阵贪心剪枝...")
        result = store.k_gate_prune(theta_dead=0.5)
        assert result['base_size'] > 0
        print(f"  ✅ 剪枝完成，基大小: {result['base_size']}")
        
        # 测试4：k-hop查询
        print("\n4. k-hop子图查询...")
        subgraph = store.get_k_hop("v1", k=2)
        assert len(subgraph['vertices']) > 0
        print(f"  ✅ 2-hop子图: {subgraph['total_vertices']} 个顶点")
        
        # 测试5：搜索概念
        print("\n5. 搜索概念...")
        results = store.search_concepts("player")
        assert len(results) > 0
        print(f"  ✅ 找到 {len(results)} 个匹配结果")
        
        # 测试6：保存Macro（L4 Sleep-Step）
        print("\n6. 保存Macro（L4 Sleep-Step）...")
        assert store.save_macro("reach_goal", "Move player to goal", ["v1", "v2"])
        print(f"  ✅ Macro保存成功")
        
        # 测试7：EML快照
        print("\n7. EML快照...")
        eml_path = store.save_eml_snapshot("test")
        assert os.path.exists(eml_path)
        print(f"  ✅ EML快照保存成功: {eml_path}")
        
        # 统计
        stats = store.get_stats()
        print(f"\n📊 统计信息:")
        print(f"  顶点数: {stats['vertex_count']}")
        print(f"  超边数: {stats['edge_count']}")
        print(f"  拟阵基大小: {stats['matroid_base_size']}")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！L1 JSN存储子系统可用。")
        print("=" * 60)
        
    finally:
        # 清理
        store.clear()
        os.unlink(db_path)
    
    return store


if __name__ == "__main__":
    test_jsn_store()
