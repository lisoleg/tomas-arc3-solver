"""
L1 JSN 存储子系统 - 接口契约文档
======================================

定义 L2 κ-Snap 和 L4 Sleep-Step 如何与 L1 JSN 存储子系统交互。

架构位置：
  L2 κ-Snap (读) → JSNStore (查询) → L4 Sleep-Step (写)
  
NAR-Net (L5) 不直接访问 L1 JSN。

Author: TOMAS Team
Version: 0.1.0
"""

# ============================================================================
# L2 κ-Snap 读取接口契约
# ============================================================================

def l2_k_snap_interface_contract():
    """
    L2 κ-Snap 读取接口契约
    
    职责：
      - 从 JSNStore 查询概念（顶点）
      - 从 JSNStore 查询关系（超边）
      - 获取 k-hop 子图用于推理
    
    接口：
      - get_vertex(vertex_id) → JSNVertex
      - get_vertex_by_name(name) → JSNVertex
      - search_concepts(query, category) → [JSNVertex]
      - get_k_hop(vertex_id, k) → {vertices, edges}
      - get_hyperedges(source_id, target_id, relation) → [JSNHyperEdge]
    """
    pass


# ============================================================================
# L4 Sleep-Step 写入接口契约
# ============================================================================

def l4_sleep_step_interface_contract():
    """
    L4 Sleep-Step 写入接口契约
    
    职责：
      - 将成功的 P* (程序节点) 抽象为 Macro
      - 保存 Macro 到 JSNStore (作为特殊顶点)
      - 创建超边连接 Macro 与相关概念
    
    接口：
      - add_vertex(vertex) → bool
      - add_hyperedge(edge) → bool
      - save_macro(macro_name, description, related_vertices) → bool
    """
    pass


# ============================================================================
# L5 NAR-Net 不直接访问 L1 JSN
# ============================================================================

def l5_nar_net_interface_contract():
    """
    L5 NAR-Net 接口契约
    
    职责：
      - NAR-Net 是 L5 的计算引擎（grid → octonion → grid）
      - 不直接访问 L1 JSN 存储
      - 通过 L2 κ-Snap 间接使用 JSN 中的概念
    
    禁止操作：
      - ❌ 不直接调用 JSNStore.get_vertex()
      - ❌ 不直接调用 JSNStore.add_hyperedge()
      - ✅ 通过 L2 κ-Snap 的搜索结果获取概念
    """
    pass


# ============================================================================
# 集成示例
# ============================================================================

def example_l2_k_snap_with_jsn():
    """
    示例：L2 κ-Snap 使用 JSNStore 查询概念
    
    场景：κ-Snap搜索需要查找已知概念
    """
    from jsn_store import JSNStore, JSNVertex
    
    # 初始化存储
    store = JSNStore(db_path="tomas_jsn.db")
    
    # L2读取：搜索相关概念
    concepts = store.search_concepts("player", category="entity", limit=5)
    
    for concept in concepts:
        # 获取概念的八元数φ场
        phi = concept.to_octonion()
        
        # 获取概念的 k-hop 子图
        subgraph = store.get_k_hop(concept.id, k=2)
        
        # 用于 κ-Snap 推理...
        print(f"Concept: {concept.name}, φ-norm: {np.linalg.norm(phi):.4f}")
    
    return concepts


def example_l4_sleep_step_with_jsn():
    """
    示例：L4 Sleep-Step 保存 Macro 到 JSNStore
    
    场景：Sleep-Step 将成功的 P* 抽象为 Macro
    """
    from jsn_store import JSNStore
    
    # 初始化存储
    store = JSNStore(db_path="tomas_jsn.db")
    
    # L4写入：保存 Macro
    success = store.save_macro(
        macro_name="solve_ls20_level0",
        description="Successfully solved ls20 level 0 with 5 steps",
        related_vertices=["player", "goal", "wall"],
        asym=0.3  # 允许变体
    )
    
    if success:
        print("Macro saved to L1 JSN")
        
        # 运行 κ-Gate 剪枝（可选）
        result = store.k_gate_prune(theta_dead=0.1)
        print(f"Pruning result: {result}")
    
    return success


def example_l5_nar_net_indirect_access():
    """
    示例：L5 NAR-Net 间接使用 JSN（通过 L2）
    
    正确的流程：
      L5 NAR-Net → L2 κ-Snap (查询) → L1 JSN (存储)
    """
    from jsn_store import JSNStore
    
    # 错误做法 ❌
    # store = JSNStore(...)
    # phi = store.get_vertex("player").to_octonion()  # NAR-Net不应直接访问
    
    # 正确做法 ✅
    # 1. L2 κ-Snap 查询相关概念
    store = JSNStore(db_path="tomas_jsn.db")
    concepts = store.search_concepts("player")
    
    # 2. L2 将查询结果传递给 L5
    concept_features = []
    for c in concepts:
        phi = c.to_octonion()
        concept_features.append(phi)
    
    # 3. L5 NAR-Net 使用概念特征进行推理
    # nar_output = nar_net.forward(grid, concept_features)
    
    print("✅ L5 indirectly accessed L1 via L2")


# ============================================================================
# 主函数（示例运行）
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("L1 JSN 接口契约示例")
    print("=" * 60)
    
    print("\n1. L2 κ-Snap 读取示例...")
    example_l2_k_snap_with_jsn()
    
    print("\n2. L4 Sleep-Step 写入示例...")
    example_l4_sleep_step_with_jsn()
    
    print("\n3. L5 NAR-Net 间接访问示例...")
    example_l5_nar_net_indirect_access()
    
    print("\n" + "=" * 60)
    print("✅ 接口契约示例运行完成")
    print("=" * 60)
