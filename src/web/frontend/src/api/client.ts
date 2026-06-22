import axios, { AxiosInstance } from 'axios';

const apiClient: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface SolverMode {
  id: string;
  name: string;
  description: string;
}

export interface TaskStatus {
  task_id: string;
  status: string;
  params: any;
  result?: any;
  error?: string;
}

export interface TreeNode {
  name: string;
  level: number;
  phase: string;
  mdl: number;
  children?: TreeNode[];
  verified?: boolean;
  confidence?: number;
}

export interface SearchTreeData {
  task_id: string;
  tree: TreeNode;
  total_candidates: number;
  phase_a_passed: number;
  phase_a_failed: number;
  phase_b_verified: number;
  max_depth: number;
}

export interface FiberDemo {
  demo_id: string;
  input_shape: number[];
  output_shape: number[];
  fiber_count: number;
  intersection_size: number;
  crc32_input: string;
  crc32_output: string;
  crc32_match: boolean;
}

export interface FiberCandidate {
  candidate_id: string;
  mdl: number;
  verified_demos: { demo_id: string; pass: boolean; fiber_overlap: number; hash_match: boolean }[];
  all_pass: boolean;
  confidence: number;
}

export interface FiberVerificationData {
  task_id: string;
  demos: FiberDemo[];
  candidates: FiberCandidate[];
  total_candidates: number;
  verified: number;
  failed: number;
  verification_rate: number;
}

export interface PruningStage {
  name: string;
  label: string;
  color: string;
  candidates_before: number;
  candidates_after: number;
  pruned: number;
  prune_rate: number;
  cumulative_rate: number;
}

export interface PruningStatsData {
  task_id: string;
  total_initial: number;
  total_remaining: number;
  total_pruned: number;
  overall_prune_rate: number;
  stages: PruningStage[];
}

export interface HistoryEntry {
  task_id: string;
  input_path: string;
  mode: string;
  status: string;
  duration_sec: number;
  candidates_generated: number;
  candidates_verified: number;
  prune_rate: number;
  confidence: number;
  mdl_best: number | null;
  timestamp: string;
  psi_gate_enabled: boolean;
  aegis_enabled: boolean;
}

export interface BenchmarkData {
  psi_gate: {
    enabled: any;
    disabled: any;
  };
  aegis: {
    aegis: any;
    normal: any;
  };
}

export const solverApi = {
  getModes: async (): Promise<SolverMode[]> => {
    const response = await apiClient.get('/solver/modes');
    return response.data.modes;
  },

  runSolver: async (params: {
    input_path: string;
    output_path?: string;
    mode?: string;
    config_overrides?: any;
  }): Promise<{ task_id: string; status: string; message: string }> => {
    const response = await apiClient.post('/solver/run', params);
    return response.data;
  },

  getStatus: async (taskId: string): Promise<TaskStatus> => {
    const response = await apiClient.get(`/solver/status/${taskId}`);
    return response.data;
  },

  getProgress: (taskId: string): EventSource => {
    const eventSource = new EventSource(`/api/solver/progress/${taskId}`);
    return eventSource;
  },

  getHistory: async (): Promise<any[]> => {
    const response = await apiClient.get('/solver/history');
    return response.data.history;
  },
};

export const vizApi = {
  getSearchTree: async (taskId: string): Promise<SearchTreeData> => {
    const response = await apiClient.get(`/viz/search-tree/${taskId}`);
    return response.data;
  },

  getFiberVerification: async (taskId: string): Promise<FiberVerificationData> => {
    const response = await apiClient.get(`/viz/fiber-verification/${taskId}`);
    return response.data;
  },

  getPruningStats: async (taskId: string): Promise<PruningStatsData> => {
    const response = await apiClient.get(`/viz/pruning-stats/${taskId}`);
    return response.data;
  },

  getTaskDetail: async (taskId: string): Promise<any> => {
    const response = await apiClient.get(`/viz/task-detail/${taskId}`);
    return response.data;
  },

  getHistory: async (params?: {
    limit?: number;
    mode?: string;
    status?: string;
  }): Promise<{ history: HistoryEntry[]; total: number }> => {
    const response = await apiClient.get('/viz/history', { params });
    return response.data;
  },

  getHistoryDetail: async (taskId: string): Promise<any> => {
    const response = await apiClient.get(`/viz/history/${taskId}`);
    return response.data;
  },

  deleteHistory: async (taskId: string): Promise<any> => {
    const response = await apiClient.delete(`/viz/history/${taskId}`);
    return response.data;
  },

  getBenchmark: async (): Promise<BenchmarkData> => {
    const response = await apiClient.get('/viz/benchmark');
    return response.data;
  },
};

export default apiClient;
