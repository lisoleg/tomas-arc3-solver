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

export default apiClient;
