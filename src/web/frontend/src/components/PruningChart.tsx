import React, { useEffect, useState } from 'react';
import {
  Paper,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
  Grid,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
} from 'recharts';
import { vizApi, PruningStatsData } from '../api/client';

interface PruningChartProps {
  taskId: string;
}

const PruningChart: React.FC<PruningChartProps> = ({ taskId }) => {
  const [data, setData] = useState<PruningStatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadData();
  }, [taskId]);

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      const result = await vizApi.getPruningStats(taskId);
      setData(result);
    } catch (err: any) {
      setError(err.message || 'Failed to load pruning statistics');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        8-Strategy Pruning Pipeline
      </Typography>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {data && (
        <>
          {/* Summary Stats */}
          <Box sx={{ mb: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              label={`Initial: ${data.total_initial}`}
              color="default"
              size="small"
            />
            <Chip
              label={`Pruned: ${data.total_pruned}`}
              color="error"
              size="small"
            />
            <Chip
              label={`Remaining: ${data.total_remaining}`}
              color="success"
              size="small"
            />
            <Chip
              label={`Overall: ${data.overall_prune_rate}%`}
              color="primary"
              size="small"
            />
          </Box>

          <Grid container spacing={3}>
            {/* Pruning Count Bar Chart */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Candidates Pruned per Strategy
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={data.stages}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="label"
                    angle={-35}
                    textAnchor="end"
                    height={80}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis />
                  <Tooltip
                    formatter={(value: any, name: string) => {
                      if (name === 'pruned') return [value, 'Pruned'];
                      if (name === 'candidates_after') return [value, 'Remaining'];
                      return [value, name];
                    }}
                  />
                  <Legend />
                  <Bar dataKey="pruned" fill="#f44336" name="Pruned" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="candidates_after" fill="#4caf50" name="Remaining" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Grid>

            {/* Prune Rate + Cumulative Rate */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Prune Rate (%) per Strategy + Cumulative
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={data.stages}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="label"
                    angle={-35}
                    textAnchor="end"
                    height={80}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis domain={[0, 100]} />
                  <Tooltip
                    formatter={(value: any, name: string) => {
                      if (name === 'prune_rate') return [`${value}%`, 'Prune Rate'];
                      if (name === 'cumulative_rate') return [`${value}%`, 'Cumulative'];
                      return [value, name];
                    }}
                  />
                  <Legend />
                  <Bar
                    dataKey="prune_rate"
                    fill="#ff9800"
                    name="Prune Rate"
                    radius={[4, 4, 0, 0]}
                  />
                  <Line
                    type="monotone"
                    dataKey="cumulative_rate"
                    stroke="#1976d2"
                    strokeWidth={3}
                    name="Cumulative"
                    dot={{ r: 5 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Grid>

            {/* Candidate Funnel */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Candidate Funnel (Before -> After)
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <ComposedChart data={data.stages}>
                  <defs>
                    <linearGradient id="beforeGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#1976d2" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#1976d2" stopOpacity={0.2} />
                    </linearGradient>
                    <linearGradient id="afterGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4caf50" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#4caf50" stopOpacity={0.2} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="label"
                    angle={-35}
                    textAnchor="end"
                    height={80}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="candidates_before"
                    stroke="#1976d2"
                    fill="url(#beforeGrad)"
                    name="Before"
                  />
                  <Area
                    type="monotone"
                    dataKey="candidates_after"
                    stroke="#4caf50"
                    fill="url(#afterGrad)"
                    name="After"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        </>
      )}
    </Paper>
  );
};

export default PruningChart;
