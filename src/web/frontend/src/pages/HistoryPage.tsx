import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  TextField,
  MenuItem,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { Delete, Visibility, Refresh, FilterList } from '@mui/icons-material';
import { vizApi, HistoryEntry } from '../api/client';
import KappaSnapTree from '../components/KappaSnapTree';
import GaussExFiber from '../components/GaussExFiber';
import PruningChart from '../components/PruningChart';

const HistoryPage: React.FC = () => {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filterMode, setFilterMode] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [detailTaskId, setDetailTaskId] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const params: any = { limit: 50 };
      if (filterMode) params.mode = filterMode;
      if (filterStatus) params.status = filterStatus;
      const result = await vizApi.getHistory(params);
      setHistory(result.history);
    } catch (err: any) {
      setError(err.message || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (taskId: string) => {
    try {
      await vizApi.deleteHistory(taskId);
      setHistory(history.filter((h) => h.task_id !== taskId));
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleViewDetail = (taskId: string) => {
    setDetailTaskId(taskId);
  };

  const handleCloseDetail = () => {
    setDetailTaskId(null);
  };

  const statusColor = (status: string): 'success' | 'error' | 'warning' | 'default' => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'timeout': return 'warning';
      default: return 'default';
    }
  };

  const modeColor = (mode: string): 'primary' | 'secondary' | 'info' | 'default' => {
    switch (mode) {
      case 'video': return 'primary';
      case 'bayesian': return 'secondary';
      case 'fusion': return 'info';
      default: return 'default';
    }
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Task History
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Filter Bar */}
      <Paper sx={{ p: 2, mb: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
        <FilterList color="action" />
        <TextField
          select
          label="Mode"
          value={filterMode}
          onChange={(e) => setFilterMode(e.target.value)}
          size="small"
          sx={{ minWidth: 120 }}
        >
          <MenuItem value="">All</MenuItem>
          <MenuItem value="video">Video</MenuItem>
          <MenuItem value="bayesian">Bayesian</MenuItem>
          <MenuItem value="fusion">Fusion</MenuItem>
          <MenuItem value="auto">Auto</MenuItem>
        </TextField>
        <TextField
          select
          label="Status"
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          size="small"
          sx={{ minWidth: 120 }}
        >
          <MenuItem value="">All</MenuItem>
          <MenuItem value="completed">Completed</MenuItem>
          <MenuItem value="failed">Failed</MenuItem>
          <MenuItem value="timeout">Timeout</MenuItem>
        </TextField>
        <Button
          variant="contained"
          startIcon={<Refresh />}
          onClick={loadHistory}
          size="small"
        >
          Refresh
        </Button>
      </Paper>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Task ID</TableCell>
                <TableCell>Input</TableCell>
                <TableCell>Mode</TableCell>
                <TableCell>Status</TableCell>
                <TableCell align="right">Duration</TableCell>
                <TableCell align="right">Candidates</TableCell>
                <TableCell align="right">Verified</TableCell>
                <TableCell align="right">Prune%</TableCell>
                <TableCell align="right">Confidence</TableCell>
                <TableCell align="right">MDL</TableCell>
                <TableCell>psi-Gate</TableCell>
                <TableCell>AEGIS</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {history.map((entry) => (
                <TableRow key={entry.task_id} hover>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {entry.task_id}
                  </TableCell>
                  <TableCell sx={{ fontSize: 11 }}>
                    {entry.input_path.split('/').pop()}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={entry.mode}
                      color={modeColor(entry.mode)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={entry.status}
                      color={statusColor(entry.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="right">
                    {entry.duration_sec.toFixed(1)}s
                  </TableCell>
                  <TableCell align="right">
                    {entry.candidates_generated}
                  </TableCell>
                  <TableCell align="right">
                    {entry.candidates_verified}
                  </TableCell>
                  <TableCell align="right">
                    {entry.prune_rate}%
                  </TableCell>
                  <TableCell align="right">
                    {entry.confidence > 0
                      ? `${(entry.confidence * 100).toFixed(0)}%`
                      : '-'}
                  </TableCell>
                  <TableCell align="right">
                    {entry.mdl_best ?? '-'}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={entry.psi_gate_enabled ? 'ON' : 'OFF'}
                      color={entry.psi_gate_enabled ? 'success' : 'default'}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={entry.aegis_enabled ? 'ON' : 'OFF'}
                      color={entry.aegis_enabled ? 'success' : 'default'}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell sx={{ fontSize: 11 }}>
                    {new Date(entry.timestamp).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={() => handleViewDetail(entry.task_id)}
                      title="View Details"
                    >
                      <Visibility fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleDelete(entry.task_id)}
                      title="Delete"
                      color="error"
                    >
                      <Delete fontSize="small" />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {history.length === 0 && (
            <Typography variant="body2" sx={{ p: 3, textAlign: 'center' }}>
              No history entries found
            </Typography>
          )}
        </TableContainer>
      )}

      {/* Detail Dialog */}
      <Dialog
        open={!!detailTaskId}
        onClose={handleCloseDetail}
        maxWidth="lg"
        fullWidth
        scroll="paper"
      >
        <DialogTitle>
          Task Detail: {detailTaskId}
        </DialogTitle>
        <DialogContent>
          {detailTaskId && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
              <KappaSnapTree taskId={detailTaskId} />
              <GaussExFiber taskId={detailTaskId} />
              <PruningChart taskId={detailTaskId} />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDetail}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default HistoryPage;
