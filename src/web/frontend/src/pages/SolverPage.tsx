import React, { useState } from 'react';
import {
  Container,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Paper,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { solverApi, SolverMode } from '../api/client';

const SolverPage: React.FC = () => {
  const [modes, setModes] = useState<SolverMode[]>([]);
  const [selectedMode, setSelectedMode] = useState<string>('auto');
  const [inputPath, setInputPath] = useState<string>('');
  const [outputPath, setOutputPath] = useState<string>('output/result.json');
  const [taskId, setTaskId] = useState<string>('');
  const [status, setStatus] = useState<string>('');
  const [progress, setProgress] = useState<number>(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string>('');

  React.useEffect(() => {
    loadModes();
  }, []);

  const loadModes = async () => {
    try {
      const data = await solverApi.getModes();
      setModes(data);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleRun = async () => {
    setError('');
    setLogs([]);
    setProgress(0);
    setStatus('starting');

    try {
      const result = await solverApi.runSolver({
        input_path: inputPath,
        output_path: outputPath,
        mode: selectedMode,
      });

      setTaskId(result.task_id);
      setStatus('running');

      // 连接 SSE 进度流
      const eventSource = solverApi.getProgress(result.task_id);

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'started') {
          setLogs((prev) => [...prev, `Started: ${data.message || 'Task started'}`]);
        } else if (data.type === 'progress') {
          setProgress(data.progress);
          setLogs((prev) => [...prev, `Step ${data.step}/${data.total}: ${data.message}`]);
        } else if (data.type === 'done') {
          setStatus('completed');
          setLogs((prev) => [...prev, `Completed: ${data.result?.message || 'Task completed'}`]);
          eventSource.close();
        } else if (data.type === 'heartbeat') {
          // 忽略心跳
        }
      };

      eventSource.onerror = () => {
        setError('SSE connection error');
        eventSource.close();
      };
    } catch (err: any) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>
        TOMAS Solver
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ p: 3, mb: 3 }}>
        <TextField
          fullWidth
          label="Input Path"
          value={inputPath}
          onChange={(e) => setInputPath(e.target.value)}
          margin="normal"
          placeholder="data/task_001.json or data/"
        />

        <TextField
          fullWidth
          label="Output Path"
          value={outputPath}
          onChange={(e) => setOutputPath(e.target.value)}
          margin="normal"
          placeholder="output/result.json"
        />

        <FormControl fullWidth margin="normal">
          <InputLabel>Inference Mode</InputLabel>
          <Select
            value={selectedMode}
            onChange={(e) => setSelectedMode(e.target.value)}
          >
            {modes.map((mode) => (
              <MenuItem key={mode.id} value={mode.id}>
                {mode.name} - {mode.description}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Box sx={{ mt: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleRun}
            disabled={!inputPath || status === 'running'}
          >
            Run Solver
          </Button>
        </Box>
      </Paper>

      {status === 'running' && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Solving in Progress...
          </Typography>
          <LinearProgress variant="determinate" value={progress} />
          <Typography variant="body2" sx={{ mt: 1 }}>
            {progress.toFixed(0)}% complete
          </Typography>
        </Paper>
      )}

      {logs.length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Logs
          </Typography>
          <List>
            {logs.map((log, index) => (
              <ListItem key={index}>
                <ListItemText primary={log} />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
    </Container>
  );
};

export default SolverPage;
