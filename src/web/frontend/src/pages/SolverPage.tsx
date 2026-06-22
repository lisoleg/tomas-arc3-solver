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
  Collapse,
  FormControlLabel,
  Switch,
  Grid,
  Divider,
} from '@mui/material';
import { ExpandMore, ExpandLess, Analytics } from '@mui/icons-material';
import { solverApi, SolverMode } from '../api/client';
import KappaSnapTree from '../components/KappaSnapTree';
import GaussExFiber from '../components/GaussExFiber';
import PruningChart from '../components/PruningChart';

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
  const [showViz, setShowViz] = useState<boolean>(false);
  const [psiGateEnabled, setPsiGateEnabled] = useState<boolean>(true);
  const [aegisEnabled, setAegisEnabled] = useState<boolean>(false);
  const [causalPriorEnabled, setCausalPriorEnabled] = useState<boolean>(false);

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
    setShowViz(false);

    try {
      const configOverrides: any = {
        'psi_gate.enabled': psiGateEnabled,
        'aegis.enabled': aegisEnabled,
        'causal_prior.enabled': causalPriorEnabled,
      };

      const result = await solverApi.runSolver({
        input_path: inputPath,
        output_path: outputPath,
        mode: selectedMode,
        config_overrides: configOverrides,
      });

      setTaskId(result.task_id);
      setStatus('running');

      // Connect SSE progress stream
      const eventSource = solverApi.getProgress(result.task_id);

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'started') {
          setLogs((prev) => [...prev, `[Started] ${data.message || 'Task started'}`]);
        } else if (data.type === 'init') {
          setLogs((prev) => [...prev, `[Init] ${data.message}`]);
        } else if (data.type === 'progress') {
          setProgress(data.progress);
          setLogs((prev) => [...prev, `[Step ${data.step}/${data.total}] ${data.message}`]);
        } else if (data.type === 'done') {
          setStatus('completed');
          setProgress(100);
          setLogs((prev) => [...prev, `[Done] ${data.result?.message || 'Task completed'}`]);
          setShowViz(true);
          eventSource.close();
        } else if (data.type === 'heartbeat') {
          // Ignore heartbeat
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
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        TOMAS Solver
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Task Configuration
            </Typography>

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

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" gutterBottom>
              Advanced Features (v2.4)
            </Typography>

            <FormControlLabel
              control={
                <Switch
                  checked={psiGateEnabled}
                  onChange={(e) => setPsiGateEnabled(e.target.checked)}
                  color="primary"
                />
              }
              label="psi-Gate Semantic Gating"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={aegisEnabled}
                  onChange={(e) => setAegisEnabled(e.target.checked)}
                  color="primary"
                />
              }
              label="AEGIS Evolution Engine"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={causalPriorEnabled}
                  onChange={(e) => setCausalPriorEnabled(e.target.checked)}
                  color="primary"
                />
              }
              label="Causal DSL Prior"
            />

            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleRun}
                disabled={!inputPath || status === 'running'}
                size="large"
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
            <Paper sx={{ p: 3, mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography variant="h6" gutterBottom>
                  Logs
                </Typography>
                <Button
                  size="small"
                  onClick={() => setLogs([])}
                >
                  Clear
                </Button>
              </Box>
              <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
                {logs.map((log, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={log}
                      primaryTypographyProps={{ variant: 'body2', fontFamily: 'monospace' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          )}
        </Grid>

        {/* Side Panel: Quick Stats */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Task Info
            </Typography>
            {taskId ? (
              <Box>
                <Typography variant="body2" color="textSecondary">
                  Task ID
                </Typography>
                <Typography variant="body1" sx={{ fontFamily: 'monospace', mb: 2 }}>
                  {taskId}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Status
                </Typography>
                <Typography variant="body1" sx={{ mb: 2 }}>
                  {status}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Mode
                </Typography>
                <Typography variant="body1" sx={{ mb: 2 }}>
                  {selectedMode}
                </Typography>
              </Box>
            ) : (
              <Typography variant="body2" color="textSecondary">
                No active task. Configure and run the solver.
              </Typography>
            )}
          </Paper>

          {showViz && taskId && (
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                <Analytics sx={{ mr: 1, verticalAlign: 'middle' }} />
                Visualizations
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Click to expand visualizations for task {taskId}
              </Typography>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => setShowViz(!showViz)}
                endIcon={showViz ? <ExpandLess /> : <ExpandMore />}
              >
                {showViz ? 'Hide' : 'Show'} Visualizations
              </Button>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Visualizations */}
      <Collapse in={showViz && !!taskId} timeout="auto">
        <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', gap: 3 }}>
          <KappaSnapTree taskId={taskId || 'demo'} />
          <GaussExFiber taskId={taskId || 'demo'} />
          <PruningChart taskId={taskId || 'demo'} />
        </Box>
      </Collapse>
    </Container>
  );
};

export default SolverPage;
