import React, { useEffect, useState } from 'react';
import {
  Paper,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Grid,
} from '@mui/material';
import { CheckCircle, Cancel, FiberManualRecord } from '@mui/icons-material';
import { vizApi, FiberVerificationData } from '../api/client';

interface GaussExFiberProps {
  taskId: string;
}

const GaussExFiber: React.FC<GaussExFiberProps> = ({ taskId }) => {
  const [data, setData] = useState<FiberVerificationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadData();
  }, [taskId]);

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      const result = await vizApi.getFiberVerification(taskId);
      setData(result);
    } catch (err: any) {
      setError(err.message || 'Failed to load fiber verification data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        GaussEx Fiber Verification
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
              label={`Total Candidates: ${data.total_candidates}`}
              color="default"
              size="small"
            />
            <Chip
              label={`Verified: ${data.verified}`}
              color="success"
              size="small"
            />
            <Chip
              label={`Failed: ${data.failed}`}
              color="error"
              size="small"
            />
            <Chip
              label={`Rate: ${data.verification_rate}%`}
              color="info"
              size="small"
            />
          </Box>

          {/* Verification Rate Progress Bar */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" gutterBottom>
              Verification Rate
            </Typography>
            <LinearProgress
              variant="determinate"
              value={data.verification_rate}
              color={data.verification_rate > 50 ? 'success' : 'warning'}
              sx={{ height: 12, borderRadius: 6 }}
            />
          </Box>

          <Grid container spacing={3}>
            {/* Demo Pairs */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Demo Pairs (Fiber Data)
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Demo</TableCell>
                      <TableCell>Input Shape</TableCell>
                      <TableCell>Output Shape</TableCell>
                      <TableCell>Fibers</TableCell>
                      <TableCell>Intersect</TableCell>
                      <TableCell>CRC32 Match</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.demos.map((demo) => (
                      <TableRow key={demo.demo_id}>
                        <TableCell>{demo.demo_id}</TableCell>
                        <TableCell>{demo.input_shape.join('x')}</TableCell>
                        <TableCell>{demo.output_shape.join('x')}</TableCell>
                        <TableCell>
                          <FiberManualRecord fontSize="small" color="primary" />
                          {demo.fiber_count}
                        </TableCell>
                        <TableCell>{demo.intersection_size}</TableCell>
                        <TableCell>
                          {demo.crc32_match ? (
                            <CheckCircle color="success" fontSize="small" />
                          ) : (
                            <Cancel color="error" fontSize="small" />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>

            {/* Candidate Verification Matrix */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Candidate Verification Matrix
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Candidate</TableCell>
                      <TableCell>MDL</TableCell>
                      {data.demos.map((d) => (
                        <TableCell key={d.demo_id} align="center">
                          {d.demo_id}
                        </TableCell>
                      ))}
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.candidates.slice(0, 12).map((cand) => (
                      <TableRow
                        key={cand.candidate_id}
                        sx={{
                          backgroundColor: cand.all_pass
                            ? 'rgba(76, 175, 80, 0.08)'
                            : 'inherit',
                        }}
                      >
                        <TableCell sx={{ fontFamily: 'monospace' }}>
                          {cand.candidate_id}
                        </TableCell>
                        <TableCell>{cand.mdl}</TableCell>
                        {cand.verified_demos.map((vd) => (
                          <TableCell key={vd.demo_id} align="center">
                            {vd.pass ? (
                              <CheckCircle color="success" fontSize="small" />
                            ) : (
                              <Cancel color="error" fontSize="small" />
                            )}
                          </TableCell>
                        ))}
                        <TableCell>
                          {cand.all_pass ? (
                            <Chip label={`${(cand.confidence * 100).toFixed(0)}%`} color="success" size="small" />
                          ) : (
                            <Chip label="FAIL" color="error" size="small" />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              {data.candidates.length > 12 && (
                <Typography variant="caption" color="textSecondary" sx={{ mt: 1 }}>
                  Showing 12 of {data.candidates.length} candidates
                </Typography>
              )}
            </Grid>
          </Grid>
        </>
      )}
    </Paper>
  );
};

export default GaussExFiber;
