import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Container, Button, CircularProgress, Box } from '@mui/material';

// 懒加载页面组件 —— 每个页面单独打包成 chunk
const SolverPage  = lazy(() => import('./pages/SolverPage'));
const HistoryPage = lazy(() => import('./pages/HistoryPage'));

/** 路由级懒加载的 fallback */
const PageLoading: React.FC = () => (
  <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
    <CircularProgress />
  </Box>
);

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            TOMAS ARC-AGI-3 Solver Dashboard
          </Typography>
          <Button color="inherit" component={Link} to="/">
            Solver
          </Button>
          <Button color="inherit" component={Link} to="/history">
            History
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Suspense fallback={<PageLoading />}>
          <Routes>
            <Route path="/" element={<SolverPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </Suspense>
      </Container>
    </BrowserRouter>
  );
};

export default App;
