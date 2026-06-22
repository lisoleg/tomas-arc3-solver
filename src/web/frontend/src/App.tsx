import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Container, Button } from '@mui/material';
import SolverPage from './pages/SolverPage';
import HistoryPage from './pages/HistoryPage';

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
        <Routes>
          <Route path="/" element={<SolverPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
};

export default App;
