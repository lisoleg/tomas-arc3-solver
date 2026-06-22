import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Container, Box, Button } from '@mui/material';
import SolverPage from './pages/SolverPage';

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
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Routes>
          <Route path="/" element={<SolverPage />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
};

export default App;
