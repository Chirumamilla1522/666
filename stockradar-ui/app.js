import React, { useState } from 'react';
import {
  createTheme,
  ThemeProvider,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Tabs,
  Tab,
} from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';
import PortfolioTab from './components/PortfolioTab';
import RecsTab from './components/RecsTab';
import NewsTab from './components/NewsTab';

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [tab, setTab] = useState(0);

  const theme = createTheme({ palette: { mode: darkMode ? 'dark' : 'light' } });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="sticky">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            📊 StockRadar
          </Typography>
          <IconButton color="inherit" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? <Brightness7 /> : <Brightness4 />}
          </IconButton>
        </Toolbar>
        <Tabs value={tab} onChange={(_,v)=>setTab(v)} centered>
          <Tab label="Portfolio" />
          <Tab label="Recommendations" />
          <Tab label="News Impact" />
        </Tabs>
      </AppBar>

      {tab === 0 && <PortfolioTab />}
      {tab === 1 && <RecsTab />}
      {tab === 2 && <NewsTab />}
    </ThemeProvider>
  );
}
