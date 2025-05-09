import React, { useState } from 'react';
import { ThemeProvider, createTheme, GlobalStyles } from '@mui/material';
import DashboardLayout from './DashboardLayout';
import PortfolioDashboard from './components/PortfolioDashboard';
import RecsTab from './components/RecsTab';
import NewsTab from './components/NewsTab';


export default function App() {
  const [tab, setTab] = useState(0);
  const [dark, setDark] = useState(false);

  const theme = createTheme({
    palette: {
      mode: dark ? 'dark' : 'light',
      // 1️⃣ Use solid colors here
      primary: {
        main: dark ? '#203a43' : '#4ca2cd',  // pick a representative shade
        contrastText: '#fff'
      },
      secondary: {
        main: '#f50057',
        contrastText: '#fff'
      },
      background: {
        default: dark ? '#121212' : '#f0f2f5',
        paper:   dark ? '#1e1e1e' : '#ffffff',
      },
      text: {
        primary: dark ? '#e0e0e0' : '#212121',
        secondary: dark ? '#a0a0a0' : '#555555',
      }
    },
    typography: {
      fontFamily: '"Inter", sans-serif',
      h5: { fontWeight: 600 },
      h6: { fontWeight: 500 },
      subtitle1: { fontWeight: 500 },
      body1: { fontWeight: 400, fontSize: '1rem' },
    },
    shape: { borderRadius: 16 },
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            // 2️⃣ Move your gradient here—this is valid CSS
            background: dark
              ? 'linear-gradient(90deg, #0f2027, #203a43, #2c5364)'
              : 'linear-gradient(90deg, #67b26f, #4ca2cd)',
          }
        }
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            boxShadow: dark
              ? '0 8px 16px rgba(0,0,0,0.6)'
              : '0 8px 16px rgba(0,0,0,0.1)',
            transition: 'transform .2s, box-shadow .2s'
          }
        }
      }
    }
  });

  const renderTab = () => {
    switch(tab) {
      case 0:
        return <PortfolioDashboard />;
      case 1:
        return <RecsTab />;
      case 2:
        return <NewsTab />;
      default:
        return <PortfolioDashboard />;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyles styles={{
        body: {
          margin: 0,
          padding: 0,
          fontFamily: `"Inter", sans-serif`,
          backgroundColor: theme.palette.background.default,
        }
      }} />

      <DashboardLayout
        currentTab={tab}
        onTabChange={setTab}
        darkMode={dark}
        toggleDark={() => setDark(!dark)}
      >
        {renderTab()}
      </DashboardLayout>
    </ThemeProvider>
  );
}
