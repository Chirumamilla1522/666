import React from 'react';
import { styled, useTheme } from '@mui/material/styles';
import {
  Box,
  CssBaseline,
  AppBar as MuiAppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Switch
} from '@mui/material';
import {
  Menu as MenuIcon,
  ChevronLeft as ChevronLeftIcon,
  BarChart as BarChartIcon,
  Star as StarIcon,
  Newspaper as NewspaperIcon
} from '@mui/icons-material';

const drawerWidth = 240;

const AppBar = styled(MuiAppBar, {
  shouldForwardProp: prop => prop !== 'open',
})(({ theme, open }) => ({
  zIndex: theme.zIndex.drawer + 1,
  transition: theme.transitions.create(['width','margin'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  ...(open && {
    marginLeft: drawerWidth,
    width: `calc(100% - ${drawerWidth}px)`,
    transition: theme.transitions.create(['width','margin'], {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
  }),
}));

const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'flex-end',
  padding: theme.spacing(0,1),
  ...theme.mixins.toolbar,
}));

export default function DashboardLayout({ children, onTabChange, currentTab, darkMode, toggleDark }) {
  const theme = useTheme();
  const [open, setOpen] = React.useState(true);

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline/>

      {/* Top App Bar */}
      <AppBar position="fixed" open={open}>
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={()=>setOpen(!open)}
            sx={{ mr: 2 }}
          >
            {open ? <ChevronLeftIcon/> : <MenuIcon/>}
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            StockRadar
          </Typography>
          <Switch checked={darkMode} onChange={toggleDark}/>
        </Toolbar>
      </AppBar>

      {/* Sidebar Drawer */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={open}
        sx={{
          width: drawerWidth,
          '& .MuiDrawer-paper': { width: drawerWidth, boxSizing: 'border-box' },
        }}
      >
        <DrawerHeader/>
        <Divider/>
        <List>
          {[
            { label: 'Portfolio',    icon: <BarChartIcon/>,   idx: 0 },
            { label: 'Recommendations', icon: <StarIcon/>,      idx: 1 },
            { label: 'News Impact',  icon: <NewspaperIcon/>, idx: 2 },
          ].map(item => (
            <ListItem 
              button 
              key={item.label} 
              selected={currentTab===item.idx}
              onClick={()=>onTabChange(item.idx)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label}/>
            </ListItem>
          ))}
        </List>
      </Drawer>
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow:1,
          p:3,
          bgcolor: theme.palette.background.default,
          minHeight: '100vh'
        }}
      >
        <DrawerHeader/>
        {children}
      </Box>
    </Box>
  );
}
