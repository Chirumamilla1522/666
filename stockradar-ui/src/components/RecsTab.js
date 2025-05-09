// src/components/RecsTab.js

import React, { useEffect, useState, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Stack,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  MenuItem,
  Select,
  InputLabel,
  FormControl,
  Button,
  IconButton,
  Chip,
  Switch,
  FormControlLabel,
  useTheme,
  LinearProgress
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TableChart as TableIcon,
  ViewModule as GridIcon
} from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import axios from 'axios';

export default function RecsTab() {
  const theme = useTheme();

  const [data, setData]                 = useState([]);
  const [filter, setFilter]             = useState('');
  const [sectorFilter, setSectorFilter] = useState('');
  const [sortField, setSortField]       = useState('similarity_score');
  const [sortDir, setSortDir]           = useState('desc');
  const [viewTable, setViewTable]       = useState(true);

  const fetchData = async () => {
    try {
      const res = await axios.get('/recommendations');
      setData(res.data);
    } catch {
      setData([]);
    }
  };

  useEffect(() => {
    let isMounted = true;
    
    const loadData = async () => {
      try {
        const res = await axios.get('/recommendations');
        if (isMounted) {
          setData(res.data);
        }
      } catch {
        if (isMounted) {
          setData([]);
        }
      }
    };
    
    loadData();
    
    return () => {
      isMounted = false;
    };
  }, []);

  const sectors = useMemo(() => {
    const all = data.map(r => r.sector).filter(Boolean);
    return [...new Set(all)].sort();
  }, [data]);

  const displayed = useMemo(() => {
    return data
      .filter(r =>
        r.ticker.includes(filter.toUpperCase()) ||
        r.name.toLowerCase().includes(filter.toLowerCase())
      )
      .filter(r => (sectorFilter ? r.sector === sectorFilter : true))
      .sort((a, b) => {
        let cmp = 0;
        if (['ticker', 'name', 'sector'].includes(sortField)) {
          cmp = a[sortField].localeCompare(b[sortField]);
        } else {
          cmp = a[sortField] - b[sortField];
        }
        return sortDir === 'asc' ? cmp : -cmp;
      });
  }, [data, filter, sectorFilter, sortField, sortDir]);

  const columns = [
    { field: 'ticker', headerName: 'Ticker', width: 100 },
    { field: 'name', headerName: 'Name', flex: 1 },
    { field: 'sector', headerName: 'Sector', width: 160 },
    {
      field: 'similarity_score',
      headerName: 'Score',
      width: 120,
      type: 'number',
      valueFormatter: ({ value }) => value.toFixed(3)
    }
  ];

  return (
    <Box p={2}>
      {/* Controls Panel */}
      <Paper elevation={2} sx={{ p: 2, mb: 2, borderRadius: 2 }}>
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          alignItems="center"
        >
          <TextField
            label="Search"
            size="small"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel>Sector</InputLabel>
            <Select
              label="Sector"
              value={sectorFilter}
              onChange={e => setSectorFilter(e.target.value)}
            >
              <MenuItem value="">All</MenuItem>
              {sectors.map(sec => (
                <MenuItem key={sec} value={sec}>
                  {sec}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              label="Sort By"
              value={sortField}
              onChange={e => setSortField(e.target.value)}
            >
              <MenuItem value="ticker">Ticker</MenuItem>
              <MenuItem value="name">Name</MenuItem>
              <MenuItem value="sector">Sector</MenuItem>
              <MenuItem value="similarity_score">Score</MenuItem>
            </Select>
          </FormControl>
          <IconButton
            onClick={() => setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))}
          >
            <RefreshIcon
              sx={{
                transform:
                  sortDir === 'asc' ? 'rotate(0deg)' : 'rotate(180deg)'
              }}
            />
          </IconButton>
          <FormControlLabel
            control={
              <Switch
                checked={viewTable}
                onChange={e => setViewTable(e.target.checked)}
              />
            }
            label={viewTable ? <TableIcon /> : <GridIcon />}
          />
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={fetchData}
          >
            Refresh
          </Button>
        </Stack>
      </Paper>

      {/* Table or Card Grid */}
      {viewTable ? (
        <Paper elevation={1} sx={{ height: 500, borderRadius: 2, overflow: 'hidden' }}>
          <DataGrid
            rows={displayed.map((r, i) => ({ id: i, ...r }))}
            columns={columns}
            pageSize={10}
            rowsPerPageOptions={[10]}
            disableSelectionOnClick
            sx={{
              border: 0,
              '.MuiDataGrid-columnHeaders': {
                background: theme.palette.grey[100]
              },
              '.MuiDataGrid-cell': { py: 1 }
            }}
          />
        </Paper>
      ) : (
        <Grid container spacing={2}>
          {displayed.map((r, i) => {
            const scorePct = Math.round(r.similarity_score * 100);
            return (
              <Grid item xs={12} sm={6} md={4} key={i}>
                <Card
                  sx={{
                    borderRadius: 2,
                    boxShadow: 2,
                    ':hover': { boxShadow: 6 }
                  }}
                >
                  <CardHeader
                    title={r.ticker}
                    subheader={r.name}
                    action={
                      <Chip
                        label={`${(r.similarity_score).toFixed(3)}`}
                        color="primary"
                        size="small"
                      />
                    }
                  />
                  <CardContent>
                    <Typography variant="body2" color="textSecondary">
                      {r.sector}
                    </Typography>
                    <Box mt={1}>
                      <Typography variant="caption">Similarity</Typography>
                      <LinearProgress
                        variant="determinate"
                        value={scorePct}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          mt: 0.5,
                          background: theme.palette.grey[200],
                          '& .MuiLinearProgress-bar': {
                            background: theme.palette.primary.main
                          }
                        }}
                      />
                      <Typography variant="caption" align="right" display="block">
                        {scorePct}%
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
          {displayed.length === 0 && (
            <Grid item xs={12}>
              <Typography>No recommendations found.</Typography>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
}
