// src/components/PortfolioDashboard.js

import React, { useEffect, useState, useMemo } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  Autocomplete,
  IconButton,
  Skeleton,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useTheme,
  MenuItem,
  FormControlLabel,
  Switch
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import { Sparklines, SparklinesLine } from 'react-sparklines';
import axios from 'axios';

export default function PortfolioDashboard() {
  const theme = useTheme();

  // Data state
  const [portfolio, setPortfolio] = useState(null);
  const [topPerf, setTopPerf] = useState(null);
  const [companies, setCompanies] = useState([]);

  // UI state
  const [filter, setFilter] = useState('');
  const [sortField, setSortField] = useState('ticker');
  const [sortDir, setSortDir] = useState('asc');
  const emptyForm = { id: '', ticker: '', quantity: '' };
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [darkMode, setDarkMode] = useState(false);

  // Fetch list of all stocks for the Autocomplete
  useEffect(() => {
    let isMounted = true;
    
    const fetchStocks = async () => {
      try {
        const res = await axios.get('/stocks');
        if (isMounted) {
          setCompanies(
            res.data.map(o => ({
              label: `${o.name} (${o.ticker})`,
              ticker: o.ticker,
              sector: o.sector
            }))
          );
        }
      } catch (error) {
        console.error("Failed to fetch stocks", error);
        if (isMounted) {
          setCompanies([]);
        }
      }
    };
    
    fetchStocks();
    
    return () => {
      isMounted = false;
    };
  }, []);

  // Load portfolio and quotes
  const loadPortfolio = async () => {
    setPortfolio(null);
    try {
      const [hRes, qRes] = await Promise.all([
        axios.get('/portfolio'),
        axios.get('/quotes')
      ]);
      setPortfolio(hRes.data.map(h => {
        const q = qRes.data.find(x => x.ticker === h.ticker) || {};
        return {
          id: h.id,
          ticker: h.ticker,
          quantity: h.quantity,
          name: q.name || h.ticker,
          price: q.price || 0,
          change: q.change || 0,
          spark: q.spark || [],
          sector: companies.find(c => c.ticker === h.ticker)?.sector || ''
        };
      }));
    } catch {
      setPortfolio([]);
    }
  };

  // Load top performers
  const loadTopPerf = () => {
    setTopPerf(null);
    axios.get('/top-performers')
      .then(res => setTopPerf(res.data))
      .catch(() => setTopPerf([]));
  };

  // Initial data fetch
  useEffect(() => {
    let isMounted = true;
    
    const fetchData = async () => {
      try {
        await loadPortfolio();
        if (isMounted) {
          const res = await axios.get('/top-performers');
          if (isMounted) {
            setTopPerf(res.data);
          }
        }
      } catch (error) {
        console.error("Error loading data", error);
        if (isMounted) {
          setPortfolio([]);
          setTopPerf([]);
        }
      }
    };
    
    fetchData();
    
    return () => {
      isMounted = false;
    };
  }, [companies]);

  const refreshAll = () => {
    loadPortfolio();
    loadTopPerf();
  };

  // Dialog handlers
  const openForm = (h = emptyForm) => {
    setForm({ ...emptyForm, ...h });
    setOpen(true);
  };
  const closeForm = () => {
    setOpen(false);
    setForm(emptyForm);
  };
  const submitForm = async () => {
    const payload = { ticker: form.ticker, quantity: +form.quantity };
    try {
      if (form.id) {
        await axios.put(`/portfolio/${form.id}`, payload);
      } else {
        await axios.post('/portfolio', payload);
      }
      closeForm();
      refreshAll();
    } catch (e) {
      console.error(e);
      alert(e.response?.data?.detail || 'Save failed');
    }
  };
  const deleteHolding = async () => {
    if (!form.id) return;
    try {
      await axios.delete(`/portfolio/${form.id}`);
      closeForm();
      refreshAll();
    } catch (e) {
      console.error(e);
      alert('Delete failed');
    }
  };

  // Filter & sort displayed holdings
  const displayed = (portfolio || [])
    .filter(q =>
      q.ticker.includes(filter.toUpperCase()) ||
      q.name.toLowerCase().includes(filter.toLowerCase())
    )
    .sort((a, b) => {
      let cmp = 0;
      if (['ticker', 'name', 'sector'].includes(sortField)) {
        cmp = a[sortField].localeCompare(b[sortField]);
      } else {
        cmp = a[sortField] - b[sortField];
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

  // Compute total portfolio sparkline
  const totalSpark = useMemo(() => {
    if (!portfolio || portfolio.length === 0) return [];
    const minLen = Math.min(...portfolio.map(h => h.spark.length));
    const data = [];
    for (let i = 0; i < minLen; i++) {
      let sum = 0;
      portfolio.forEach(h => {
        sum += (h.spark[i] || 0) * h.quantity;
      });
      data.push(parseFloat(sum.toFixed(2)));
    }
    return data;
  }, [portfolio]);

  return (
    <Box
      p={3}
      bgcolor={darkMode ? '#121212' : '#fafafa'}
      color={darkMode ? '#fff' : 'inherit'}
    >
      {/* Dark Mode Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={darkMode}
            onChange={e => setDarkMode(e.target.checked)}
          />
        }
        label="Dark Mode"
      />

      {/* Summary Card */}
      <Card
        sx={{
          mb: 3,
          borderRadius: 3,
          background: `linear-gradient(135deg, ${theme.palette.primary.main}55, ${theme.palette.secondary.main}55)`,
          color: theme.palette.getContrastText(theme.palette.primary.main),
          boxShadow: 4
        }}
      >
        <CardContent>
          <Typography variant="h6">Portfolio Summary</Typography>
          <Box display="flex" alignItems="center" mt={1} gap={4}>
            <Box>
              <Typography variant="subtitle2">Total Value</Typography>
              <Typography variant="h4">
                ${totalSpark.length
                  ? totalSpark[totalSpark.length - 1].toFixed(2)
                  : '0.00'}
              </Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2">Value Trend</Typography>
              {totalSpark.length > 1 && (
                <Sparklines data={totalSpark} width={200} height={50} margin={5}>
                  <SparklinesLine
                    style={{
                      stroke: theme.palette.primary.dark,
                      fill: theme.palette.primary.light,
                      fillOpacity: 0.2
                    }}
                  />
                </Sparklines>
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Controls */}
      <Box display="flex" gap={2} flexWrap="wrap" mb={2}>
        <TextField
          label="Search"
          size="small"
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
        <TextField
          label="Sort by"
          size="small"
          select
          value={sortField}
          onChange={e => setSortField(e.target.value)}
        >
          {['ticker', 'name', 'sector', 'quantity', 'price', 'change'].map(f => (
            <MenuItem key={f} value={f}>
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </MenuItem>
          ))}
        </TextField>
        <IconButton onClick={() => setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))}>
          <RefreshIcon
            sx={{ transform: sortDir === 'asc' ? 'rotate(0deg)' : 'rotate(180deg)' }}
          />
        </IconButton>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => openForm()}
        >
          Add Holding
        </Button>
        <IconButton onClick={refreshAll}>
          <RefreshIcon />
        </IconButton>
      </Box>

      {/* Portfolio Grid */}
      <Grid container spacing={2}>
        {displayed.length === 0 && portfolio !== null ? (
          <Grid item xs={12}>
            <Typography>No holdings.</Typography>
          </Grid>
        ) : (
          displayed.map(q => (
            <Grid item xs={12} sm={6} md={4} key={q.id}>
              <Card
                sx={{
                  borderRadius: 3,
                  boxShadow: 2,
                  '&:hover': { boxShadow: 6 }
                }}
              >
                <CardHeader
                  avatar={
                    <IconButton size="small" onClick={() => openForm(q)}>
                      <EditIcon fontSize="small" />
                    </IconButton>
                  }
                  title={`${q.ticker} • ${q.name}`}
                  subheader={q.sector}
                  action={
                    <Chip
                      label={`${q.change >= 0 ? '+' : ''}${q.change.toFixed(2)}`}
                      color={q.change >= 0 ? 'success' : 'error'}
                      size="small"
                    />
                  }
                />
                <CardContent>
                  <Box mb={1}>
                    <Typography variant="h5">
                      ${q.price.toFixed(2)}
                    </Typography>
                    <Typography variant="subtitle2" color="textSecondary">
                      Qty: {q.quantity}
                    </Typography>
                  </Box>
                  <Sparklines data={q.spark} limit={30} width={120} height={40}>
                    <SparklinesLine
                      style={{
                        stroke:
                          q.change >= 0
                            ? theme.palette.success.main
                            : theme.palette.error.main,
                        fill: 'none'
                      }}
                    />
                  </Sparklines>
                </CardContent>
              </Card>
            </Grid>
          ))
        )}
      </Grid>

      {/* Top Performers */}
      <Box mt={4} mb={2} display="flex" justifyContent="space-between">
        <Typography variant="h5">Today's Top Performers</Typography>
        <IconButton onClick={refreshAll}>
          <RefreshIcon />
        </IconButton>
      </Box>
      <Grid container spacing={2}>
        {(topPerf || []).map(p => (
          <Grid item xs={12} sm={6} md={4} key={p.ticker}>
            <Card variant="outlined" sx={{ borderRadius: 3, boxShadow: 1 }}>
              <CardContent sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="subtitle1">{p.ticker}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    {p.name}
                  </Typography>
                </Box>
                <Chip
                  label={`${p.change_percent >= 0 ? '+' : ''}${p.change_percent}%`}
                  color={p.change_percent >= 0 ? 'success' : 'error'}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Add/Edit Dialog */}
      <Dialog open={open} onClose={closeForm}>
        <DialogTitle>{form.id ? 'Edit' : 'Add'} Holding</DialogTitle>
        <DialogContent>
          <Autocomplete
            options={companies}
            getOptionLabel={o => o.label}
            value={companies.find(o => o.ticker === form.ticker) || null}
            onChange={(_, opt) => setForm(f => ({ ...f, ticker: opt ? opt.ticker : '' }))}
            renderInput={params => (
              <TextField {...params} margin="dense" label="Company" fullWidth />
            )}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Quantity"
            type="number"
            fullWidth
            value={form.quantity}
            onChange={e => setForm(f => ({ ...f, quantity: e.target.value }))}
          />
        </DialogContent>
        <DialogActions>
          {form.id && (
            <IconButton color="error" onClick={deleteHolding}>
              <DeleteIcon />
            </IconButton>
          )}
          <Button onClick={closeForm}>Cancel</Button>
          <Button
            onClick={submitForm}
            variant="contained"
            disabled={!form.ticker || !form.quantity}
          >
            {form.id ? 'Update' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
