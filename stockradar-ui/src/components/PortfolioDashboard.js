// src/components/PortfolioDashboard.js
import React, { useEffect, useState, useMemo } from 'react';
import {
  Box, Grid, Card, CardContent, CardHeader, Typography,
  TextField, IconButton, Chip, Button, Dialog, DialogTitle,
  DialogContent, DialogActions, FormControl, InputLabel,
  Select, MenuItem, Stack, useTheme, Skeleton, Autocomplete
} from '@mui/material';
import {
  Edit as EditIcon,
  Refresh as RefreshIcon,
  ShowChart as ChartIcon
} from '@mui/icons-material';
import axios from 'axios';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid
} from 'recharts';

export default function PortfolioDashboard() {
  const theme = useTheme();

  // ─── State ──────────────────────────────────────────────────────────────────
  const [portfolio, setPortfolio] = useState(null);
  const [quotes, setQuotes]       = useState([]);
  const [stocks, setStocks]       = useState([]);     // master stock list
  const [filter, setFilter]       = useState('');
  const [sortField, setSortField] = useState('ticker');
  const [sortDir, setSortDir]     = useState('asc');
  const [error, setError]         = useState(null);

  // add/edit dialog
  const [editOpen, setEditOpen]   = useState(false);
  const [editing, setEditing]     = useState(null);   // if null → add mode
  const [newTicker, setNewTicker] = useState(null);
  const [newQty, setNewQty]       = useState('');

  // detail chart
  const [openChart, setOpenChart] = useState(false);
  const [detail, setDetail]       = useState({
    ticker: '', name: '', data: [], period: '1d'
  });

  // top performers
  const [topPerf, setTopPerf]     = useState(null);
  const [tpPeriod, setTpPeriod]   = useState('1d');

  // ─── Effects ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    loadAll();
    axios.get('/stocks')
      .then(res => setStocks(res.data))
      .catch(() => setStocks([]));
  }, []);

  useEffect(() => {
    loadTopPerf(tpPeriod);
  }, [tpPeriod]);

  // ─── Data Loading ─────────────────────────────────────────────────────────────
  async function loadAll() {
    setError(null);
    setPortfolio(null);
    try {
      const [pRes, qRes] = await Promise.all([
        axios.get('/portfolio'),
        axios.get('/quotes')
      ]);
      setPortfolio(pRes.data);
      setQuotes(qRes.data);
    } catch {
      setError('Failed to load portfolio');
      setPortfolio([]);
      setQuotes([]);
    }
    loadTopPerf(tpPeriod);
  }

  async function loadTopPerf(period) {
    setTopPerf(null);
    try {
      const interval = period === '1d' ? '1m' : '60m';
      const res = await axios.get(
        `/top-performers?period=${period}&interval=${interval}`
      );
      setTopPerf(res.data);
    } catch {
      setTopPerf([]);
    }
  }

  // ─── Merge portfolio + quotes ────────────────────────────────────────────────
  function merged() {
    if (!portfolio) return [];
    return portfolio.map(h => {
      const q     = quotes.find(x => x.ticker === h.ticker) || {};
      const spark = Array.isArray(q.spark) ? q.spark : [];
      const times = Array.isArray(q.spark_times) && q.spark_times.length === spark.length
        ? q.spark_times
        : spark.map((_, i) => Date.now() - (spark.length - i) * 60000);

      return {
        ...h,
        name:   q.name    || h.ticker,
        price:  q.price   || 0,
        change: q.change  || 0,
        spark,
        times
      };
    });
  }

  // ─── Total value trend ───────────────────────────────────────────────────────
  const totalSeries = useMemo(() => {
    const m = merged();
    if (!m.length) return [];
    const L = Math.min(...m.map(x => x.spark.length));
    return Array.from({ length: L }).map((_, i) => ({
      time:  m[0].times[i],
      value: parseFloat(
        m.reduce((sum, s) => sum + (s.spark[i] || s.price) * s.quantity, 0)
        .toFixed(2)
      )
    }));
  }, [portfolio, quotes]);

  // ─── Open small detail chart ─────────────────────────────────────────────────
  async function handleOpenChart(ticker, name) {
    const interval = detail.period === '1d' ? '1m' : '60m';
    const res = await axios.get(
      `/top-performers?period=${detail.period}&interval=${interval}`
    );
    const item  = res.data.find(p => p.ticker === ticker) || {};
    const spark = Array.isArray(item.spark) ? item.spark : [];
    const times = Array.isArray(item.spark_times) && item.spark_times.length === spark.length
      ? item.spark_times
      : spark.map((_, i) => Date.now() - (spark.length - i) * 60000);

    setDetail({
      ticker,
      name,
      data: spark.map((price,i) => ({ time: times[i], price })),
      period: detail.period
    });
    setOpenChart(true);
  }

  function handleDetailPeriod(newP) {
    setDetail(d => ({ ...d, period: newP }));
    handleOpenChart(detail.ticker, detail.name);
  }

  // ─── Add holding ─────────────────────────────────────────────────────────────
  async function handleAdd() {
    if (!newTicker) return;
    try {
      await axios.post('/portfolio', {
        ticker: newTicker.ticker,
        quantity: parseFloat(newQty)
      });
      setEditOpen(false);
      setNewTicker(null);
      setNewQty('');
      loadAll();
    } catch {
      alert('Failed to add holding');
    }
  }

  // ─── Start editing ───────────────────────────────────────────────────────────
  function openEditDialog(h) {
    setEditing(h);
    setNewTicker({ ticker: h.ticker, name: h.name });
    setNewQty(h.quantity.toString());
    setEditOpen(true);
  }

  // ─── Submit edits ───────────────────────────────────────────────────────────
  async function handleSubmitEdit() {
    if (!editing) return;
    try {
      await axios.put(`/portfolio/${editing.id}`, {
        ticker: editing.ticker,
        quantity: parseFloat(newQty)
      });
      setEditOpen(false);
      setEditing(null);
      setNewTicker(null);
      setNewQty('');
      loadAll();
    } catch {
      alert('Failed to update holding');
    }
  }

  // ─── Delete holding ─────────────────────────────────────────────────────────
  async function handleDelete() {
    if (!editing) return;
    try {
      await axios.delete(`/portfolio/${editing.id}`);
      setEditOpen(false);
      setEditing(null);
      loadAll();
    } catch {
      alert('Failed to delete holding');
    }
  }

  // ─── Filter & sort holdings ─────────────────────────────────────────────────
  const displayed = useMemo(() => {
    return merged()
      .filter(h =>
        h.ticker.includes(filter.toUpperCase()) ||
        h.name.toLowerCase().includes(filter.toLowerCase())
      )
      .sort((a,b) => {
        let cmp = 0;
        if (['ticker','name'].includes(sortField)) {
          cmp = a[sortField].localeCompare(b[sortField]);
        } else {
          cmp = (a[sortField]||0) - (b[sortField]||0);
        }
        return sortDir === 'asc' ? cmp : -cmp;
      });
  }, [portfolio, quotes, filter, sortField, sortDir]);

  // ─── Render ─────────────────────────────────────────────────────────────────
  return (
    <Box p={3}>
      {error && <Typography color="error">{error}</Typography>}

      {/* Portfolio Value */}
      <Card sx={{ mb:3, background: theme.palette.grey[100] }}>
        <CardContent>
          <Typography variant="h6">Portfolio Value</Typography>
          <Typography variant="h4" gutterBottom>
            ${ totalSeries.length
               ? totalSeries[totalSeries.length-1].value.toFixed(2)
               : '0.00' }
          </Typography>
          <Box height={80}>
            {totalSeries.length > 1 && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={totalSeries}>
                  <XAxis
                    dataKey="time"
                    tickFormatter={t =>
                      new Date(t).toLocaleTimeString([], {
                        hour:'2-digit', minute:'2-digit'
                      })
                    }
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis domain={['auto','auto']} hide/>
                  <Tooltip
                    labelFormatter={t => new Date(t).toLocaleString()}
                    formatter={v => `$${v.toFixed(2)}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke={theme.palette.primary.main}
                    dot={false}
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Controls */}
      <Stack direction="row" spacing={2} mb={2} alignItems="center">
        <TextField
          label="Search holdings"
          size="small"
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Sort by</InputLabel>
          <Select
            value={sortField}
            label="Sort by"
            onChange={e => setSortField(e.target.value)}
          >
            <MenuItem value="ticker">Ticker</MenuItem>
            <MenuItem value="name">Name</MenuItem>
            <MenuItem value="price">Price</MenuItem>
            <MenuItem value="change">Change</MenuItem>
          </Select>
        </FormControl>
        <IconButton
          onClick={() => setSortDir(d => d==='asc'?'desc':'asc')}
        >
          <RefreshIcon
            sx={{
              transform: sortDir==='asc'
                ? 'rotate(0)' : 'rotate(180deg)'
            }}
          />
        </IconButton>
        <Button variant="outlined" onClick={loadAll}>Refresh</Button>
        <Button
          variant="contained"
          onClick={() => {
            setEditing(null);
            setNewTicker(null);
            setNewQty('');
            setEditOpen(true);
          }}
        >
          Add Holding
        </Button>
      </Stack>

      {/* Holdings Grid */}
      <Grid container spacing={2}>
        {portfolio === null
          ? Array.from({ length: 6 }).map((_,i)=>(
              <Grid item xs={12} sm={6} md={4} key={i}>
                <Skeleton variant="rectangular" height={200}/>
              </Grid>
            ))
          : displayed.map(h=>(
              <Grid item xs={12} sm={6} md={4} key={h.id}>
                <Card sx={{ borderRadius:2, boxShadow:2 }}>
                  <CardHeader
                    avatar={
                      <IconButton
                        size="small"
                        onClick={() => openEditDialog(h)}
                      >
                        <EditIcon/>
                      </IconButton>
                    }
                    title={`${h.ticker} • ${h.name}`}
                    action={
                      <Chip
                        label={`${h.change>=0?'+':''}${h.change.toFixed(2)}`}
                        color={h.change>=0?'success':'error'}
                        size="small"
                      />
                    }
                  />
                  <CardContent>
                    <Typography variant="h5">
                      ${h.price.toFixed(2)}
                    </Typography>
                    <Box height={120} mt={1}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={h.spark.map((p,i)=>({
                            time: h.times[i],
                            price: p
                          }))}
                        >
                          <XAxis
                            dataKey="time"
                            tickFormatter={t=>
                              new Date(t).toLocaleTimeString([],{
                                hour:'2-digit', minute:'2-digit'
                              })
                            }
                            tick={{ fontSize:10 }}
                          />
                          <YAxis domain={['auto','auto']} hide/>
                          <Tooltip
                            labelFormatter={t=>new Date(t).toLocaleTimeString()}
                            formatter={v=>`$${v.toFixed(2)}`}
                          />
                          <Line
                            type="monotone"
                            dataKey="price"
                            stroke={h.change>=0
                              ? theme.palette.success.main
                              : theme.palette.error.main}
                            dot={false}
                            strokeWidth={2}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                    <Box display="flex" justifyContent="flex-end">
                      <IconButton
                        size="small"
                        onClick={() => handleOpenChart(h.ticker,h.name)}
                      >
                        <ChartIcon/>
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))
        }
      </Grid>

      {/* Top Performers */}
      <Box mt={5} mb={2} display="flex" justifyContent="space-between">
        <Typography variant="h5">Top Performers</Typography>
        <Stack direction="row" spacing={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth:120 }}>
            <InputLabel>Period</InputLabel>
            <Select
              value={tpPeriod}
              label="Period"
              onChange={e => setTpPeriod(e.target.value)}
            >
              <MenuItem value="1d">1 Day</MenuItem>
              <MenuItem value="5d">5 Days</MenuItem>
              <MenuItem value="1mo">1 Month</MenuItem>
            </Select>
          </FormControl>
          <IconButton onClick={()=>loadTopPerf(tpPeriod)}>
            <RefreshIcon/>
          </IconButton>
        </Stack>
      </Box>
      <Grid container spacing={2}>
        {topPerf===null
          ? Array.from({length:3}).map((_,i)=>(
              <Grid item xs={12} sm={6} md={4} key={i}>
                <Skeleton variant="rectangular" height={160}/>
              </Grid>
            ))
          : topPerf.map(p=>{
              const spark = Array.isArray(p.spark)?p.spark:[];
              const times = Array.isArray(p.spark_times) && p.spark_times.length===spark.length
                ? p.spark_times
                : spark.map((_,i)=>Date.now() - (spark.length-i)*60000);
              const data = spark.map((price,i)=>({
                time: times[i], price
              }));
              return (
                <Grid item xs={12} sm={6} md={4} key={p.ticker}>
                  <Card variant="outlined" sx={{ borderRadius:2, boxShadow:1 }}>
                    <CardHeader
                      title={p.ticker}
                      subheader={p.name}
                      action={
                        <Chip
                          label={`${p.change_percent>=0?'+':''}${p.change_percent.toFixed(2)}%`}
                          color={p.change_percent>=0?'success':'error'}
                          size="small"
                        />
                      }
                    />
                    <CardContent>
                      <Box height={100}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data}>
                            <XAxis
                              dataKey="time"
                              tickFormatter={t=>
                                new Date(t).toLocaleTimeString([],{
                                  hour:'2-digit', minute:'2-digit'
                                })
                              }
                              tick={{ fontSize:10 }}
                            />
                            <YAxis domain={['auto','auto']} hide/>
                            <Tooltip
                              labelFormatter={t=>new Date(t).toLocaleTimeString()}
                              formatter={v=>`$${v.toFixed(2)}`}
                            />
                            <Line
                              type="monotone"
                              dataKey="price"
                              stroke={p.change_percent>=0
                                ? theme.palette.success.main
                                : theme.palette.error.main}
                              dot={false}
                              strokeWidth={2}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })
        }
      </Grid>

      {/* Detail Chart Dialog */}
      <Dialog open={openChart} onClose={()=>setOpenChart(false)} maxWidth="lg" fullWidth>
        <DialogTitle>
          {detail.ticker} • {detail.name}
          <FormControl sx={{ ml:4, minWidth:120 }} size="small">
            <InputLabel>Period</InputLabel>
            <Select
              value={detail.period}
              onChange={e=>handleDetailPeriod(e.target.value)}
              label="Period"
            >
              <MenuItem value="1d">1 Day</MenuItem>
              <MenuItem value="5d">5 Days</MenuItem>
              <MenuItem value="1mo">1 Month</MenuItem>
            </Select>
          </FormControl>
        </DialogTitle>
        <DialogContent>
          <Box height={400}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={detail.data}>
                <CartesianGrid strokeDasharray="3 3"/>
                <XAxis
                  dataKey="time"
                  tickFormatter={t=>
                    new Date(t).toLocaleTimeString([],{
                      hour:'2-digit', minute:'2-digit'
                    })
                  }
                />
                <YAxis domain={['auto','auto']}/>
                <Tooltip
                  labelFormatter={t=>new Date(t).toLocaleString()}
                  formatter={v=>`$${v.toFixed(2)}`}
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke={theme.palette.primary.main}
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={()=>setOpenChart(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Add/Edit Dialog */}
      <Dialog open={editOpen} onClose={()=>setEditOpen(false)}>
        <DialogTitle>
          {editing ? 'Edit Holding' : 'Add New Holding'}
        </DialogTitle>
        <DialogContent sx={{ pt:1, pb:2 }}>
          <Autocomplete
            options={stocks}
            getOptionLabel={opt => `${opt.ticker} — ${opt.name}`}
            value={ editing ? { ticker: editing.ticker, name: editing.name } : newTicker }
            onChange={(_, v) => setNewTicker(v)}
            disabled={!!editing}
            isOptionEqualToValue={(opt, val) => opt.ticker === val.ticker}
            renderInput={params=>(
              <TextField
                {...params}
                label="Ticker / Company"
                margin="dense"
                fullWidth
              />
            )}
          />
          <TextField
            margin="dense"
            label="Quantity"
            type="number"
            value={newQty}
            onChange={e=>setNewQty(e.target.value)}
            fullWidth
          />
        </DialogContent>
        <DialogActions>
          {editing && (
            <Button color="error" onClick={handleDelete}>
              Delete
            </Button>
          )}
          <Button onClick={()=>setEditOpen(false)}>Cancel</Button>
          <Button
            onClick={editing ? handleSubmitEdit : handleAdd}
            disabled={
              (!editing && (!newTicker || !newQty)) ||
              (editing && !newQty)
            }
          >
            {editing ? 'Save' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
