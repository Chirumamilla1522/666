// import React, { useEffect, useState, useMemo } from 'react';
// import {
//   Box, Paper, Stack, TextField, FormControl, InputLabel,
//   Select, MenuItem, IconButton, Button, Switch, FormControlLabel,
//   Typography
// } from '@mui/material';
// import { Refresh as RefreshIcon, TableChart, ViewModule } from '@mui/icons-material';
// import { DataGrid } from '@mui/x-data-grid';
// import {
//   ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip
// } from 'recharts';
// import axios from 'axios';

// export default function RecsTab() {
//   const [data, setData]           = useState([]);
//   const [filter, setFilter]       = useState('');
//   const [sectorFilter, setSector] = useState('');
//   const [sortField, setSortField] = useState('similarity_score');
//   const [sortDir, setSortDir]     = useState('desc');
//   const [viewTable, setViewTable] = useState(true);
//   const [error, setError]         = useState(null);

//   const fetchData = async () => {
//     setError(null);
//     try {
//       const res = await axios.get('/recommendations');
//       setData(res.data);
//     } catch {
//       setError('Failed to load recommendations');
//     }
//   };
//   useEffect(fetchData, []);

//   const sectors = useMemo(() => [...new Set(data.map(r => r.sector))], [data]);
//   const displayed = useMemo(() => {
//     return data
//       .filter(r =>
//         r.ticker.includes(filter.toUpperCase()) ||
//         r.name.toLowerCase().includes(filter.toLowerCase())
//       )
//       .filter(r => sectorFilter ? r.sector === sectorFilter : true)
//       .sort((a,b) => {
//         let cmp = ['ticker','name','sector'].includes(sortField)
//           ? a[sortField].localeCompare(b[sortField])
//           : a[sortField] - b[sortField];
//         return sortDir === 'asc' ? cmp : -cmp;
//       });
//   }, [data, filter, sectorFilter, sortField, sortDir]);

//   const columns = [
//     { field: 'ticker', headerName: 'Ticker', width: 100 },
//     { field: 'name',   headerName: 'Name',   flex: 1 },
//     { field: 'sector', headerName: 'Sector', width: 160 },
//     {
//       field: 'description',
//       headerName: 'About',
//       flex: 2,
//       renderCell: ({ value }) => (
//         <Tooltip title={value}>
//           <Typography variant="body2" noWrap>{value}</Typography>
//         </Tooltip>
//       ),
//       sortable: false,
//     },
//     {
//       field: 'similarity_score',
//       headerName: 'Score',
//       width: 120,
//       valueFormatter: ({ value }) => value.toFixed(3)
//     },
//     {
//       field: 'spark',
//       headerName: '1d Trend',
//       width: 160,
//       renderCell: ({ row }) => (
//         <ResponsiveContainer width="100%" height={40}>
//           <LineChart data={row.spark.map((v,i)=>({ i,v }))}>
//             <XAxis dataKey="i" hide/>
//             <YAxis hide domain={['auto','auto']}/>
//             <Tooltip formatter={v=>v.toFixed(2)} />
//             <Line dataKey="v" stroke="#1976d2" dot={false} strokeWidth={2}/>
//           </LineChart>
//         </ResponsiveContainer>
//       ),
//       sortable: false,
//       filterable: false,
//     }
//   ];

//   return (
//     <Box p={2}>
//       {error && <Typography color="error">{error}</Typography>}
//       <Paper sx={{ p:2, mb:2 }}>
//         <Stack direction="row" spacing={2} alignItems="center">
//           <TextField label="Search" size="small" value={filter} onChange={e=>setFilter(e.target.value)} />
//           <FormControl size="small" sx={{ minWidth:160 }}>
//             <InputLabel>Sector</InputLabel>
//             <Select value={sectorFilter} onChange={e=>setSector(e.target.value)} label="Sector">
//               <MenuItem value="">All</MenuItem>
//               {sectors.map(sec=> <MenuItem key={sec} value={sec}>{sec}</MenuItem>)}
//             </Select>
//           </FormControl>
//           <FormControl size="small" sx={{ minWidth:160 }}>
//             <InputLabel>Sort By</InputLabel>
//             <Select value={sortField} onChange={e=>setSortField(e.target.value)} label="Sort By">
//               <MenuItem value="ticker">Ticker</MenuItem>
//               <MenuItem value="name">Name</MenuItem>
//               <MenuItem value="sector">Sector</MenuItem>
//               <MenuItem value="similarity_score">Score</MenuItem>
//             </Select>
//           </FormControl>
//           <IconButton onClick={()=>setSortDir(d=>d==='asc'?'desc':'asc')}>
//             <RefreshIcon sx={{ transform: sortDir==='asc'?'rotate(0)':'rotate(180deg)' }}/>
//           </IconButton>
//           <FormControlLabel
//             control={<Switch checked={viewTable} onChange={e=>setViewTable(e.target.checked)}/>}
//             label={viewTable ? <TableChart/> : <ViewModule/>}
//           />
//           <Button startIcon={<RefreshIcon/>} onClick={fetchData}>Refresh</Button>
//         </Stack>
//       </Paper>

//       {viewTable
//         ? <Paper sx={{ height: 600 }}>
//             <DataGrid
//               rows={displayed.map((r,i)=>({ id:i, ...r }))}
//               columns={columns}
//               pageSize={10}
//               rowsPerPageOptions={[10]}
//               disableSelectionOnClick
//             />
//           </Paper>
//         : <Grid container spacing={2}>
//             {displayed.map((r,i)=>(
//               <Grid item xs={12} sm={6} md={4} key={i}>
//                 <Card variant="outlined">
//                   <CardContent>
//                     <Typography variant="h6">{r.ticker} • {r.name}</Typography>
//                     <Typography variant="body2" noWrap gutterBottom>{r.description}</Typography>
//                     <ResponsiveContainer width="100%" height={80}>
//                       <LineChart data={r.spark.map((v,i)=>({ i,v }))}>
//                         <XAxis dataKey="i" hide/>
//                         <YAxis hide domain={['auto','auto']}/>
//                         <Tooltip formatter={v=>v.toFixed(2)}/>
//                         <Line dataKey="v" stroke="#1976d2" dot={false} strokeWidth={2}/>
//                       </LineChart>
//                     </ResponsiveContainer>
//                     <Chip label={`Score: ${r.similarity_score.toFixed(3)}`} size="small" sx={{ mt:1 }}/>
//                   </CardContent>
//                 </Card>
//               </Grid>
//             ))}
//           </Grid>
//       }
//     </Box>
//   );
// }
// src/components/RecsTab.js
import React, { useEffect, useState, useMemo } from 'react';
import {
  Box, Paper, Stack, TextField, FormControl, InputLabel,
  Select, MenuItem, IconButton, Button, Switch, FormControlLabel,
  Typography, Grid, Card, CardContent, Chip, Tooltip
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TableChart,
  ViewModule
} from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip as ReTooltip
} from 'recharts';
import axios from 'axios';

export default function RecsTab() {
  const [data, setData]           = useState([]);
  const [filter, setFilter]       = useState('');
  const [sectorFilter, setSector] = useState('');
  const [sortField, setSortField] = useState('similarity_score');
  const [sortDir, setSortDir]     = useState('desc');
  const [viewTable, setViewTable] = useState(true);
  const [error, setError]         = useState(null);

  const fetchData = async () => {
    setError(null);
    try {
      const res = await axios.get('/recommendations');
      setData(res.data);
    } catch {
      setError('Failed to load recommendations');
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const sectors = useMemo(
    () => Array.from(new Set(data.map(r => r.sector).filter(Boolean))).sort(),
    [data]
  );

  const displayed = useMemo(() => {
    return data
      .filter(r =>
        r.ticker.includes(filter.toUpperCase()) ||
        r.name.toLowerCase().includes(filter.toLowerCase())
      )
      .filter(r => (sectorFilter ? r.sector === sectorFilter : true))
      .sort((a, b) => {
        let cmp;
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
    { field: 'name',   headerName: 'Name',   flex: 1 },
    { field: 'sector', headerName: 'Sector', width: 160 },
    {
      field: 'description',
      headerName: 'About',
      flex: 2,
      sortable: false,
      renderCell: ({ value }) => (
        <Tooltip title={value}>
          <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
            {value}
          </Typography>
        </Tooltip>
      )
    },
    {
      field: 'similarity_score',
      headerName: 'Score',
      width: 120,
      valueFormatter: ({ value }) => value.toFixed(3)
    },
    {
      field: 'spark',
      headerName: '1d Trend',
      width: 160,
      sortable: false,
      filterable: false,
      renderCell: ({ row }) => {
        const chartData = row.spark.map((v, i) => ({ x: i, y: v }));
        return (
          <ResponsiveContainer width="100%" height={40}>
            <LineChart data={chartData}>
              <XAxis dataKey="x" hide />
              <YAxis domain={['auto','auto']} hide />
              <ReTooltip formatter={v => v.toFixed(2)} />
              <Line dataKey="y" stroke="#1976d2" dot={false} strokeWidth={2}/>
            </LineChart>
          </ResponsiveContainer>
        );
      }
    }
  ];

  return (
    <Box p={2}>
      {error && (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      <Paper sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" spacing={2} alignItems="center">
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
              onChange={e => setSector(e.target.value)}
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

          <IconButton onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}>
            <RefreshIcon
              sx={{
                transform: sortDir === 'asc' ? 'rotate(0)' : 'rotate(180deg)'
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
            label={viewTable ? <TableChart/> : <ViewModule/>}
          />

          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchData}
          >
            Refresh
          </Button>
        </Stack>
      </Paper>

      {viewTable ? (
        <Paper sx={{ height: 600 }}>
          <DataGrid
            rows={displayed.map((r, i) => ({ id: i, ...r }))}
            columns={columns}
            pageSize={10}
            rowsPerPageOptions={[10]}
            disableSelectionOnClick
          />
        </Paper>
      ) : (
        <Grid container spacing={2}>
          {displayed.map((r, i) => (
            <Grid item xs={12} sm={6} md={4} key={i}>
              <Card variant="outlined" sx={{ p:1 }}>
                <CardContent>
                  <Typography variant="h6">
                    {r.ticker} • {r.name}
                  </Typography>
                  <Typography variant="body2" noWrap gutterBottom>
                    {r.description}
                  </Typography>
                  <Box height={80} mb={1}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={r.spark.map((v, idx) => ({ x: idx, y: v }))}>
                        <XAxis dataKey="x" hide />
                        <YAxis domain={['auto','auto']} hide />
                        <ReTooltip formatter={v => v.toFixed(2)} />
                        <Line dataKey="y" stroke="#1976d2" dot={false} strokeWidth={2}/>
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                  <Chip
                    label={`Score: ${r.similarity_score.toFixed(3)}`}
                    size="small"
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
}
