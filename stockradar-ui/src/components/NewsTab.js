import React, { useEffect, useState } from 'react';
import { Box, CircularProgress, Link, Typography, Alert } from '@mui/material';
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import axios from 'axios';

export default function NewsTab() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
      setError(null);
      try {
        const res = await axios.get('/news-impact');
        if (isMounted) {
          const data = res.data.length ? res.data : [];
          setRows(data.map((n, i) => ({
            id: i,
            ticker: n.ticker,
            headline: n.headline,
            link: n.link,
            source: n.source,
            published: new Date(n.published).toLocaleString(),
            delta: n.delta,
          })));
        }
      } catch (e) {
        console.error(e);
        if (isMounted) {
          setError('Failed to load news impact data.');
          setRows([]);
        }
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchData();
    return () => {
      isMounted = false;
    };
  }, []);

  const columns = [
    { field: 'ticker', headerName: 'Ticker', width: 100 },
    {
      field: 'headline',
      headerName: 'Headline',
      flex: 1,
      renderCell: (p) => (
        <Link href={p.row.link} target="_blank" underline="hover">
          {p.value}
        </Link>
      ),
      sortable: false,
      filterable: false
    },
    { field: 'source', headerName: 'Source', width: 120 },
    { field: 'published', headerName: 'Published', width: 180 },
    {
      field: 'delta',
      headerName: 'ΔP',
      width: 100,
      renderCell: (p) => (
        <Typography color={p.value >= 0 ? 'green' : 'error'}>
          {p.value >= 0 ? '+' : ''}{p.value}
        </Typography>
      )
    },
  ];

  return (
    <Box p={2} height="70vh">
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {loading ? (
        <Box
          height="100%"
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <CircularProgress />
        </Box>
      ) : rows.length === 0 ? (
        <Box
          height="100%"
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <Typography variant="h6" color="textSecondary">
            No news impact data available. Try refreshing or adding stocks to your portfolio.
          </Typography>
        </Box>
      ) : (
        <DataGrid
          rows={rows}
          columns={columns}
          slots={{ toolbar: GridToolbar }}
          slotProps={{ toolbar: { showQuickFilter: true } }}
          initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
          pageSizeOptions={[10, 25, 50]}
          disableRowSelectionOnClick
        />
      )}
    </Box>
  );
}