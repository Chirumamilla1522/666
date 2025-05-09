import React, { useEffect, useState } from 'react';
import {
  Box, Button, Dialog, DialogTitle, DialogContent, TextField, DialogActions
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';

export default function PortfolioTab() {
  const [rows, setRows] = useState([]);
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ id:'', ticker:'', quantity:'', avg_price:'' });

  const fetchData = async () => {
    const res = await axios.get('/portfolio');
    setRows(res.data.map(r=>({ id:r.id, ...r })));
  };
  
  useEffect(() => {
    let isMounted = true;
    
    const load = async () => {
      try {
        const res = await axios.get('/portfolio');
        if (isMounted) {
          setRows(res.data.map(r => ({ id: r.id, ...r })));
        }
      } catch (error) {
        console.error("Failed to fetch portfolio", error);
      }
    };
    
    load();
    
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSubmit = async () => {
    const payload = {
      ticker: form.ticker,
      quantity: parseFloat(form.quantity),
      avg_price: parseFloat(form.avg_price)
    };
    if (form.id) await axios.put(`/portfolio/${form.id}`, payload);
    else await axios.post('/portfolio', payload);
    setOpen(false);
    setForm({ id:'', ticker:'', quantity:'', avg_price:'' });
    fetchData();
  };

  const handleDelete = async (id) => {
    await axios.delete(`/portfolio/${id}`);
    fetchData();
  };

  const columns = [
    { field:'ticker', headerName:'Ticker', width:100 },
    { field:'quantity', headerName:'Qty', type:'number', width:100 },
    { field:'avg_price', headerName:'Avg Price', type:'number', width:120 },
    {
      field:'actions', headerName:'Actions', width:150, renderCell: (params) => (
        <>
          <Button size="small" onClick={()=>{ setForm(params.row); setOpen(true); }}>Edit</Button>
          <Button size="small" color="error" onClick={()=>handleDelete(params.row.id)}>Del</Button>
        </>
      )
    }
  ];

  return (
    <Box p={2}>
      <Box mb={2} display="flex" justifyContent="space-between">
        <Button variant="contained" onClick={()=>setOpen(true)}>Add Holding</Button>
      </Box>

      <Box height={300} mb={4}>
        <ResponsiveContainer>
          <BarChart data={rows}>
            <XAxis dataKey="ticker" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="quantity" fill="#1976d2" />
          </BarChart>
        </ResponsiveContainer>
      </Box>

      <div style={{ height: 400, width: '100%' }}>
        <DataGrid rows={rows} columns={columns} pageSize={5} />
      </div>

      <Dialog open={open} onClose={()=>setOpen(false)}>
        <DialogTitle>{form.id ? 'Edit' : 'Add'} Holding</DialogTitle>
        <DialogContent>
          <TextField
            label="Ticker" fullWidth margin="dense"
            value={form.ticker}
            onChange={e=>setForm({...form,ticker:e.target.value.toUpperCase()})}
          />
          <TextField
            label="Quantity" fullWidth margin="dense" type="number"
            value={form.quantity}
            onChange={e=>setForm({...form,quantity:e.target.value})}
          />
          <TextField
            label="Avg Price" fullWidth margin="dense" type="number"
            value={form.avg_price}
            onChange={e=>setForm({...form,avg_price:e.target.value})}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={()=>setOpen(false)}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
