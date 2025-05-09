// src/components/PortfolioList.js

import React, { useEffect, useState } from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemText,
  Typography,
  TextField,
  Divider,
  Chip,
  IconButton
} from '@mui/material';
import { Sparklines, SparklinesLine } from 'react-sparklines';
import { ArrowDropDown } from '@mui/icons-material';
import axios from 'axios';

export default function PortfolioList() {
  const [quotes, setQuotes] = useState([]);
  const [filter, setFilter] = useState("");
  const [sortAsc, setSortAsc] = useState(false);
  const [topPerf, setTopPerf] = useState([]);

  // Load portfolio quotes
  useEffect(() => {
    let isMounted = true;
    
    const loadQuotes = async () => {
      try {
        const res = await axios.get("/quotes");
        if (isMounted) {
          setQuotes(res.data);
        }
      } catch (err) {
        console.error("Failed to fetch quotes", err);
      }
    };
    
    loadQuotes();
    
    return () => {
      isMounted = false;
    };
  }, []);

  // Load top performers
  useEffect(() => {
    let isMounted = true;
    
    const loadTop = async () => {
      try {
        const res = await axios.get("/top-performers");
        if (isMounted) {
          setTopPerf(res.data);
        }
      } catch (err) {
        console.error("Failed to fetch top performers", err);
      }
    };
    
    loadTop();
    
    return () => {
      isMounted = false;
    };
  }, []);

  // Filter & sort portfolio
  const displayed = quotes
    .filter(q =>
      q.ticker.includes(filter.toUpperCase()) ||
      q.name.toLowerCase().includes(filter.toLowerCase())
    )
    .sort((a, b) =>
      sortAsc
        ? a.ticker.localeCompare(b.ticker)
        : b.ticker.localeCompare(a.ticker)
    );

  return (
    <Box>
      {/* Portfolio Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h6">My Symbols</Typography>
        <IconButton size="small" onClick={() => setSortAsc(!sortAsc)}>
          <ArrowDropDown style={{ transform: sortAsc ? "rotate(180deg)" : "none" }} />
        </IconButton>
      </Box>

      {/* Portfolio Search */}
      <TextField
        fullWidth
        size="small"
        placeholder="Search"
        value={filter}
        onChange={e => setFilter(e.target.value)}
        sx={{ mb: 2 }}
      />

      {/* Portfolio List */}
      <List disablePadding>
        {displayed.map((q, idx) => (
          <React.Fragment key={q.ticker}>
            <ListItem
              secondaryAction={
                <Box textAlign="right">
                  <Typography variant="subtitle1">${q.price.toFixed(2)}</Typography>
                  <Chip
                    label={`${q.change >= 0 ? "+" : ""}${q.change.toFixed(2)}`}
                    size="small"
                    color={q.change >= 0 ? "success" : "error"}
                    sx={{ mt: 0.5 }}
                  />
                </Box>
              }
            >
              <Box width={80} mr={2}>
                <Sparklines data={q.spark} limit={30} width={80} height={30}>
                  <SparklinesLine style={{ stroke: "#3f51b5", fill: "none", strokeWidth: 1 }} />
                </Sparklines>
              </Box>
              <ListItemText
                primary={<Typography variant="subtitle1">{q.ticker}</Typography>}
                secondary={<Typography variant="body2" color="textSecondary">{q.name}</Typography>}
              />
            </ListItem>
            {idx < displayed.length - 1 && <Divider component="li" />}
          </React.Fragment>
        ))}
      </List>

      {/* Top Performers Section */}
      <Box mt={4} mb={2}>
        <Typography variant="h6">Today's Top Performers</Typography>
      </Box>
      <List disablePadding>
        {topPerf.map((p, idx) => (
          <React.Fragment key={p.ticker}>
            <ListItem
              secondaryAction={
                <Chip
                  label={`${p.change_percent >= 0 ? "+" : ""}${p.change_percent.toFixed(2)}%`}
                  size="small"
                  color={p.change_percent >= 0 ? "success" : "error"}
                />
              }
            >
              <ListItemText
                primary={<Typography variant="subtitle1">{p.ticker} — {p.name}</Typography>}
                secondary={<Typography variant="body2" color="textSecondary">{p.sector}</Typography>}
              />
            </ListItem>
            {idx < topPerf.length - 1 && <Divider component="li" />}
          </React.Fragment>
        ))}
      </List>
    </Box>
  );
}
