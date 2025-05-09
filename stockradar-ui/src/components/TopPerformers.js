// src/components/TopPerformers.js
import React, { useEffect, useState } from 'react';
import { Box, Typography, Chip, List, ListItem, ListItemText, Divider } from '@mui/material';
import axios from 'axios';

export default function TopPerformers() {
  const [list, setList] = useState([]);

  useEffect(() => {
    let isMounted = true;
    
    async function load() {
      try {
        const res = await axios.get('/top-performers');
        if (isMounted) {
          setList(res.data);
        }
      } catch (error) {
        console.error("Failed to fetch top performers", error);
      }
    }
    
    load();
    
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <Box mt={4}>
      <Typography variant="h6" gutterBottom>
        Today's Top Performers
      </Typography>

      <List disablePadding>
        {list.map((p, i) => (
          <React.Fragment key={p.ticker}>
            <ListItem
              secondaryAction={
                <Chip
                  label={`${p.change_percent >= 0 ? '+' : ''}${p.change_percent.toFixed(2)}%`}
                  size="small"
                  color={p.change_percent >= 0 ? "success" : "error"}
                />
              }
            >
              <ListItemText
                primary={`${p.ticker} — ${p.name}`}
                secondary={p.sector}
              />
            </ListItem>
            {i < list.length - 1 && <Divider component="li" />}
          </React.Fragment>
        ))}
      </List>
    </Box>
  );
}
