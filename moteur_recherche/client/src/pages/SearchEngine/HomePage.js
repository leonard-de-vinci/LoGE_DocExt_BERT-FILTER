import React, { useState } from "react";
import {
  Typography,
  TextField,
  Container,
  FormControl,
  InputLabel,
  Select,
  CssBaseline,
  Grid,
  MenuItem,
  InputAdornment,
  IconButton,
} from "@mui/material";

import { useNavigate } from "react-router-dom";
import CustomAppBar from "../../components/AppBar";
import SearchIcon from "@mui/icons-material/Search";

export default function HomePage() {
  let navigate = useNavigate();
  const [index, setIndex] = useState("");
  const [query, setQuery] = useState("");
  const [indexEmpty, setIndexEmpty] = useState(false);

  const handleIndex = (event) => {
    setIndex(event.target.value);
    if (event.target.value !== "") {
      setIndexEmpty(false);
    }
  };

  const handleQuery = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmit = () => {
    if (index !== "") {
      if (query !== "") {
        navigate(`/search/${index}/${query}`);
      }
    } else {
      setIndexEmpty(true);
    }
  };

  const handleSubmitEnterKey = (e) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };

  return (
    <>
      <CssBaseline>
        <CustomAppBar></CustomAppBar>
        <main>
          <div>
            <Container>
              <Typography
                variant="h2"
                align="center"
                color="textPrimary"
                marginTop={20}
                gutterBottom
              >
                Search
              </Typography>
            </Container>
          </div>
          <div>
            <Grid
              container
              spacing={0.5}
              direction="row"
              alignItems="center"
              justifyContent="center"
              wrap="nowrap"
            >
              <Grid item>
                <FormControl sx={{ minWidth: 115 }} error={indexEmpty}>
                  <InputLabel id="select-index-label">Index</InputLabel>
                  <Select
                    required
                    labelId="select-index-label"
                    id="select-index"
                    value={index}
                    onChange={handleIndex}
                    label="Index"
                  >
                    <MenuItem value="">
                      <em>None</em>
                    </MenuItem>
                    <MenuItem value="antique">Antique</MenuItem>
                    <MenuItem value="antiquebm25">Antique BM25</MenuItem>
                    <MenuItem value="nfcorpus">Nfcorpus</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item>
                <TextField
                  autoFocus
                  sx={{ minWidth: 490 }}
                  onKeyDown={handleSubmitEnterKey}
                  value={query}
                  onChange={handleQuery}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          edge="end"
                          color="primary"
                          onClick={handleSubmit}
                        >
                          <SearchIcon />
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                ></TextField>
              </Grid>
            </Grid>
          </div>
        </main>
      </CssBaseline>
    </>
  );
}
