import React, { useEffect, useRef, useState } from "react";
import {
  Card,
  Typography,
  TextField,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  CssBaseline,
  Grid,
  MenuItem,
  Box,
  Chip,
  InputAdornment,
  IconButton,
  CardActionArea,
} from "@mui/material";

import SearchIcon from "@mui/icons-material/Search";
import { useNavigate, useParams } from "react-router-dom";
import CustomAppBar from "../../components/AppBar";
import Popup from "./Popup";

const filtersList = ["abstract", "bertTokens","basic","score","perfectPredictions","pca_axes","pca_centroid"];

export default function Search() {
  let { _index, _query } = useParams();
  let navigate = useNavigate();
  const [nbResultsPerPage, setNbResultsPerPage] = useState(10);
  const [index, setIndex] = useState(_index);
  const [query, setQuery] = useState(_query);
  const [filters, setFilters] = useState(["abstract"]);
  const [documents, setDocuments] = useState([]);
  const [indexEmpty, setIndexEmpty] = useState(false);
  const [open, setOpen] = useState(false);
  const currentDoc = useRef({});

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleCurrentDoc = (doc) => {
    currentDoc.current = doc;
    handleClickOpen();
  };

  const handleFilters = (event) => {
    if (event.target.value.length > 0) {
      setFilters(
        typeof event.target.value === "string"
          ? event.target.value.split(",")
          : event.target.value
      );
    } else {
      setFilters(["abstract"]);
    }
  };

  const handleNbResultsPerPage = (event) => {
    setNbResultsPerPage(event.target.value);
  };

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
        fetchDocuments();
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

  const fetchDocuments = () => {
    let url = `http://localhost:8000/search/${index}/${query}/?limit=${nbResultsPerPage}`;

    for (let filter of filters) {
      url += `&filters=${filter}`;
    }

    fetch(url)
      .then((response) => response.json())
      .then((json) => setDocuments(json["documents"]));
  };

  useEffect(() => {
    fetchDocuments();
    //eslint-disable-next-line
  }, [nbResultsPerPage, filters,index]);

  return (
    <>
      <CssBaseline>
        <CustomAppBar></CustomAppBar>
        <main>
          <div>
            <Grid container spacing={1} direction="row" sx={{ px: 4 }}>
              <Grid item>
                <FormControl sx={{ minWidth: 115 }} error={indexEmpty}>
                  <InputLabel id="select-index-label">Index</InputLabel>
                  <Select
                    labelId="select-index-label"
                    id="select-index"
                    value={index}
                    onChange={handleIndex}
                    label="Index"
                  >
                    <MenuItem value="antique">Antique</MenuItem>
                    <MenuItem value="antiquebm25">Antique BM25</MenuItem>
                    <MenuItem value="nfcorpus">Nfcorpus</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item>
                <TextField
                  sx={{ minWidth: 500 }}
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

              <Grid item>
                <FormControl>
                  <InputLabel id="select-filters-label">Filters</InputLabel>
                  <Select
                    labelId="select-filters-label"
                    id="select-filters"
                    value={filters}
                    onChange={handleFilters}
                    label="Filters"
                    multiple
                    sx={{ height: 56}}
                    renderValue={(selected) => (
                      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                        {selected.map((value) => (
                          <Chip key={value} label={value} />
                        ))}
                      </Box>
                    )}
                  >
                    {filtersList.sort().map((filter) => (
                      <MenuItem key={filter} value={filter}>
                        {filter}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item sx={{ marginLeft: "auto" }}>
                <FormControl sx={{ minWidth: 100 }}>
                  <InputLabel id="select-nb-results-label">
                    Nb of results
                  </InputLabel>
                  <Select
                    labelId="select-nb-results-label"
                    id="select-nb-results"
                    value={nbResultsPerPage}
                    onChange={handleNbResultsPerPage}
                    label="Number of results"
                  >
                    <MenuItem value={10}>10</MenuItem>
                    <MenuItem value={25}>25</MenuItem>
                    <MenuItem value={50}>50</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </div>
          <div>
            <Grid
              container
              direction="column"
              rowSpacing={3}
              margin={0}
              px={4}
              pb={4}
            >
              {documents.map((doc) => (
                <Grid item xs={1} key={doc.doc_id}>
                  <Card sx={{ boxShadow: 3, borderRadius: "9px" }}>
                    <CardActionArea onClick={() => handleCurrentDoc(doc)}>
                      <CardContent sx={{ p: 2.5, pb: 1 }}>
                        <Typography
                          variant="h6"
                          color="text.primary"
                          marginBottom={1}
                        >
                          {doc.abstract}
                        </Typography>
                        <Typography variant="overline" color="text.secondary">
                          ID : {doc.doc_id} | score : {doc.score}
                        </Typography>
                      </CardContent>
                    </CardActionArea>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </div>
        </main>
        <Popup
          open={open}
          handleClose={handleClose}
          currentDoc={currentDoc.current}
          index={index}
          filters={filters}
          query={query}
        ></Popup>
      </CssBaseline>
    </>
  );
}
