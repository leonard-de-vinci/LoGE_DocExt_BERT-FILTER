import React, { useState, useEffect } from "react";
import { DataGrid } from "@mui/x-data-grid";
import { Box, Typography } from "@mui/material";

const filters = [
  "bertTokens",
  "basic",
  "score",
  "perfectPredictions",
  "pca_axes",
  "pca_centroid",
];

export default function GeneratedTokens(props) {
  const index = props.index;
  const currentDoc = props.currentDoc;
  const [tokens, setTokens] = useState({});

  const fetchTokens = () => {
    let url = `http://localhost:8000/tokens/${index}/${currentDoc["doc_id"]}/`;
    fetch(url)
      .then((response) => response.json())
      .then((json) => setTokens(json["tokens"]));
  };

  useEffect(() => {
    fetchTokens();
    //eslint-disable-next-line
  }, []);

  console.log(tokens);
  if (Object.keys(tokens).length > 0) {
    const columns = [{ field: "word", headerName: "word" }];

    filters.sort().forEach((filter) =>
      columns.push({
        field: filter,
        headerName: `${filter} (${tokens["totalNbTokensPerFilter"][filter]})`,
        description:
          tokens["totalNbTokensPerFilter"][filter] !== "-"
            ? `${tokens["totalNbTokensPerFilter"][filter]} token(s) in total in the ${filter} filter`
            : `No token generated in the ${filter} filter`,
        type: "number",
        flex: 1,
      })
    );

    return tokens["tokensList"].length > 0 ? (
      <Box sx={{ width: "100%", height: 850 }}>
        <DataGrid
          rows={tokens["tokensList"]}
          columns={columns}
          pageSize={14}
          disableSelectionOnClick
          getRowId={(row) => row.word}
        ></DataGrid>
      </Box>
    ) : (
      <Typography variant="body" color="text.secondary" fontStyle={"italic"}>
        No tokens found
      </Typography>
    );
  }
}
