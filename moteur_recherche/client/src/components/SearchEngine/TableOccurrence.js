import React from "react";
import {
  Paper,
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableCell,
  TableRow,
} from "@mui/material";

export default function TableOccurrence(props) {
  const data = props.data;
  return (
    <TableContainer component={Paper} sx={{ boxShadow: 0 }}>
      <Table
        sx={{ minWidth: 400, margin: "auto" }}
        aria-label="simple table"
        size="small"
      >
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: "bold" }}>word</TableCell>
            <TableCell sx={{ fontWeight: "bold" }} align="right">
              occurrence
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row) => (
            <TableRow
              key={row.word}
              sx={{
                "&:last-child td, &:last-child th": {
                  border: 0,
                },
              }}
            >
              <TableCell
                component="th"
                scope="row"
                sx={{ color: "text.secondary" }}
              >
                {row["word"]}
              </TableCell>
              <TableCell align="right" sx={{ color: "text.secondary" }}>
                {row["occurrence"]}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
