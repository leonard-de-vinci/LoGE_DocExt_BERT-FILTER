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

export default function TableFilters(props) {
  const data = props.data;
  const tableColumns=Object.keys(data[0]) // ex : ["word", "bertTokens", "basic", ..]

  return (
    <TableContainer component={Paper} sx={{ boxShadow: 0 }}>
      <Table
        sx={{ minWidth: 400, margin: "auto" }}
        aria-label="simple table"
        size="small"
      >
        <TableHead>
          <TableRow>
            {tableColumns.map((column)=>(
               column==="word" ? <TableCell key={column} sx={{ fontWeight: "bold" }}>{column}</TableCell>:<TableCell align="right" key={column} sx={{ fontWeight: "bold" }}>{column}</TableCell>
            ))}
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
              {tableColumns.map((column)=>(
                column==="word" ? <TableCell  sx={{ color: "text.secondary" }} >{row[column]}</TableCell> : <TableCell  sx={{ color: "text.secondary" }} align="right">{row[column]}</TableCell>
              ))}

            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
