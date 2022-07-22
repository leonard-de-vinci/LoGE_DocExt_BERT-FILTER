import { CssBaseline } from "@mui/material";
import React from "react";
import CustomAppBar from "../../components/AppBar";

export default function Kibana() {
  return (
    <>
      <CssBaseline>
        <CustomAppBar></CustomAppBar>
        <div>Kibana</div>
      </CssBaseline>
    </>
  );
}
