import React from "react";
import { AppBar, Toolbar, IconButton } from "@mui/material";

import { useNavigate } from "react-router-dom";
import HomeIcon from "@mui/icons-material/Home";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";

export default function CustomAppBar() {
  let navigate = useNavigate();

  return (
    <AppBar position="relative" color="transparent" elevation={0}>
      <Toolbar>
        <IconButton onClick={() => navigate("/")}>
          <HomeIcon></HomeIcon>
        </IconButton>
        <IconButton onClick={() => navigate("/visualization")}>
          <AutoGraphIcon></AutoGraphIcon>
        </IconButton>
      </Toolbar>
    </AppBar>
  );
}
