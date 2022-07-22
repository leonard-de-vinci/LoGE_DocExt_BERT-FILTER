import {
  Dialog,
  DialogTitle,
  DialogContent,
  Typography,
  Divider,
  Grid,
  Card,
  CardContent,
  IconButton,
  Switch,
  FormControlLabel,
  Box,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import React, { useState, useEffect } from "react";
import TableOccurrence from "../../components/SearchEngine/TableOccurrence";
import TableFilters from "../../components/SearchEngine/TableFilters";
import GeneratedTokens from "../../components/SearchEngine/GeneratedTokens";

export default function Popup(props) {
  const { open, handleClose, currentDoc, index, filters, query } = props;

  const [docInfo, setDocInfo] = useState({}); //les informations du document actuel (nbWordsInAbstract, scores, occurrences ...)

  const [showTokens, setShowTokens] = useState(false); // switch qui indique si on affiche la liste des bertTokens

  const handleShowTokens = (event) => {
    setShowTokens(event.target.checked);
  };

  const fetchDocument = () => {
    let url = `http://localhost:8000/search_doc/${index}/${query}/${currentDoc["doc_id"]}/?`;
    for (let filter of filters) {
      url += `&filters=${filter}`;
    }
    fetch(url)
      .then((response) => response.json())
      .then((json) => setDocInfo(json["document"]));
  };

  useEffect(() => {
    if (open === true) {
      fetchDocument();
    }
    //eslint-disable-next-line
  }, [open]);

  const handleCloseDialog = () => {
    handleClose(false);
    setShowTokens(false);
  };

  if (Object.keys(docInfo).length > 0) {
    return (
      <Dialog
        open={open}
        onClose={handleCloseDialog}
        fullWidth
        maxWidth={"lg"}
        PaperProps={{
          style: { borderRadius: "20px" },
        }}
        scroll="body"
      >
        <DialogTitle>
          ID : {currentDoc["doc_id"]} | Score : {currentDoc["score"]}
          {
            <IconButton
              aria-label="close"
              onClick={handleCloseDialog}
              sx={{
                position: "absolute",
                right: 8,
                top: 8,
                color: (theme) => theme.palette.grey[500],
              }}
            >
              <CloseIcon />
            </IconButton>
          }
        </DialogTitle>
        <DialogContent dividers sx={{ m: 0, p: 0 }}>
          <Grid
            container
            direction="column"
            rowSpacing={4}
            margin={0}
            px={5}
            pb={showTokens ? 5 : 0}
          >
            <Grid item xs={1} key={"abstract"}>
              <Card sx={{ boxShadow: 24, borderRadius: "10px" }}>
                <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 } }}>
                  <Grid
                    container
                    direction="row"
                    spacing={1}
                    alignItems="center"
                  >
                    <Grid item>
                      <Typography color="text.primary" variant="h6">
                        Abstract
                      </Typography>
                    </Grid>
                    <Grid item>
                      <Typography color="text.secondary" fontSize={14}>
                        ({docInfo["nbWordsInAbstract"]} word(s))
                      </Typography>
                    </Grid>
                  </Grid>
                  <Divider sx={{ my: 1 }} />
                  <Typography color="text.secondary">
                    {docInfo["abstract"]}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={1} key={"occurrenceAbstract"}>
              <Card sx={{ boxShadow: 24, borderRadius: "10px" }}>
                <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 } }}>
                  <Typography variant="h6" color="text.primary">
                    Occurrence of the query words in the abstract
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <TableOccurrence
                    data={docInfo["occurrenceOfQueryWords"]["abstract"]}
                  />
                </CardContent>
              </Card>
            </Grid>

            {docInfo["occurrenceOfQueryWords"]["filters"] && (
              <Grid item xs={1} key={"occurrenceFilters"}>
                <Card sx={{ boxShadow: 24, borderRadius: "10px" }}>
                  <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 } }}>
                    <Typography variant="h6" color="text.primary">
                      Occurrence of the query words per filter
                    </Typography>
                    <Divider sx={{ my: 1 }} />
                    <TableFilters
                      data={docInfo["occurrenceOfQueryWords"]["filters"]}
                    />
                  </CardContent>
                </Card>
              </Grid>
            )}

            {docInfo["scoresOfQueryWords"] && (
              <Grid item xs={1} key={"scores"}>
                <Card sx={{ boxShadow: 24, borderRadius: "10px" }}>
                  <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 } }}>
                    <Typography variant="h6" color="text.primary">
                      Scores of the query words per filter
                    </Typography>
                    <Divider sx={{ my: 1 }} />
                    <TableFilters data={docInfo["scoresOfQueryWords"]} />
                  </CardContent>
                </Card>
              </Grid>
            )}

            <Box marginTop={showTokens ? 4 : 0}>
              <FormControlLabel
                control={
                  <Switch checked={showTokens} onChange={handleShowTokens} />
                }
                sx={{ color: "text.secondary", fontSize: 5 }}
                label={
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    fontStyle={"italic"}
                  >
                    Generated tokens
                  </Typography>
                }
              />
            </Box>

            {showTokens && (
              <Grid item xs={1} key={"test"}>
                <Card sx={{ boxShadow: 24, borderRadius: "10px" }}>
                  <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 } }}>
                    <Typography variant="h6" color="text.primary">
                      List of all generated tokens with their occurrences by
                      filter
                    </Typography>
                    <Divider sx={{ my: 1 }} />
                    <GeneratedTokens
                      currentDoc={currentDoc}
                      index={index}
                    ></GeneratedTokens>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </DialogContent>
      </Dialog>
    );
  }
}
