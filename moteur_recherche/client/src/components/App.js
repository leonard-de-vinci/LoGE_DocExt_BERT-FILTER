import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ErrorPage from "../pages/ErrorPage";
import HomePage from "../pages/SearchEngine/HomePage";
import Search from "../pages/SearchEngine/SearchPage";
import Kibana from "../pages/Visualization/KibanaPage";

function App() {
  return (
    <>
      <main>
        <div>
          <Router>
            <Routes>
              <Route exact path="/" element={<HomePage />} />
              <Route path="/search/:_index/:_query" element={<Search />} />
              <Route path="/visualization" element={<Kibana />} />
              <Route path="*" element={<ErrorPage />} />
            </Routes>
          </Router>
        </div>
      </main>
    </>
  );
}

export default App;
