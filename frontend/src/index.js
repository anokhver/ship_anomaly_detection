import React from "react";
import ReactDOM from "react-dom/client";

const App = () => {
  const [msg, setMsg] = React.useState("");

  React.useEffect(() => {
    fetch("/api/hello/")
      .then(res => res.json())
      .then(data => setMsg(data.message));
  }, []);

  return <h1>{msg}</h1>;
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
