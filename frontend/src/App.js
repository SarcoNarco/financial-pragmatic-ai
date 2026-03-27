import { useMemo, useState } from "react";

import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";

const TOKEN_KEY = "financial_pragmatic_ai_token";

export default function App() {
  const [token, setToken] = useState(localStorage.getItem(TOKEN_KEY) || "");
  const [authView, setAuthView] = useState("login");

  const isAuthenticated = useMemo(() => Boolean(token), [token]);

  const handleLoggedIn = (nextToken) => {
    localStorage.setItem(TOKEN_KEY, nextToken);
    setToken(nextToken);
  };

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_KEY);
    setToken("");
    setAuthView("login");
  };

  if (!isAuthenticated) {
    if (authView === "signup") {
      return (
        <SignupPage
          onSignedUp={handleLoggedIn}
          onSwitchToLogin={() => setAuthView("login")}
        />
      );
    }

    return (
      <LoginPage
        onLoggedIn={handleLoggedIn}
        onSwitchToSignup={() => setAuthView("signup")}
      />
    );
  }

  return <DashboardPage token={token} onLogout={handleLogout} />;
}
