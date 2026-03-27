import { useState } from "react";

import { login } from "../api/client";

export default function LoginPage({ onLoggedIn, onSwitchToSignup }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!email || !password) {
      setError("Email and password are required.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await login(email, password);
      onLoggedIn(response.access_token);
    } catch (_err) {
      setError("Invalid credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4 text-slate-100">
      <form onSubmit={handleSubmit} className="w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-xl">
        <h1 className="mb-1 text-xl font-semibold">Sign in</h1>
        <p className="mb-6 text-sm text-slate-400">Access your transcript analysis workspace.</p>

        <div className="space-y-4">
          <label className="block text-sm">
            <span className="mb-1 block text-slate-300">Email</span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none"
              placeholder="you@company.com"
            />
          </label>

          <label className="block text-sm">
            <span className="mb-1 block text-slate-300">Password</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none"
              placeholder="••••••••"
            />
          </label>
        </div>

        {error ? <p className="mt-3 text-sm text-rose-400">{error}</p> : null}

        <button
          type="submit"
          disabled={loading}
          className="mt-5 w-full rounded bg-sky-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          {loading ? "Signing in..." : "Sign in"}
        </button>

        <button
          type="button"
          onClick={onSwitchToSignup}
          className="mt-3 w-full rounded border border-slate-700 px-4 py-2 text-sm text-slate-300"
        >
          Create account
        </button>
      </form>
    </div>
  );
}
