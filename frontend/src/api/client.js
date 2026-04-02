import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

function authHeader(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function signup(email, password) {
  const response = await api.post("/auth/signup", { email, password });
  return response.data;
}

export async function login(email, password) {
  const response = await api.post("/auth/login", { email, password });
  return response.data;
}

export async function analyzeTranscript(transcript) {
  const response = await api.post("/analyze", { transcript });
  return response.data;
}

export async function saveAnalysis(token, transcript) {
  const response = await api.post(
    "/save-analysis",
    { transcript },
    { headers: authHeader(token) }
  );
  return response.data;
}

export async function getHistory(token) {
  const response = await api.get("/history", { headers: authHeader(token) });
  return response.data;
}

export async function getAnalysisById(token, analysisId) {
  const response = await api.get(`/analysis/${analysisId}`, {
    headers: authHeader(token),
  });
  return response.data;
}

export async function compareAnalyses(token, analysisId1, analysisId2) {
  const response = await api.post(
    "/compare",
    { analysis_id_1: analysisId1, analysis_id_2: analysisId2 },
    { headers: authHeader(token) }
  );
  return response.data;
}

export async function uploadTranscript(token, file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/upload", formData, {
    headers: {
      ...authHeader(token),
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export function isUnauthorizedError(error) {
  return axios.isAxiosError(error) && error.response?.status === 401;
}
