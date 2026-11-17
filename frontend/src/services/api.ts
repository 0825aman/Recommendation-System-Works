/**
 * Axios instance configured for the backend API.
 *
 * Base URL is read from VITE_API_BASE_URL environment variable.
 * Falls back to an empty string so the Vite proxy handles routing in development.
 */

import axios from "axios";

const baseURL = import.meta.env.VITE_API_BASE_URL ?? "";

const apiClient = axios.create({
  baseURL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

export default apiClient;
