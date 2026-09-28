/**
 * Central client-side configuration.
 * VITE_API_URL can be set during Vercel deployment to point directly
 * to the Render backend service (e.g., https://agentic-graphrag.onrender.com).
 * If left unset, requests default to relative paths (''), which works with Vite proxy
 * during local development or when served directly from the FastAPI static mount.
 */
export const API_BASE = import.meta.env.VITE_API_URL || '';
