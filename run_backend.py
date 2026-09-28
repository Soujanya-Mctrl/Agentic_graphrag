#!/usr/bin/env python3
"""Run the FastAPI backend server."""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Agentic GraphRAG FastAPI Server on http://localhost:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
