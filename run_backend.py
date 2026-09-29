import sys
import uvicorn

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("[SERVER] Starting Agentic GraphRAG FastAPI Server on http://localhost:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")

