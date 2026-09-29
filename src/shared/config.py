"""
Central configuration. Everything that varies between local/dev/prod
or between your machine and a teammate's lives here, pulled from env vars.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" | "anthropic" | "openai"
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    hf_token: str = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_HUB_TOKEN", "")

    # --- TigerGraph / Savanna ---
    tg_host: str = os.getenv("TG_HOST", "")
    tg_graph_name: str = os.getenv("TG_GRAPH_NAME") or os.getenv("TG_GRAPHNAME", "GraphRAG")
    tg_username: str = os.getenv("TG_USERNAME", "")
    tg_password: str = os.getenv("TG_PASSWORD", "")
    tg_secret: str = os.getenv("TG_SECRET", "")  # alternative to user/pass
    tg_use_mock: bool = os.getenv("TG_USE_MOCK", "true").lower() == "true"

    # --- Vector search ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    top_k_vector: int = int(os.getenv("TOP_K_VECTOR", "8"))

    # --- Agent harness ---
    max_investigation_steps: int = int(os.getenv("MAX_STEPS", "3"))
    min_confidence_to_stop: float = float(os.getenv("MIN_CONFIDENCE", "0.75"))

    # --- Benchmark ---
    dataset_path: str = os.getenv("DATASET_PATH", "data/sample_questions.json")
    results_dir: str = os.getenv("RESULTS_DIR", "results")


CONFIG = Config()
