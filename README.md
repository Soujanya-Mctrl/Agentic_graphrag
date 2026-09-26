# Agentic GraphRAG — TigerGraph Hackathon

An agentic investigation system on TigerGraph, benchmarked against plain RAG
and fixed-sequence GraphRAG on accuracy, completeness, and token efficiency.

Runs fully offline in **mock mode** out of the box (no API keys, no
TigerGraph credentials) so you can develop and demo the control logic before
Savanna access or a real dataset is wired in.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env defaults to TG_USE_MOCK=true and no LLM key — orchestrator will run
# in stub mode (every step returns "answer" immediately). Set
# ANTHROPIC_API_KEY (or OPENAI_API_KEY + LLM_PROVIDER=openai) to get real
# multi-step investigations.

pytest                                  # smoke tests, mock mode
python -m src.benchmark.runner          # runs all 3 pipelines over data/sample_questions.json
python -m src.benchmark.dashboard       # writes results/dashboard.html
```

Open `results/dashboard.html` in a browser for the metrics dashboard.

## Wiring up the real dataset / TigerGraph

1. Set `TG_USE_MOCK=false`, fill in `TG_HOST` + `TG_SECRET` (or
   `TG_USERNAME`/`TG_PASSWORD`) from your Savanna instance.
2. Install the GSQL queries `entityLinkByName`, `multiHopTraverse`,
   `vectorTopK`, `getDocument` referenced in `src/tigergraph_client.py`
   (`TigerGraphClient`) — write these against your actual schema; the
   parameter names there are the contract the rest of the code expects.
3. Replace `data/sample_questions.json` with the provided dataset (same
   `{"id", "question", "gold_answer"}` shape — add `"complexity"` if you
   want to slice metrics by difficulty later).

## Layout

Each retrieval approach is a self-contained module; only genuinely shared
infra lives in `shared/`. This means you can hand `src/rag/`, `src/graphrag/`,
or `src/agentic_graphrag/` to a teammate on their own — the only thing they
need to know about is the `shared/` interface, not each other's internals.

```
src/
  shared/                        # common infra, imported by all three pipelines
    config.py                    # all env-driven settings
    state.py                     # InvestigationState — the agentic harness's shared state
    llm.py                       # single LLM call point (Anthropic/OpenAI/mock)
    embeddings.py                # text -> vector, with offline fallback
    tigergraph_client.py         # real + mock TigerGraph clients, same interface

  rag/                            # baseline 1 — no graph, no agent loop
    pipeline.py                   # single vector search -> generate

  graphrag/                       # baseline 2 — uses graph, but a FIXED sequence
    pipeline.py                    # entity link -> 1-hop traverse -> vector search -> generate

  agentic_graphrag/                # the system under test
    pipeline.py                     # thin entry point: builds state, runs the loop
    agents/
      orchestrator.py                # the control loop — see docs/architecture.md
      retrieval.py                    # entity_link, graph_traverse, vector_search, document_retrieve
      reasoning.py                     # aggregate, multi_hop_reason, evaluate_evidence

  benchmark/                       # ties all three modules together for scoring
    metrics.py                      # LLM-as-judge accuracy/completeness + token efficiency
    runner.py                        # runs all 3 pipelines over the dataset, saves JSON
    dashboard.py                      # renders results/*.json -> dashboard.html

docs/architecture.md              # Mermaid diagrams + design rationale
data/sample_questions.json        # toy dataset matching the mock graph
tests/test_harness.py             # offline smoke tests, import all 3 modules
```

Import rule of thumb: `rag/`, `graphrag/`, and `agentic_graphrag/` may each
import from `shared/`, and `benchmark/` may import from all three — but
`rag/`, `graphrag/`, and `agentic_graphrag/` never import from each other.
If you ever find yourself wanting to, that logic almost certainly belongs
in `shared/` instead.

## Mapping to the submission checklist

| Requirement | Where |
|---|---|
| Working Agentic GraphRAG system | `src/agentic_graphrag/agents/orchestrator.py` + `src/agentic_graphrag/pipeline.py` |
| Architecture diagram | `docs/architecture.md` |
| Three-way benchmark (RAG/GraphRAG/Agentic) | `src/benchmark/runner.py` |
| Metrics dashboard (tokens, accuracy, completeness) | `src/benchmark/dashboard.py` → `results/dashboard.html` |
| Demo video | record a terminal run of `runner.py` + a walkthrough of `full_trace` from one `agentic_graphrag.run()` call — **not included here**, record separately |
| GitHub repository | push this scaffold as-is |

## Known gaps to close before Round 1 submission

- `graph_rag._extract_candidate_mention` is a regex heuristic — fine as a
  deliberately "dumb" fixed baseline, don't over-invest here.
- No retry/backoff on LLM calls yet — add if you see rate-limit failures
  during a full dataset run.
- `evidence_evaluation_agent`'s `contradictions` field is populated but not
  yet acted on by the orchestrator (see docs/architecture.md's Round 2
  section) — that's intentionally a Round 2 stretch item, not a Round 1 bug.
