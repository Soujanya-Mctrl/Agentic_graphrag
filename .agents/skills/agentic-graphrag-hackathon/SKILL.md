---
name: agentic-graphrag-hackathon
description: >-
  Comprehensive guide for the TigerGraph Agentic GraphRAG Hackathon. Use this
  skill when the user asks about hackathon requirements, problem statement,
  deliverables, judging criteria, timeline, dataset, evaluation methodology,
  how to set up TigerGraph, how to build RAG/GraphRAG/Agentic GraphRAG pipelines,
  or any other aspect of the Agentic GraphRAG Hackathon by TigerGraph.
---

# 🐯 Agentic GraphRAG Hackathon Guidebook

> **Single source of truth for the TigerGraph Agentic GraphRAG Hackathon.**
> Source: https://alluring-beryllium-491.notion.site/Agentic-GraphRAG-Hackathon-Guidebook-34fc2cb129c08146998af3568d7d2594

---

## Quick Reference

| What | Detail |
|---|---|
| **Prize Pool** | ₹70,000 (top 3) + certificates for all valid submissions |
| **Team Size** | Solo or up to 5 members |
| **Round 1 Deadline** | Sep 30, 2026 (Wed) |
| **Round 2 Deadline** | Oct 7, 2026 (Wed) — Top 15 only |
| **Results** | Oct 14, 2026 (Tue) |
| **Dataset** | [Google Drive](https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9?usp=sharing) |
| **GraphRAG Repo** | [github.com/tigergraph/graphrag](https://github.com/tigergraph/graphrag) |
| **TigerGraph Savanna** | [tgcloud.io](https://tgcloud.io/) |
| **Discord** | [discord.com/invite/eKWm3mbkw2](https://discord.com/invite/eKWm3mbkw2) |

---

## Core Objective

Build an AI agent that **autonomously investigates complex questions** using graph, vector, and document evidence — and **benchmark three approaches side-by-side**: RAG, GraphRAG, and Agentic GraphRAG.

**The headline question:** *When does a complex question require an agentic, multi-step investigation rather than a single GraphRAG or RAG retrieval?*

- **RAG** — retrieves similar text chunks via similarity search
- **GraphRAG** — adds structure: entities, relationships, multi-hop reasoning
- **Agentic GraphRAG** — adds autonomous planning: the system decides its own retrieval path based on what it finds

---

## Getting Started (3 Steps)

> Detailed setup & connection runbook: [TigerGraph & MCP Setup Guide](./references/tigergraph_mcp_setup.md)

### Step 1: Pick Your TigerGraph Environment
- **Option A (Recommended): TigerGraph Savanna** — web-based, zero installation → [tgcloud.io](https://tgcloud.io/) (credits provided)
- **Option B: Community Edition** — free, runs locally → [dl.tigergraph.com](https://dl.tigergraph.com/)

### Step 2: Clone the GraphRAG Repo
```bash
git clone https://github.com/tigergraph/graphrag.git
```

### Step 3: Get an LLM API Key
Any LLM provider of your choice. Most major providers offer free tiers sufficient for hackathon-scale usage.

---

## What to Build

**Three pipelines** that answer the same questions, plus a comparison:

1. **Pipeline 1: RAG** — Retrieve relevant text through similarity search → generate answer
2. **Pipeline 2: GraphRAG** — Use graph structure, entities, relationships, and supporting content to retrieve context → answer
3. **Pipeline 3: Agentic GraphRAG** — Agent plans the investigation, selects retrieval methods, evaluates intermediate results, performs additional steps as needed

### Agent Architecture Requirements

Your system must include:
- **Agent harness** — manages state, tools, context, evidence, and stopping criteria
- **Orchestrator agent** — determines what needs investigating and picks the next action (NOT a fixed retrieval sequence)
- **Specialised agents** for:
  - Entity linking
  - Graph traversal
  - Similarity search
  - Document retrieval
  - Aggregation
  - Multi-hop reasoning
  - Evidence evaluation

### Dynamic Investigation Examples

The orchestrator's next move depends on: original question + graph/entities + previous evidence + information still needed.

| Simple question | `Entity Linking → Graph Traversal → Answer` |
|---|---|
| Medium question | `Similarity Search → Identify Entity → Graph Traversal → Retrieve Docs → Answer` |
| Complex question | Multiple iterations before sufficient evidence |

---

## Dataset

**Download:** [Google Drive folder](https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9?usp=sharing)

| Component | Description |
|---|---|
| **Corpus** | Document collection with source materials to ingest and reason over |
| **100 visible questions** | Test, tune, and benchmark all 3 pipelines against these |
| **50 hidden questions** | Submit raw outputs (tokens, answers, agentic trace) — scored against held-out ground truth |

You can bring your own dataset as a bonus — the provided one is the common benchmark everyone is scored on.

---

## Evaluation Metrics

For **every question** and **every pipeline**, measure:

### Accuracy
- Correctness, completeness, grounding in available evidence

### Token Efficiency
- Context tokens
- LLM input tokens
- LLM output tokens
- Total tokens per answer

### Trace & Agentic Behavior (Agentic GraphRAG only)
- Number of retrieval and reasoning steps
- Retrieval methods selected
- Specialised agents invoked
- Tools called
- Time per operation / Tokens per operation
- Total tokens used
- Number of chunks and citations
- Whether the system changed strategy during investigation
- When and why the system decided to stop

> **Goal:** Not just whether Agentic GraphRAG produces a better answer, but whether the additional steps are *worth* the additional complexity and token cost.

---

## Judging Criteria

| Criteria | Weight | What We're Looking For |
|---|---|---|
| Investigation accuracy | **30%** | Answers complex questions correctly and completely using right evidence |
| Evidence quality & explainability | **15%** | Grounded answers with clear citations and investigation path |
| Agentic effectiveness & efficiency | **15%** | Picks right retrieval methods, uses agentic steps where they add value, balances accuracy with token cost |
| Agentic design, engineering & code quality | **15%** | Architecture, tool use, reliability, reproducibility, repo quality |
| Innovation | **15%** | Novel investigation methods, graph reasoning, or user experience |
| Final presentation & Q&A | **10%** | Demo quality, technical clarity, responses to judges |

---

## Two-Round Structure

### Round 1 — Open to All
- Build working Agentic GraphRAG system with three-way benchmark
- Top 15 teams advance to Round 2

### Round 2 — Top 15 Only (Oct 1 → Oct 10)
- Extend agent to reason over **evolving, conflicting, and uncertain facts**
- System must: detect conflicting versions, determine what supersedes what, identify authoritative sources, handle uncertainty
- Submit: demo video, writeup, metrics dashboard
- Top teams: live presentation to judges

---

## Required Deliverables

### Round 1
- [ ] Working Agentic GraphRAG system
- [ ] GitHub repository
- [ ] Architecture diagram
- [ ] Demo video
- [ ] Metrics dashboard (tokens, accuracy, completeness across 3 pipelines)
- [ ] *(Optional)* Social media post tagging @TigerGraph

### Round 2 (Finalists)
- [ ] Refined Agentic GraphRAG system
- [ ] Updated GitHub repository
- [ ] Architecture diagram
- [ ] 3–5 minute demo video
- [ ] Metrics dashboard
- [ ] Short writeup (what you built, how it works, key results, limitations, future work)
- [ ] *(Top teams)* Live presentation to judges

---

## Timeline

| Date | Milestone |
|---|---|
| Sep 2 (Tue) | Registration opens, guidebook and dataset live |
| Sep 14 (Sun) | Registration closes |
| **Sep 30 (Wed)** | **Round 1 submission deadline** |
| **Oct 7 (Wed)** | **Round 2 / final submission deadline** |
| Oct 8–10 (Thu–Sun) | Judging |
| **Oct 14 (Tue)** | **Results announced** |

---

## Accuracy Evaluation Methods

For the **100 visible questions**, use any approach:
- **LLM-as-Judge** — PASS/FAIL grading against ground truth
- **BERTScore** — semantic similarity
- **Manual comparison**

For the **50 hidden questions** — TigerGraph evaluates on their end against held-out ground truth.

---

## Important Links

| Resource | Link |
|---|---|
| TigerGraph GraphRAG Repo | https://github.com/tigergraph/graphrag |
| TigerGraph MCP | https://github.com/tigergraph/tigergraph-mcp |
| TigerGraph Savanna | https://tgcloud.io/ |
| Community Edition | https://dl.tigergraph.com/ |
| TigerGraph Docs | https://www.tigergraph.com/docs/home/ |
| Discord Community | https://discord.com/invite/eKWm3mbkw2 |
| Hackathon WhatsApp Group | https://chat.whatsapp.com/GN0x6yajuroDJTzLMEZvBR |
| Schedule 1:1 with Devanshu | https://calendly.com/devanshu-saxena-tigergraph/20min |
| Dataset | https://drive.google.com/drive/folders/10C0hzRaHlm00VYPFbjapKtWj0EPmLvQ9?usp=sharing |

---

## Need Help?

- **1:1 advice call:** https://calendly.com/devanshu-saxena-tigergraph/20min
- **WhatsApp:** https://wa.me/917404313376
- **Hackathon WhatsApp Group:** https://chat.whatsapp.com/GN0x6yajuroDJTzLMEZvBR
- **Discord:** https://discord.com/invite/eKWm3mbkw2

---

## TL;DR

1. Build 3 pipelines: **RAG, GraphRAG, Agentic GraphRAG** on TigerGraph
2. Two rounds:
   - **Round 1** (open): build the system, benchmark all three approaches
   - **Round 2** (top 15): extend to reasoning over evolving/conflicting/uncertain facts
3. Benchmark everything: accuracy, completeness, token efficiency, agentic trace
4. Judging: Investigation Accuracy (30%) + Evidence Quality (15%) + Agentic Effectiveness (15%) + Engineering (15%) + Innovation (15%) + Presentation (10%)
5. Ship publicly: GitHub repo, demo video, architecture diagram, metrics dashboard
6. **Build it. Benchmark it. Prove when agents matter.**
