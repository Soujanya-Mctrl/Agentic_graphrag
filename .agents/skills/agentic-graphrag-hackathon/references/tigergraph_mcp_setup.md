# TigerGraph Database & MCP Connection Guide

This guide details how to:
1. Connect to TigerGraph Database (Savanna Cloud or Local) using `pyTigerGraph`.
2. Set up and run the official `tigergraph-mcp` (Model Context Protocol) server.
3. Integrate TigerGraph MCP into Antigravity IDE and Agent frameworks (LangGraph, CrewAI, etc.).

---

## 1. Prerequisites & Installation

TigerGraph MCP requires Python 3.10 to 3.14 and TigerGraph 4.1+ (4.2+ recommended for TigerVector & hybrid search).

```bash
# Install tigergraph-mcp (includes pyTigerGraph, mcp SDK, pydantic, click)
pip install tigergraph-mcp

# Optional: Install LLM query generation extras (Cypher/GSQL generator tools)
pip install "tigergraph-mcp[llm]"

# Optional: If you want to serve MCP over HTTP/SSE instead of stdio
pip install uvicorn starlette
```

---

## 2. Setting Up TigerGraph Database

You have two main options:

### Option A: TigerGraph Savanna / TG Cloud (Recommended)
1. Sign up / Log in to [tgcloud.io](https://tgcloud.io/).
2. Create a new cluster / solution (select TigerGraph version 4.1 or 4.2+).
3. Note your credentials:
   - **Host URL**: e.g., `https://my-subdomain.i.tgcloud.io`
   - **Username**: usually `tigergraph`
   - **Password**: your cluster password
   - **Graph Name**: the graph you created in GraphStudio (e.g. `MyGraph`)
   - **TG Cloud flag**: `true`

### Option B: Local Docker (Community Edition)
Run the TigerGraph Docker container locally:
```bash
docker run -d -p 14240:14240 -p 9000:9000 -p 14088:14088 \
  --name tigergraph \
  --ulimit nofile=1000000:1000000 \
  tigergraph/tigergraph:latest
```
- **Host**: `http://localhost`
- **Username**: `tigergraph`
- **Password**: `tigergraph`
- **RESTPP Port**: `9000`
- **GSQL Port**: `14240`

---

## 3. Environment Variables Configuration (`.env`)

Create a `.env` file in the root of your project:

```bash
# --- TigerGraph Savanna / Cloud Connection ---
TG_HOST=https://your-instance.i.tgcloud.io
TG_GRAPHNAME=MyGraph
TG_USERNAME=tigergraph
TG_PASSWORD=your_cluster_password
TG_TGCLOUD=true

# Optional: REST++ & GSQL Ports (default 443 for Cloud, 9000/14240 for local)
# TG_RESTPP_PORT=443
# TG_GS_PORT=443

# Optional: Alternatively use API / JWT Token instead of password
# TG_API_TOKEN=your_token_here

# Tool permissions & logging
TG_LOG_TOOL_CALLS=true
```

---

## 4. Connecting Directly via Python (`pyTigerGraph`)

Test database connectivity using `pyTigerGraph`:

```python
import os
from dotenv import load_dotenv
import pyTigerGraph as tg

load_dotenv()

# Initialize connection
conn = tg.TigerGraphConnection(
    host=os.getenv("TG_HOST", "http://localhost"),
    graphname=os.getenv("TG_GRAPHNAME", ""),
    username=os.getenv("TG_USERNAME", "tigergraph"),
    password=os.getenv("TG_PASSWORD", "tigergraph"),
    tgCloud=os.getenv("TG_TGCLOUD", "false").lower() == "true"
)

# Authenticate / Fetch token if required
conn.getToken()

# Verify connection
print("Connected to TigerGraph!")
print("Version:", conn.getVer())
print("Vertices:", conn.getVertexTypes())
print("Edges:", conn.getEdgeTypes())
```

---

## 5. Connecting TigerGraph MCP

`tigergraph-mcp` exposes tools to AI agents using the standard MCP protocol.

### Testing MCP Server in CLI
Check that the server initializes properly:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}' | tigergraph-mcp
```

### Option A: Configure MCP in Antigravity IDE
To allow Antigravity agents to directly call TigerGraph tools, configure `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "tigergraph": {
      "command": "tigergraph-mcp",
      "args": ["--env-file", "/Users/tousifazam/Documents/DEV/Agentic_graphrag/.env"]
    }
  }
}
```

Or pass environment variables inline:
```json
{
  "mcpServers": {
    "tigergraph": {
      "command": "tigergraph-mcp",
      "env": {
        "TG_HOST": "https://your-instance.i.tgcloud.io",
        "TG_GRAPHNAME": "MyGraph",
        "TG_USERNAME": "tigergraph",
        "TG_PASSWORD": "your_password",
        "TG_TGCLOUD": "true"
      }
    }
  }
}
```

### Option B: Use with LangGraph / LangChain Agents
In your Python pipeline code for the hackathon, agents can connect to TigerGraph MCP over stdio:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="tigergraph-mcp",
    args=["--env-file", ".env"],
    env=None
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available MCP Tools:", [t.name for t in tools.tools])
```

---

## 6. Key MCP Tools Provided

Once connected, the agent gets access to built-in tools:

- `tigergraph__list_graphs` — List available graphs
- `tigergraph__get_graph_schema` — Inspect vertex types, edge types, attributes
- `tigergraph__get_vertex_count` — Count vertices
- `tigergraph__run_installed_query` — Execute parameterized pre-installed GSQL queries
- `tigergraph__gsql` — Execute custom GSQL statements / schema queries
- `tigergraph__vector_search` — Vector similarity search on TigerVector attributes
- `tigergraph__generate_gsql` / `tigergraph__generate_cypher` — LLM-assisted query generation (requires `[llm]` extra)
