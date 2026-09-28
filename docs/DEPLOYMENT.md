# 🚀 Production Deployment Guide: Vercel & Render

This guide provides step-by-step instructions for deploying the **Agentic GraphRAG** full-stack system:
- **Frontend**: Deployed on **Vercel** (Global Edge CDN, React 19 + Vite 8)
- **Backend**: Deployed on **Render** (Python 3.11 / FastAPI Web Service)

---

## 🏛️ Deployment Architecture

```
[ User Browser ]
       │
       ├───► [ Vercel: React 19 SPA ] (https://agentic-graphrag.vercel.app)
       │         │
       │         └───► API Requests (VITE_API_URL or rewrite proxy)
       │                     │
       └─────────────────────┴───► [ Render: FastAPI Service ] (https://agentic-backend.onrender.com)
                                         │
                                         ├───► [ TigerGraph Savanna Cloud ] (Graph Database)
                                         └───► [ Groq Cloud / OpenAI API ] (LLM Completions)
```

---

## 📦 Part 1: Deploy Backend on Render

Deploy the backend first so you have your live API URL ready for the frontend.

### Step 1: Push Repository to GitHub
Ensure all latest commits are pushed to your GitHub repository:
```bash
git push origin main
```

### Step 2: Create a Web Service on Render
1. Log in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository (`Agentic_graphrag`).
4. Configure the service settings:
   - **Name**: `agentic-graphrag-backend`
   - **Region**: Any preferred region (e.g., `Oregon (US West)` or `Frankfurt`)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (repository root)
   - **Runtime**: `Python 3` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`

*(Alternatively: Render will automatically detect `render.yaml` if you choose **New +** $\rightarrow$ **Blueprint**!)*

### Step 3: Configure Environment Variables on Render
Under the **Environment** tab on Render, add the following variables:

| Variable | Recommended Value / Notes |
| :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `OMP_NUM_THREADS` | `1` |
| `LLM_PROVIDER` | `groq` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` |
| `GROQ_API_KEY` | `your_groq_api_key` |
| `HF_TOKEN` | `your_huggingface_token` |
| `TG_TGCLOUD` | `true` |
| `TG_USE_MOCK` | `false` |
| `TG_HOST` | `https://tg-405c9fef-db94-41c5-a870-87a230e1e5b7.tg-2635877100.i.tgcloud.io` |
| `TG_GRAPHNAME` | `AgenticGraphRag` |
| `TG_SECRET` | `2m0lbortiu8s18bh87vh05egfjqhskpp` |
| `TG_USERNAME` | `tigergraph` |
| `TG_PASSWORD` | `your_tigergraph_password` |

### Step 4: Verify Backend Health
Once Render completes building, copy your service URL (e.g., `https://agentic-graphrag-backend.onrender.com`).
Test the health endpoint in your browser or terminal:
```bash
curl https://<your-render-service>.onrender.com/api/status
```
You should receive a `{"status": "online", "tigergraph": {...}}` JSON response.

---

## ⚡ Part 2: Deploy Frontend on Vercel

### Step 1: Import Project into Vercel
1. Log in to [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Import your GitHub repository (`Agentic_graphrag`).

### Step 2: Configure Project Settings
In the Vercel project configuration screen:
- **Framework Preset**: `Vite`
- **Root Directory**: Click `Edit` and select **`frontend`** (or leave root since `vercel.json` is provided)
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Install Command**: `npm install`

### Step 3: Add Environment Variables on Vercel
Under **Environment Variables**, add:

| Key | Value | Description |
| :--- | :--- | :--- |
| `VITE_API_URL` | `https://<your-render-service>.onrender.com` | Your deployed Render backend URL (no trailing slash) |

### Step 4: Deploy
Click **Deploy**. Vercel will build and deploy the React 19 app within ~30 seconds.

---

## 🔄 How the Cross-Service Connection Works

1. **Client Requests**: The React app in [`frontend/src/config.js`](../frontend/src/config.js) reads `import.meta.env.VITE_API_URL`.
2. **Direct API Calls**: When set, all calls (`/api/investigate`, `/api/benchmark`, `/api/status`) dispatch directly to your Render backend.
3. **CORS Handling**: The FastAPI backend in [`backend/main.py`](../backend/main.py) is pre-configured with `CORSMiddleware` (`allow_origins=["*"]`), allowing cross-origin requests from Vercel without browser blocks.
4. **Single Page Application Routing**: Both [`frontend/vercel.json`](../frontend/vercel.json) and root [`vercel.json`](../vercel.json) include client-side rewrites (`/* -> /index.html`) so refreshing routes works without 404 errors.

---

## 🛠️ Verification Checklist

- [ ] Render service logs show: `Uvicorn running on http://0.0.0.0:...`
- [ ] `GET /api/status` on Render returns `200 OK` with connected TigerGraph cluster.
- [ ] Vercel app loads the React dashboard UI with live TigerGraph badge.
- [ ] Running a single query on the **Investigate** tab returns all 3 pipeline answers with live Groq 120B completions.
- [ ] Diagnostics modal passes all 5 autonomous verification checks.
