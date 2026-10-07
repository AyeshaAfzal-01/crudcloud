# Cloud CRUD – FastAPI + LangGraph on Vercel

One-page app: create / view / edit / delete users (name, email, about, optional picture),
plus a prompt box. A LangGraph graph routes the prompt to a dedicated LLM per CRUD operation
(tool calling via structured output). Missing key/field → it returns and asks, nothing is written.

    prompt → router LLM → create | read | update | delete (own LLM each) → result

## Free-tier setup
Vercel Hobby (hosting) · Groq API · Upstash Redis free plan (storage). Groq model access and free quotas depend on the account. The selected GPT-OSS model has published per-token rates, so confirm it is available at $0 in your Groq console and keep paid billing disabled if the project must stay free.

## Run locally
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export GROQ_API_KEY=your_groq_api_key   # get a key from https://console.groq.com/keys
    export GROQ_MODEL=openai/gpt-oss-20b
    uvicorn app:app --reload        # http://127.0.0.1:8000
(No cloud DB keys locally → data lives in memory.)

## Deploy on Vercel
1. Push this folder to GitHub (or run `npm i -g vercel && vercel` inside it).
2. Import the repo in Vercel – it auto-detects FastAPI (`app.py`) and serves `public/` as static.
3. Project → Storage → Marketplace → add **Upstash Redis** (pick the Free plan). This injects `KV_REST_API_URL` and `KV_REST_API_TOKEN` (cloud storage).
4. Project → Settings → Environment Variables → add `GROQ_API_KEY` and set `GROQ_MODEL` to `openai/gpt-oss-20b`. Remove any old `GROQ_MODEL` override such as `llama-3.3-70b-versatile`.
5. Confirm the selected model is available at $0 on your account, keep paid billing disabled, and check the current free-tier quota. When the quota is reached, assistant requests will fail until it resets; manual add / edit / delete remain available.
6. Redeploy. Check `/api/health` – it should show `"storage":"upstash-redis"`.

## Try these prompts
- `Add Sara Khan, sara@mail.com, loves cycling`   → creates
- `Add Ali`                                        → returns: missing email
- `Add Bob, bob@x.com with his photo`              → returns: picture mentioned but not attached
- `Delete a user`                                  → returns: which user?
- `Change Sara's bio to runner`                    → updates
