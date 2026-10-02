# Hosting the Redraw API (free)

The Vercel site is a static viewer: it can show the world, the baseline and
snapshotted plans, but checking, saving, running and voting on plans need the
Python API. This hosts it for free:

| Piece | Service (free tier) | Holds |
| --- | --- | --- |
| API | Hugging Face Docker Space, CPU basic (2 vCPU, 16 GB) | code + `data/processed` |
| Database | MongoDB Atlas M0 (512 MB) | players, plans, votes, jobs, reactions, chats, town halls |
| Viewer | Vercel (existing project) | static site; `/live-api/*` proxied to the Space |

The Space disk is wiped on every restart, so all player data lives in Atlas.
Plan playback files are rebuilt on request from the saved plan (fixed seeds give
identical bytes), so they never need to be stored.

## 1. MongoDB Atlas

1. Create a free **M0** cluster (https://cloud.mongodb.com).
2. Database Access: add a user with read and write access (password auth).
3. Network Access: allow `0.0.0.0/0` (Hugging Face has no fixed outbound IPs).
4. Connect, Drivers: copy the `mongodb+srv://...` string and put the database name in the path:
   `mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/redraw?retryWrites=true&w=majority`

The API creates its collections and indexes on startup (`api/store_mongo.py`).

## 2. Hugging Face Space

Use the SAME `data/processed` that built the deployed viewer snapshot: playback
edge indices and building ids must match what the viewer draws.

```bash
pip install huggingface_hub
HF_TOKEN=hf_xxx .venv/bin/python deploy/hf-space/build.py --push USER/redraw-api
```

Then in the Space, open Settings, Variables and secrets, and add the secret
`DATABASE_URL` (the Atlas string). Optional: `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL_FAST` and `LLM_MODEL_SMART` turn on resident chat, town hall
statements and custom ideas. The API answers at `https://USER-redraw-api.hf.space`
(`/health` shows when the baseline is ready).

Free Spaces sleep after 48 hours without traffic and take about a minute to
wake. While the Space is asleep the viewer falls back to its snapshot for plan
lists.

## 3. Vercel viewer

Proxy the API through the viewer's own domain, so the player cookie is first
party and no CORS is needed. In `client/vercel.json`, put this rewrite first:

```json
{ "source": "/live-api/:path*", "destination": "https://USER-redraw-api.hf.space/:path*" }
```

Then set the build environment variable `VITE_API_BASE=/live-api` and redeploy.
With a hosted API, plan routes are live-first (new plans and votes show up) and
fall back to the snapshot when the API is unreachable (`isLiveFirst` in
`client/src/api.ts`).
