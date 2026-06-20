# Deploying ClermontAI

One Docker image runs **both** the FastAPI scoring API (internal port 8000) and
the Streamlit UI (port 8501). You only ever open the Streamlit URL.

```
            ┌─────────────── container ───────────────┐
  browser ──┼──▶ Streamlit UI :8501 ──▶ FastAPI :8000  │
            │      (app/ui.py)         (app/api.py      │
            │                           + score.py +    │
            │                           features.py +   │
            │                           models/*.joblib)│
            └──────────────────────────────────────────┘
```

---

## 1. Build the image

From `C:\clermontAI` (where the `Dockerfile` is):

```bash
docker build -t clermontai .
```

First build downloads Python + compiles deps (shap/llvmlite are the slow part) —
allow 5–15 min. Rebuilds are cached and fast.

> Not yet tested on this machine — Docker isn't installed here. If the build
> errors, it'll almost certainly be a dependency wheel; send me the message.

---

## 2. Run it (this is what your dad does)

He needs **only Docker Desktop installed**, then one command:

```bash
docker run --rm -p 8501:8501 clermontai
```

Then open **http://localhost:8501** in any browser. That's it — no Python, no
pip, no virtualenv on his machine.

If port 8501 is busy: `docker run --rm -p 9000:8501 clermontai` → open `:9000`.

**Low-end laptop notes**
- Needs ~2 GB free disk for the image and ~1 GB RAM while running.
- The **first score** is slow (model + SHAP explainer load on first request);
  every score after is fast.
- To hand him a file instead of a build: `docker save clermontai | gzip > clermontai.tar.gz`,
  he loads it with `docker load < clermontai.tar.gz` then runs step 2.

---

## 3. Push to GitHub

Make sure big/irrelevant folders are ignored (the repo should ship code +
the 4 model artifacts, **not** `venv_new/` or `data/`). Confirm `.gitignore`
covers `venv_new/` and `data/raw/`.

```bash
git init
git add Dockerfile entrypoint.sh requirements-serve.txt .dockerignore .gitattributes \
        app/ notebooks/score.py notebooks/features.py \
        models/lgbm_tuned.joblib models/calibrator.joblib \
        models/transformers.joblib models/model_contract.json \
        README.md DEPLOY.md
git commit -m "Add Docker serving stack"
git branch -M main
git remote add origin https://github.com/<you>/clermontAI.git
git push -u origin main
```

`lgbm_tuned.joblib` is 13 MB — fine for plain git. If you later add bigger
artifacts, use Git LFS.

---

## 4. Deploy to Hugging Face Spaces (Docker SDK)

1. Create a new Space → **SDK: Docker** → Blank.
2. Push this repo to the Space (it's just a git remote):
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/clermontAI
   git push space main
   ```
3. The Space's **`README.md` must start with this YAML front matter** so HF
   knows it's a Docker app listening on 8501:

   ```yaml
   ---
   title: ClermontAI Credit Risk Scorer
   emoji: 🏦
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 8501
   pinned: false
   ---
   ```

   Paste that block at the very top of `README.md` before pushing to the Space.
   (`app_port: 8501` is what makes HF route traffic to Streamlit. The API on
   8000 stays private inside the container.)

HF builds the Dockerfile automatically and serves the UI. The entrypoint already
runs Streamlit headless with XSRF/CORS disabled so it works inside the HF iframe.

---

## Split deployment (optional, advanced)

If you ever want the API and UI in **separate** containers, set
`CLERMONT_API_URL` on the UI container to point at the API host, e.g.
`-e CLERMONT_API_URL=http://api:8000`. The default is `http://localhost:8000`
(same-container mode), so you don't need this for the setups above.
