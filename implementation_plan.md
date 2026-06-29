# Implementation Plan - Decoupling and Extracting FastAPI Application

The goal is to extract the existing FastAPI web API out of the main Blob Watcher repository into a completely separate, isolated python application directory. This new directory will be ignored by git in the current main project and can be managed, run, and deployed independently.

Both applications will share the same underlying business logic modules (file processing, vectorization, naming, database operations, and report generation), but will be decoupled:
1. **Blob Watcher Application**: The current repository root remains a background service triggered by Azure Blob events.
2. **FastAPI Web API Application**: Located in `fastapi-app/` (git-ignored in the root project), it runs as an HTTP service handling web file uploads, chat queries, and webhook events.

---

## User Review Required

> [!IMPORTANT]
> The new FastAPI application will be placed in a directory called `fastapi-app` under the repository root.
> To prevent git from tracking it inside the main project, we will add `fastapi-app/` to the root's `.gitignore`.
>
> All business logic, LLM prompts, and configurations will be copied into `fastapi-app` so it has its own independent copy of these modules. That way, any deployment of `fastapi-app` will be self-contained and run on its own environment.

---

## Proposed Changes

We will group our tasks into the following steps:

### 1. Root Configuration and Cleanup

#### [MODIFY] [.gitignore](file:///c:/Users/johat.abrego/Documents/dev/blob-watcher-report-generator/.gitignore)
- Add `fastapi-app/` to exclude the new project from git tracking in the main repo.

#### [DELETE] [api](file:///c:/Users/johat.abrego/Documents/dev/blob-watcher-report-generator/api)
- Remove the old `api` directory and its contents (`api/index.py`) from the main project root since its functionality is now relocated.

---

### 2. New Isolated FastAPI Project

We will create a new directory `fastapi-app/` in the root workspace and populate it.

#### [NEW] `fastapi-app/main.py`
- Create the FastAPI main entry point based on the previous `api/index.py` implementation.
- This file will run under the execution context of the `fastapi-app/` directory and import local copies of the modules.

#### [NEW] Copy Modules into `fastapi-app/`
We will copy the following directories and files from the main project into `fastapi-app/`:
- `agent_module/`
- `core/`
- `blob_module/`
- `supabase_module/`
- `config.py`
- `.env`
- `requirements.txt`

This gives the `fastapi-app` project its own complete copy of the core pipeline (`core/pipeline.py`), RAG modules (`agent_module/rag_responder.py`), and helper classes so it is 100% self-contained.

---

## Verification Plan

We will verify both applications function correctly as separate entities.

### Automated/Manual Tests
- **Start FastAPI Application**: Run the API server locally:
  ```powershell
  cd fastapi-app
  python -m venv env
  .\env\Scripts\activate
  pip install -r requirements.txt
  uvicorn main:app --port 8000 --reload
  ```
- **Verify Endpoints**:
  - Query health endpoint: `GET http://localhost:8000/api/health`
  - Ensure it responds with `{"status": "ok"}`
- **Start Blob Watcher**: Ensure the main project's blob watcher can still be run:
  ```powershell
  python main.py
  ```
