"""
main.py

The FastAPI app. Just one real endpoint - POST /ask - that takes a business
question and runs it through the orchestrator.

The survey data and the vector store are both loaded once when the app
starts up (see the startup event below), not on every request, so repeat
questions are fast.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.models import AskRequest, AskResponse
from app.orchestrator import run_pipeline
from app.rag_pipeline import build_or_load_vector_store

SURVEY_DATA_PATH = os.path.join("data", "survey_data.json")
FAQ_PATH = os.path.join("data", "faq.txt")

# filled in on startup, used by every request after that
vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # everything before "yield" runs once when the app starts
    global vector_store
    if not os.path.exists(SURVEY_DATA_PATH):
        raise RuntimeError(
            f"Could not find {SURVEY_DATA_PATH}. Run 'python data/generate_data.py' first."
        )
    print("Loading (or building) the FAQ vector store, this can take a moment on first run...")
    vector_store = build_or_load_vector_store(FAQ_PATH)
    print("MiniSense is ready.")

    yield  # the app runs while paused here

    # anything after "yield" would run on shutdown - nothing to clean up for us


app = FastAPI(title="MiniSense", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    return run_pipeline(
        question=request.question,
        survey_data_path=SURVEY_DATA_PATH,
        vector_store=vector_store,
    )
