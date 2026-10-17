import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
load_dotenv("../.env")

from . import graph as agent
from . import rag, tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chunks = rag.init()
    app.state.graph = agent.build_graph()
    yield


app = FastAPI(title="HR Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{8,64}$")


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "chunks": request.app.state.chunks, "llm": agent.get_llm() is not None}


@app.get("/api/slots")
def slots():
    return {"timezone": "Asia/Kolkata", "slots": tools.list_available_slots.invoke({})}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):

    message = req.message.strip()
    if not message:
        raise HTTPException(422, "Message must not be blank.")
    try:
        return agent.run_turn(request.app.state.graph, req.thread_id or uuid.uuid4().hex, message)
    except Exception as e:
        print(f"[chat] error: {e!r}")
        raise HTTPException(502, "The assistant hit an error. Please try again.")
