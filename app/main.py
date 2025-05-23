import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.text_similarity import router as text_similarity_router
from app.web_socket.notifier import websocket_endpoint
from app.core.models import init_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield

app = FastAPI(
    title="Translation Agent",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://k12s307.p.ssafy.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(text_similarity_router, prefix="/agent", tags=["text-similarity"])
app.websocket("/ws")(websocket_endpoint)
