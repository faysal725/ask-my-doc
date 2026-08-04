from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import ingest, query



app = FastAPI(
    title="MyApp",
    version="1.0.0",
    description="My awesome API 🚀"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["GET", "POST"],
    allow_credentials=["*"],
    allow_headers=["*"],
)

app.get("/health", tags=["Health"])(lambda: {"status": "ok"})


app.include_router(ingest.router)
app.include_router(query.router)