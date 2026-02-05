import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from .routers import chat

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Trip Planner AI",
    description="AI-powered vacation rental search and trip planning",
    version="1.0.0"
)

# Include routers
app.include_router(chat.router)

# Serve static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def root():
    """Serve the main chat interface."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Trip Planner AI API", "docs": "/docs"}


@app.get("/health")
async def health():
    """Health check endpoint for Render."""
    return {"status": "healthy"}
