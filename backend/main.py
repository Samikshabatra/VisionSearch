"""FastAPI app: /search, /health, /images, and (in prod) the built frontend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from visionsearch.config import CONFIG
from backend.search_service import SearchService

IMAGES = CONFIG.data_dir / "flickr30k" / "images"
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

app = FastAPI(title="VisionSearch")
_service: SearchService | None = None


def get_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


class SearchRequest(BaseModel):
    query: str
    k: int = 12


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "gallery_size": get_service().gallery_size}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    return {"results": get_service().search(req.query, req.k)}


@app.get("/images/{name}")
def image(name: str) -> FileResponse:
    path = IMAGES / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


# In production the built frontend is served from the same origin.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
