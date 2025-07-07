from fastapi import APIRouter, Form
from pydantic import BaseModel
from app.services.album_scraper import scrape_album
from app.services.face_extractor import extract_faces_from_folder
from app.services.ahnlich.store_album import store_album_embeddings

import os

router = APIRouter(prefix="/album", tags=["Album"])

class AlbumRequest(BaseModel):
    album_url: str
    album_name: str

@router.post("/upload")
async def upload_album(request: AlbumRequest):
    save_path = f"data/flickr_downloads/{request.album_name}"
    os.makedirs(save_path, exist_ok=True)

    await scrape_album(request.album_url, album_name=request.album_name)
    extract_faces_from_folder(save_path)
    metadata_path = os.path.join(save_path, "metadata.json")
    await store_album_embeddings(request.album_name, metadata_path)

    return {
        "status": "success",
        "message": f"Album {request.album_name} processed"
    }