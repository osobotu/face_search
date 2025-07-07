from fastapi import APIRouter, UploadFile, Form, File
from fastapi.responses import FileResponse
from app.services.face_matcher import match_face_and_export_zip
import os, uuid

router = APIRouter(prefix="/match", tags=["Matching"])

@router.post("/upload-query")
async def upload_query_image(file: UploadFile = File(...), album_name: str = Form(...)):
    contents = await file.read()
    tmp_name = f"tmp_query_{uuid.uuid4().hex}.jpg"
    tmp_path = os.path.join("temp", tmp_name)

    os.makedirs("temp", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(contents)

    zip_path = match_face_and_export_zip(tmp_path, f"data/flickr_downloads/{album_name}", zip_name=tmp_name)
    return {
        "status": "success",
        "zip_path": zip_path
    }


@router.get("/download")
async def download_zip(zip_path: str):
    return FileResponse(zip_path, filename=os.path.basename(zip_path))