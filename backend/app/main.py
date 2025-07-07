from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import album, match

app = FastAPI(title="FaceSearch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(album.router)
app.include_router(match.router)

