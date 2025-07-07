import os
import cv2
import zipfile
import shutil
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import asyncio
from typing import List
from grpclib.client import Channel
from ahnlich_client_py.grpc.services.db_service import DbServiceStub
from ahnlich_client_py.grpc.db import query as db_query
from ahnlich_client_py.grpc.algorithm.algorithms import Algorithm
from ahnlich_client_py.grpc import keyval

# Load face analysis once
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def extract_single_face_embedding(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    faces = face_app.get(image)

    if len(faces) != 1:
        raise ValueError("The image must contain exactly one face.")

    return faces[0].embedding

def load_album_face_embeddings(album_path: str) -> List[Tuple[str, np.ndarray]]:
    embeddings = []
    embed_dir = os.path.join(album_path, "embeddings")
    for file in os.listdir(embed_dir):
        if file.endswith(".npy"):
            path = os.path.join(embed_dir, file)
            embedding = np.load(path)
            embeddings.append((file, embedding))
    return embeddings

def find_matching_faces(
    query_embedding: np.ndarray,
    album_path: str,
    threshold: float = 0.5
) -> List[str]:
    matches = []
    embeddings = load_album_face_embeddings(album_path)
    faces_dir = os.path.join(album_path, "faces")

    for filename, emb in embeddings:
        similarity = cosine_similarity([query_embedding], [emb])[0][0]
        if similarity >= threshold:
            # Map face file back to original photo name
            face_id = filename.replace("_face_", ".").split(".")[0]  # 'photo1'
            matches.append(face_id)

    return list(set(matches))
    
async def find_matching_faces_ahnlich(
    query_embedding: List[float],
    store_name: str,
    similarity_threshold: float = 0.5,
    top_k: int = 3
) -> List[str]:
    async with Channel(host="127.0.0.1", port=1369) as channel:
        db_client = DbServiceStub(channel)

        search_key = keyval.StoreKey(key=query_embedding.tolist())

        response = await db_client.get_sim_n(
            db_query.GetSimN(
                store=store_name,
                search_input=search_key,
                closest_n=top_k,
                algorithm=Algorithm.CosineSimilarity,
            )
        )

        matched_photo_ids = set()

        # how do I access the metadata in the returned entries?
        for entry in response.entries:
            if entry.similarity.value >= similarity_threshold:
                print(f"Similarity score for close matches: {entry.similarity.value}")
                # matched_photo_ids.add()
                photo_id = entry.value.value['photo_id'].raw_string
                matched_photo_ids.add(photo_id)
        return list(matched_photo_ids)


def export_matches_as_zip(matches: List[str], album_path: str, zip_output_path: str):
    download_dir = os.path.join(album_path, "matches_tmp")
    os.makedirs(download_dir, exist_ok=True)

    for image_id in matches:
        # Look for the full photo in album root
        for ext in [".jpg", ".jpeg", ".png"]:
            full_path = os.path.join(album_path, f"{image_id}{ext}")
            if os.path.exists(full_path):
                shutil.copy(full_path, os.path.join(download_dir, os.path.basename(full_path)))

    # Zip it up
    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        for root, _, files in os.walk(download_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

    # Clean up
    shutil.rmtree(download_dir)

    return zip_output_path

async def match_face_and_export_zip(query_image: str, album_path: str, zip_name: str = "matches.zip") -> str:
    print(f"Matching face in: {query_image}")
    query_embedding = extract_single_face_embedding(query_image)
    print(query_embedding.shape)
    print(f"Searching in album: {album_path}")
    # matches = find_matching_faces(query_embedding, album_path)
    album_id = os.path.basename(album_path)
    matches = await find_matching_faces_ahnlich(query_embedding, album_id)
    print(f"Found {len(matches)} matching image(s)")

    zip_path = os.path.join(album_path, zip_name)
    return export_matches_as_zip(matches, album_path, zip_path)

if __name__ == "__main__":
    query_image = "C:/Users/STUDENT/dev/face-search/data/query_image5.JPG"
    # album_path = "data/flickr_downloads/small"
    album_path = "C:/Users/STUDENT/dev/face-search/data/flickr_downloads/small"
    async def main():
        return await match_face_and_export_zip(query_image, album_path, zip_name="steve.zip")
    
    zip_file = asyncio.run(main())

    print(f"\nDownload ready: {zip_file}")

