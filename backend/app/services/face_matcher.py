import os
import cv2
import zipfile
import shutil
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

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

    return list(set(matches))  # Remove duplicates

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

def match_face_and_export_zip(query_image: str, album_path: str, zip_name: str = "matches.zip") -> str:
    print(f"🔍 Matching face in: {query_image}")
    query_embedding = extract_single_face_embedding(query_image)
    print(f"📁 Searching in album: {album_path}")
    matches = find_matching_faces(query_embedding, album_path)
    print(f"✅ Found {len(matches)} matching image(s)")

    zip_path = os.path.join(album_path, zip_name)
    return export_matches_as_zip(matches, album_path, zip_path)

if __name__ == "__main__":
    query_image = "C:/Users/STUDENT/dev/face-search/data/query_image4.png"
    album_path = "data/flickr_downloads/mastercard_grad_2025"
    zip_file = match_face_and_export_zip(query_image, album_path, zip_name="calvin.zip")

    print(f"\n📦 Download ready: {zip_file}")

