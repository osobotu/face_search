import os
import json
import cv2
import asyncio
import numpy as np
from typing import List, Dict
from insightface.app import FaceAnalysis
from ahnlich.store_album import store_album_embeddings

# Initialize face analysis model
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def extract_faces_from_image(image_path: str, album_path: str) -> List[Dict]:

    face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    faces = face_app.get(image)
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    # create folders for faces and embeddings in album folder
    face_dir = os.path.join(album_path, "faces")
    embed_dir = os.path.join(album_path, "embeddings")
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    results = []    

    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        embedding = face.embedding
        # embedding has a shape of (512,)

        face_crop = image[y1:y2, x1:x2]
        face_filename = f"{image_id}_face_{idx}.jpg"
        embed_filename = f"{image_id}_face_{idx}.npy"

        face_path = os.path.join(face_dir, face_filename)
        embed_path = os.path.join(embed_dir, embed_filename)

        cv2.imwrite(face_path, face_crop)
        np.save(embed_path, embedding)

        results.append({
            "face_id": f"{image_id}_face_{idx}",
            "image_path": image_path,
            "face_path": face_path,
            "embedding_path": embed_path,
            "bbox": bbox.tolist()
        })

    return results

def extract_faces_from_folder(album_path: str) -> List[Dict]:
    from glob import glob

    image_paths = glob(os.path.join(album_path, "*.jpg")) + \
                  glob(os.path.join(album_path, "*.jpeg")) + \
                  glob(os.path.join(album_path, "*.png"))

    all_faces_metadata = []

    for image_path in image_paths:
        try:
            faces = extract_faces_from_image(image_path, album_path)
            all_faces_metadata.extend(faces)
            print(f"Processed {image_path} | {len(faces)} faces found")
        except Exception as e:
            print(f"Skipped {image_path}: {e}")
    
    # save metadata.json file
    METADATA_FILE = "metadata.json"
    metadata_path = os.path.join(album_path, METADATA_FILE)
    os.makedirs(album_path, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(all_faces_metadata, f, indent=4)

    return all_faces_metadata

if __name__ == "__main__":
    import asyncio

    # album_path = "../../../data/flickr_downloads/small"
    album_path = "C:/Users/STUDENT/dev/face-search/data/flickr_downloads/small"

    album_id = os.path.basename(album_path)
    metadata = extract_faces_from_folder(album_path)
    print(metadata)
    metadata_path = os.path.join(album_path, "metadata.json")

    async def main():
        await store_album_embeddings(album_id=album_id, metadata_path=metadata_path)

    asyncio.run(main())
    print(f"\nDone! Total faces extracted and stored: {len(metadata)}")

