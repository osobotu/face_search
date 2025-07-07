import os
import json
import numpy as np

def load_face_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        return json.load(f)
    
def load_embedding(embedding_path):
    return list(map(float, np.load(embedding_path).tolist()))

def get_flickr_url(photo_id: str) -> str:
    # todo: this path is not correct. need to fix.
    return f"https://live.staticflickr.com/{photo_id}.jpg"