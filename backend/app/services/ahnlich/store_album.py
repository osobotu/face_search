from ahnlich_client_py.grpc import keyval, metadata
from ahnlich_client_py.grpc.db import query as db_query
from .client import get_db_client
from .utils import load_embedding, load_face_metadata, get_flickr_url
import os

async def store_album_embeddings(album_id: str, metadata_path: str):
    db_client, channel = await get_db_client()
    try:
        await db_client.create_store(
            db_query.CreateStore(
                store=album_id,
                # InsightFace embedding shape (512,)
                dimension=512,
                # used to for searching and indexing
                create_predicates=["photo_id"],
                # throw an error if the store exists, set to False for it to do nothing
                error_if_exists=True,
            )
        )

        data = load_face_metadata(metadata_path)
        inputs = []

        for face in data:
            emb = load_embedding(face["embedding_path"])
            photo_id = os.path.splitext(os.path.basename(face['image_path']))[0]
            url = get_flickr_url(photo_id)
            print(url)
            key = keyval.StoreKey(key=emb)
            value = keyval.StoreValue(value={
                "photo_id": metadata.MetadataValue(raw_string=photo_id)
            })
            entry = keyval.DbStoreEntry(
                key=key,
                value=value,
            )
            inputs.append(entry)

        if inputs:
            await db_client.set(
                db_query.Set(store=album_id, inputs=inputs)
            )
            print(f"Stored {len(inputs)} embeddings in album store: {album_id}")
        else:
            print("No embeddings found.")
    finally:
        channel.close()