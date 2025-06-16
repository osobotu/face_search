import numpy as np
from typing import List, Dict, Tuple, Optional, Generator
import face_recognition
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import os
import logging
from collections import defaultdict
from dataclasses import dataclass
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaceMatch:
    """Data class for face match results"""
    image_path: str
    face_path: str
    similarity_score: float
    bbox: Tuple[int, int, int, int]
    cluster_id: Optional[int] = None

@dataclass
class ClusterInfo:
    """Data class for cluster information"""
    cluster_id: int
    face_count: int
    representative_face: str
    similarity_threshold: float
    member_images: List[str]

class FaceMatcher:
    """
    Face matching and clustering module.
    Handles face comparison, clustering, and streaming results.
    """
    
    def __init__(self, similarity_threshold: float = 0.6, clustering_eps: float = 0.4):
        """
        Initialize face matcher.
        
        Args:
            similarity_threshold: Minimum similarity for face matches (0-1)
            clustering_eps: DBSCAN epsilon parameter for clustering
        """
        self.similarity_threshold = similarity_threshold
        self.clustering_eps = clustering_eps
        self.face_database = {}  # image_path -> list of face encodings
        self.face_metadata = {}  # face_id -> metadata
        self.clusters = {}  # cluster_id -> ClusterInfo
        self.face_to_cluster = {}  # face_id -> cluster_id
        
    def load_face_database(self, faces_data: Dict) -> None:
        """
        Load face database from extracted faces data.
        
        Args:
            faces_data: Dictionary from FaceExtractor.save_extracted_faces
        """
        logger.info("Loading face database...")
        
        face_id = 0
        for image_path, faces in faces_data.items():
            self.face_database[image_path] = []
            
            for face_data in faces:
                # Store encoding
                self.face_database[image_path].append(face_data['face_encoding'])
                
                # Store metadata
                self.face_metadata[face_id] = {
                    'image_path': image_path,
                    'face_path': face_data['face_path'],
                    'bbox': face_data['bbox'],
                    'encoding': face_data['face_encoding']
                }
                
                face_id += 1
        
        total_faces = sum(len(faces) for faces in self.face_database.values())
        logger.info(f"Loaded {total_faces} faces from {len(self.face_database)} images")
    
    def cluster_faces(self) -> Dict[int, ClusterInfo]:
        """
        Cluster all faces in the database using DBSCAN.
        
        Returns:
            Dictionary of cluster information
        """
        logger.info("Starting face clustering...")
        
        if not self.face_metadata:
            raise ValueError("No faces loaded. Call load_face_database first.")
        
        # Prepare face encodings for clustering
        all_encodings = []
        face_ids = []
        
        for face_id, metadata in self.face_metadata.items():
            all_encodings.append(metadata['encoding'])
            face_ids.append(face_id)
        
        all_encodings = np.array(all_encodings)
        
        # Compute distance matrix (1 - cosine similarity)
        similarity_matrix = cosine_similarity(all_encodings)
        distance_matrix = 1 - similarity_matrix
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=2, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Process clustering results
        clusters = defaultdict(list)
        for face_id, cluster_label in zip(face_ids, cluster_labels):
            clusters[cluster_label].append(face_id)
            self.face_to_cluster[face_id] = cluster_label
        
        # Create cluster info objects
        self.clusters = {}
        for cluster_id, member_face_ids in clusters.items():
            if cluster_id == -1:  # Noise cluster
                continue
                
            # Find representative face (most central in cluster)
            cluster_encodings = [self.face_metadata[fid]['encoding'] for fid in member_face_ids]
            cluster_center = np.mean(cluster_encodings, axis=0)
            
            # Find face closest to center
            similarities = [cosine_similarity([cluster_center], [enc])[0][0] 
                          for enc in cluster_encodings]
            representative_idx = np.argmax(similarities)
            representative_face_id = member_face_ids[representative_idx]
            
            # Get unique images in cluster
            member_images = list(set(self.face_metadata[fid]['image_path'] 
                                   for fid in member_face_ids))
            
            self.clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                face_count=len(member_face_ids),
                representative_face=self.face_metadata[representative_face_id]['face_path'],
                similarity_threshold=self.clustering_eps,
                member_images=member_images
            )
        
        logger.info(f"Created {len(self.clusters)} clusters from {len(all_encodings)} faces")
        logger.info(f"Noise faces (unclustered): {len(clusters.get(-1, []))}")
        
        return self.clusters
    
    def find_matches(self, query_encoding: np.ndarray) -> Generator[FaceMatch, None, None]:
        """
        Find matching faces for a query face encoding.
        Yields results as they are found for streaming.
        
        Args:
            query_encoding: Face encoding of the query image
            
        Yields:
            FaceMatch objects for each matching face
        """
        logger.info("Searching for face matches...")
        
        matches_found = 0
        
        for face_id, metadata in self.face_metadata.items():
            # Calculate similarity
            similarity = cosine_similarity([query_encoding], [metadata['encoding']])[0][0]
            
            if similarity >= self.similarity_threshold:
                cluster_id = self.face_to_cluster.get(face_id)
                
                match = FaceMatch(
                    image_path=metadata['image_path'],
                    face_path=metadata['face_path'],
                    similarity_score=similarity,
                    bbox=metadata['bbox'],
                    cluster_id=cluster_id
                )
                
                matches_found += 1
                yield match
        
        logger.info(f"Found {matches_found} matching faces")
    
    def find_matches_batch(self, query_encoding: np.ndarray) -> List[FaceMatch]:
        """
        Find all matching faces at once (non-streaming version).
        
        Args:
            query_encoding: Face encoding of the query image
            
        Returns:
            List of FaceMatch objects sorted by similarity
        """
        matches = list(self.find_matches(query_encoding))
        return sorted(matches, key=lambda x: x.similarity_score, reverse=True)
    
    async def find_matches_async(self, query_encoding: np.ndarray) -> Generator[FaceMatch, None, None]:
        """
        Async version of find_matches for better streaming performance.
        
        Args:
            query_encoding: Face encoding of the query image
            
        Yields:
            FaceMatch objects for each matching face
        """
        logger.info("Searching for face matches (async)...")
        
        matches_found = 0
        batch_size = 100  # Process in batches to allow other tasks
        
        face_items = list(self.face_metadata.items())
        
        for i in range(0, len(face_items), batch_size):
            batch = face_items[i:i + batch_size]
            
            for face_id, metadata in batch:
                # Calculate similarity
                similarity = cosine_similarity([query_encoding], [metadata['encoding']])[0][0]
                
                if similarity >= self.similarity_threshold:
                    cluster_id = self.face_to_cluster.get(face_id)
                    
                    match = FaceMatch(
                        image_path=metadata['image_path'],
                        face_path=metadata['face_path'],
                        similarity_score=similarity,
                        bbox=metadata['bbox'],
                        cluster_id=cluster_id
                    )
                    
                    matches_found += 1
                    yield match
            
            # Allow other async tasks to run
            await asyncio.sleep(0.01)
        
        logger.info(f"Found {matches_found} matching faces (async)")
    
    def get_cluster_matches(self, cluster_id: int) -> List[str]:
        """
        Get all images in a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            List of image paths in the cluster
        """
        if cluster_id not in self.clusters:
            return []
        
        return self.clusters[cluster_id].member_images
    
    def save_clusters(self, filepath: str) -> None:
        """Save clustering results to file."""
        cluster_data = {
            'clusters': {k: {
                'cluster_id': v.cluster_id,
                'face_count': v.face_count,
                'representative_face': v.representative_face,
                'similarity_threshold': v.similarity_threshold,
                'member_images': v.member_images
            } for k, v in self.clusters.items()},
            'face_to_cluster': self.face_to_cluster,
            'parameters': {
                'similarity_threshold': self.similarity_threshold,
                'clustering_eps': self.clustering_eps
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(cluster_data, f, indent=2)
        
        logger.info(f"Clusters saved to {filepath}")
    
    def load_clusters(self, filepath: str) -> None:
        """Load clustering results from file."""
        with open(filepath, 'r') as f:
            cluster_data = json.load(f)
        
        # Reconstruct clusters
        self.clusters = {}
        for k, v in cluster_data['clusters'].items():
            self.clusters[int(k)] = ClusterInfo(**v)
        
        self.face_to_cluster = {int(k): v for k, v in cluster_data['face_to_cluster'].items()}
        
        # Load parameters
        params = cluster_data['parameters']
        self.similarity_threshold = params['similarity_threshold']
        self.clustering_eps = params['clustering_eps']
        
        logger.info(f"Clusters loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize matcher
    matcher = FaceMatcher(similarity_threshold=0.6, clustering_eps=0.4)
    
    # Load face database (assuming you have faces_data from FaceExtractor)
    # matcher.load_face_database(faces_data)
    
    # Perform clustering
    # clusters = matcher.cluster_faces()
    # matcher.save_clusters("face_clusters.json")
    
    # Example of finding matches for a query image
    # query_image_path = "query_face.jpg"
    # query_image = face_recognition.load_image_from_file(query_image_path)
    # query_encodings = face_recognition.face_encodings(query_image)
    
    # if query_encodings:
    #     query_encoding = query_encodings[0]
    #     
    #     # Streaming results
    #     print("Streaming matches:")
    #     for match in matcher.find_matches(query_encoding):
    #         print(f"Match: {match.image_path} (similarity: {match.similarity_score:.3f})")
    #     
    #     # Batch results
    #     matches = matcher.find_matches_batch(query_encoding)
    #     print(f"Found {len(matches)} total matches")
    
    print("Face matcher module ready!")