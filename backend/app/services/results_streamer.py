import asyncio
import json
from typing import AsyncGenerator, Dict, List, Optional
from dataclasses import asdict
import face_recognition
import numpy as np
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from face_extraction_module import FaceExtractor
from face_matching_module import FaceMatcher, FaceMatch, ClusterInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingFaceRecognitionPipeline:
    """
    Complete streaming face recognition pipeline.
    Combines face extraction, matching, and clustering with real-time streaming results.
    """
    
    def __init__(self, 
                 album_directory: str,
                 similarity_threshold: float = 0.6,
                 clustering_eps: float = 0.4,
                 min_face_size: int = 80):
        """
        Initialize the pipeline.
        
        Args:
            album_directory: Directory containing album images
            similarity_threshold: Minimum similarity for matches
            clustering_eps: DBSCAN clustering parameter
            min_face_size: Minimum face size for extraction
        """
        self.album_directory = Path(album_directory)
        self.face_extractor = FaceExtractor(min_face_size=min_face_size)
        self.face_matcher = FaceMatcher(similarity_threshold, clustering_eps)
        
        self.is_initialized = False
        self.extracted_faces_dir = self.album_directory / "extracted_faces"
        self.cache_dir = self.album_directory / "cache"
        
        # Create directories
        self.extracted_faces_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
    
    async def initialize_database(self, force_rebuild: bool = False) -> None:
        """
        Initialize the face database by extracting faces and clustering.
        
        Args:
            force_rebuild: If True, rebuild even if cache exists
        """
        logger.info("Initializing face recognition database...")
        
        cache_file = self.cache_dir / "face_database.pkl"
        clusters_file = self.cache_dir / "clusters.json"
        
        # Check if we can load from cache
        if not force_rebuild and cache_file.exists() and clusters_file.exists():
            try:
                logger.info("Loading from cache...")
                await self._load_from_cache()
                self.is_initialized = True
                return
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}. Rebuilding...")
        
        # Extract faces from all images
        logger.info("Extracting faces from album images...")
        faces_data = await self._extract_faces_async()
        
        if not faces_data:
            raise ValueError("No faces found in album directory")
        
        # Load into matcher and cluster
        self.face_matcher.load_face_database(faces_data)
        
        logger.info("Clustering faces...")
        await self._cluster_faces_async()
        
        # Save to cache
        await self._save_to_cache(faces_data)
        
        self.is_initialized = True
        logger.info("Database initialization complete!")
    
    async def _extract_faces_async(self) -> Dict:
        """Extract faces asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run face extraction in thread pool to avoid blocking
        faces_data = await loop.run_in_executor(
            None, 
            self.face_extractor.extract_faces_from_directory,
            str(self.album_directory)
        )
        
        # Save extracted faces
        saved_faces = await loop.run_in_executor(
            None,
            self.face_extractor.save_extracted_faces,
            faces_data,
            str(self.extracted_faces_dir)
        )
        
        return saved_faces
    
    async def _cluster_faces_async(self) -> None:
        """Perform face clustering asynchronously."""
        loop = asyncio.get_event_loop()
        
        await loop.run_in_executor(
            None,
            self.face_matcher.cluster_faces
        )
        
        # Save clusters
        clusters_file = self.cache_dir / "clusters.json"
        await loop.run_in_executor(
            None,
            self.face_matcher.save_clusters,
            str(clusters_file)
        )
    
    async def _save_to_cache(self, faces_data: Dict) -> None:
        """Save processed data to cache."""
        import pickle
        
        cache_data = {
            'faces_data': faces_data,
            'face_database': self.face_matcher.face_database,
            'face_metadata': self.face_matcher.face_metadata
        }
        
        cache_file = self.cache_dir / "face_database.pkl"
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_pickle, cache_data, cache_file)
    
    def _save_pickle(self, data: Dict, filepath: Path) -> None:
        """Helper to save pickle file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    async def _load_from_cache(self) -> None:
        """Load processed data from cache."""
        import pickle
        
        cache_file = self.cache_dir / "face_database.pkl"
        clusters_file = self.cache_dir / "clusters.json"
        
        loop = asyncio.get_event_loop()
        
        # Load face database
        cache_data = await loop.run_in_executor(None, self._load_pickle, cache_file)
        
        self.face_matcher.face_database = cache_data['face_database']
        self.face_matcher.face_metadata = cache_data['face_metadata']
        
        # Load clusters
        await loop.run_in_executor(
            None,
            self.face_matcher.load_clusters,
            str(clusters_file)
        )
    
    def _load_pickle(self, filepath: Path) -> Dict:
        """Helper to load pickle file."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    async def process_query_image(self, query_image_path: str) -> AsyncGenerator[Dict, None]:
        """
        Process a query image and stream matching results.
        
        Args:
            query_image_path: Path to the user's query image
            
        Yields:
            Dictionary containing match information and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_database() first.")
        
        logger.info(f"Processing query image: {query_image_path}")
        
        # Extract face from query image
        yield {"type": "status", "message": "Extracting face from query image..."}
        
        try:
            query_faces = self.face_extractor.extract_faces_from_image(query_image_path)
            
            if not query_faces:
                yield {"type": "error", "message": "No face detected in query image"}
                return
            
            if len(query_faces) > 1:
                yield {"type": "warning", "message": f"Multiple faces detected. Using the first one."}
            
            query_encoding = query_faces[0]['face_encoding']
            
        except Exception as e:
            yield {"type": "error", "message": f"Error processing query image: {e}"}
            return
        
        # Find matches
        yield {"type": "status", "message": "Searching for matches..."}
        
        match_count = 0
        cluster_matches = set()
        
        async for match in self.face_matcher.find_matches_async(query_encoding):
            match_count += 1
            
            # Track clusters for summary
            if match.cluster_id is not None:
                cluster_matches.add(match.cluster_id)
            
            # Yield individual match
            yield {
                "type": "match",
                "data": {
                    "image_path": match.image_path,
                    "face_path": match.face_path,
                    "similarity_score": round(match.similarity_score, 3),
                    "bbox": match.bbox,
                    "cluster_id": match.cluster_id,
                    "match_number": match_count
                }
            }
            
            # Allow other tasks to run
            await asyncio.sleep(0.001)
        
        # Yield summary with cluster information
        cluster_info = []
        for cluster_id in cluster_matches:
            if cluster_id in self.face_matcher.clusters:
                cluster = self.face_matcher.clusters[cluster_id]
                cluster_info.append({
                    "cluster_id": cluster_id,
                    "face_count": cluster.face_count,
                    "member_images": cluster.member_images,
                    "representative_face": cluster.representative_face
                })
        
        yield {
            "type": "summary",
            "data": {
                "total_matches": match_count,
                "clusters_matched": len(cluster_matches),
                "cluster_details": cluster_info
            }
        }
        
        logger.info(f"Completed processing. Found {match_count} matches in {len(cluster_matches)} clusters")
    
    async def get_cluster_images(self, cluster_id: int) -> List[str]:
        """Get all images in a specific cluster."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        return self.face_matcher.get_cluster_matches(cluster_id)
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the face database."""
        if not self.is_initialized:
            return {"error": "Database not initialized"}
        
        total_faces = len(self.face_matcher.face_metadata)
        total_images = len(self.face_matcher.face_database)
        total_clusters = len(self.face_matcher.clusters)
        
        # Cluster sizes
        cluster_sizes = [cluster.face_count for cluster in self.face_matcher.clusters.values()]
        avg_cluster_