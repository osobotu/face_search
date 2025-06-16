import cv2
import numpy as np
from typing import List, Tuple, Optional
import face_recognition
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceExtractor:
    """
    Face extraction module that detects and crops faces from images.
    Uses face_recognition library for robust face detection.
    """
    
    def __init__(self, min_face_size: int = 50, face_margin: float = 0.2):
        """
        Initialize face extractor.
        
        Args:
            min_face_size: Minimum face size in pixels
            face_margin: Extra margin around face (0.2 = 20% padding)
        """
        self.min_face_size = min_face_size
        self.face_margin = face_margin
        
    def extract_faces_from_image(self, image_path: str) -> List[dict]:
        """
        Extract all faces from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing face data:
            {
                'face_image': PIL Image of cropped face,
                'face_encoding': face encoding vector,
                'bbox': (top, right, bottom, left) coordinates,
                'confidence': detection confidence (if available)
            }
        """
        try:
            # Load image
            image = face_recognition.load_image_from_file(image_path)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image, model="hog")  # Use "cnn" for better accuracy but slower
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            faces = []
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location
                
                # Check minimum face size
                face_width = right - left
                face_height = bottom - top
                if face_width < self.min_face_size or face_height < self.min_face_size:
                    logger.debug(f"Face {i+1} too small: {face_width}x{face_height}")
                    continue
                
                # Add margin around face
                margin_x = int(face_width * self.face_margin)
                margin_y = int(face_height * self.face_margin)
                
                # Calculate expanded coordinates with bounds checking
                img_height, img_width = image.shape[:2]
                expanded_top = max(0, top - margin_y)
                expanded_bottom = min(img_height, bottom + margin_y)
                expanded_left = max(0, left - margin_x)
                expanded_right = min(img_width, right + margin_x)
                
                # Crop face with margin
                face_image = image[expanded_top:expanded_bottom, expanded_left:expanded_right]
                
                # Convert to PIL Image
                face_pil = Image.fromarray(face_image)
                
                faces.append({
                    'face_image': face_pil,
                    'face_encoding': face_encoding,
                    'bbox': (expanded_top, expanded_right, expanded_bottom, expanded_left),
                    'original_bbox': face_location,
                    'confidence': 1.0  # face_recognition doesn't provide confidence scores
                })
                
            logger.info(f"Extracted {len(faces)} faces from {os.path.basename(image_path)}")
            return faces
            
        except Exception as e:
            logger.error(f"Error extracting faces from {image_path}: {e}")
            return []
    
    def extract_faces_from_directory(self, directory_path: str, supported_formats: List[str] = None) -> dict:
        """
        Extract faces from all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            supported_formats: List of supported file extensions
            
        Returns:
            Dictionary mapping image paths to list of face data
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        all_faces = {}
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return all_faces
        
        # Get all image files
        image_files = []
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in supported_formats):
                image_files.append(os.path.join(directory_path, file))
        
        logger.info(f"Processing {len(image_files)} images from {directory_path}")
        
        for image_path in image_files:
            faces = self.extract_faces_from_image(image_path)
            if faces:
                all_faces[image_path] = faces
        
        total_faces = sum(len(faces) for faces in all_faces.values())
        logger.info(f"Extracted {total_faces} total faces from {len(all_faces)} images")
        
        return all_faces
    
    def save_extracted_faces(self, faces_data: dict, output_dir: str) -> dict:
        """
        Save extracted faces to disk and return metadata.
        
        Args:
            faces_data: Dictionary from extract_faces_from_directory
            output_dir: Directory to save cropped faces
            
        Returns:
            Dictionary mapping original image paths to saved face file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_faces = {}
        
        for image_path, faces in faces_data.items():
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            saved_faces[image_path] = []
            
            for i, face_data in enumerate(faces):
                # Save face image
                face_filename = f"{image_name}_face_{i+1:03d}.jpg"
                face_path = os.path.join(output_dir, face_filename)
                face_data['face_image'].save(face_path, 'JPEG', quality=95)
                
                # Store metadata
                face_metadata = {
                    'face_path': face_path,
                    'face_encoding': face_data['face_encoding'],
                    'bbox': face_data['bbox'],
                    'original_bbox': face_data['original_bbox'],
                    'confidence': face_data['confidence'],
                    'original_image': image_path
                }
                
                saved_faces[image_path].append(face_metadata)
        
        logger.info(f"Saved faces to {output_dir}")
        return saved_faces


# Example usage and testing
if __name__ == "__main__":
    # Initialize face extractor
    extractor = FaceExtractor(min_face_size=80, face_margin=0.3)
    
    # Test with single image
    test_image = "test_image.jpg"  # Replace with actual image path
    if os.path.exists(test_image):
        faces = extractor.extract_faces_from_image(test_image)
        print(f"Found {len(faces)} faces in {test_image}")
        
        # Save faces
        if faces:
            os.makedirs("extracted_faces", exist_ok=True)
            for i, face_data in enumerate(faces):
                face_data['face_image'].save(f"extracted_faces/face_{i+1}.jpg")
            print("Faces saved to extracted_faces/")
    
    # Test with directory
    test_dir = "data/flickr_downloads"  # Your downloaded images directory
    if os.path.exists(test_dir):
        all_faces = extractor.extract_faces_from_directory(test_dir)
        saved_faces = extractor.save_extracted_faces(all_faces, "extracted_faces_batch")
        print(f"Batch extraction complete. Check extracted_faces_batch/ directory")