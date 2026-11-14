#!/usr/bin/env python3
"""
Person Embedding Generation Module

This module provides multi-modal embedding generation for person recognition.
It combines CLIP visual embeddings with specialized Person ReID features to create
robust person representations for identification and clustering.

Key Features:
- CLIP-based visual embeddings for general visual understanding
- Person ReID embeddings for person-specific features
- Combined embedding generation with configurable weights
- Embedding similarity computation and clustering
- Integration with person detection and database systems

Example Usage:
    # Create embedding generator
    embedder = PersonEmbedding(device="mps")
    
    # Generate embeddings for person crops
    embeddings = embedder.generate_embeddings(person_crops)
    
    # Compute similarity between embeddings
    similarity = embedder.compute_similarity(emb1, emb2)
    
    # Cluster similar person crops
    clusters = embedder.cluster_embeddings(embeddings, threshold=0.8)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import cv2
from PIL import Image
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("WARNING: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_large
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("WARNING: torchvision not available. Install with: pip install torchvision")

from logging_config import create_logger
from person_detector import PersonCrop, PersonTrack

logger = create_logger(__name__)


@dataclass
class PersonEmbedding:
    """Data structure for person embedding with metadata."""
    
    track_id: int
    frame_number: int
    video_filename: str
    embedding: np.ndarray
    embedding_type: str = "clip+reid"  # Type of embedding used
    confidence: float = 0.0  # Original detection confidence
    embedding_quality: float = 0.0  # Quality score for the embedding
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        # Ensure embedding is numpy array
        if isinstance(self.embedding, torch.Tensor):
            self.embedding = self.embedding.cpu().numpy()
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embedding."""
        return self.embedding.shape[0] if self.embedding is not None else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'track_id': self.track_id,
            'frame_number': self.frame_number,
            'video_filename': self.video_filename,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'embedding_type': self.embedding_type,
            'confidence': self.confidence,
            'embedding_quality': self.embedding_quality,
            'timestamp': self.timestamp,
            'embedding_dim': self.embedding_dim
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonEmbedding':
        """Create from dictionary (JSON deserialization)."""
        embedding = np.array(data['embedding']) if data.get('embedding') is not None else None
        return cls(
            track_id=data['track_id'],
            frame_number=data['frame_number'],
            video_filename=data['video_filename'],
            embedding=embedding,
            embedding_type=data.get('embedding_type', 'clip+reid'),
            confidence=data.get('confidence', 0.0),
            embedding_quality=data.get('embedding_quality', 0.0),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )


class PersonEmbeddingGenerator:
    """
    Multi-modal person embedding generator using CLIP + Person ReID.
    
    This class combines different embedding approaches to create robust
    person representations suitable for identification and clustering.
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 clip_model: str = "ViT-B/32",
                 use_reid: bool = True,
                 embedding_dim: int = 512,
                 reid_weight: float = 0.3,
                 clip_weight: float = 0.7):
        """
        Initialize the person embedding generator.
        
        Args:
            device: Device to run models on ("cpu", "cuda", "mps")
            clip_model: CLIP model variant to use
            use_reid: Whether to include Person ReID features
            embedding_dim: Target dimension for combined embeddings
            reid_weight: Weight for ReID embeddings in combination
            clip_weight: Weight for CLIP embeddings in combination
        """
        self.device = self._get_device(device)
        self.clip_model_name = clip_model
        self.use_reid = use_reid and TORCHVISION_AVAILABLE
        self.embedding_dim = embedding_dim
        self.reid_weight = reid_weight
        self.clip_weight = clip_weight
        
        # Ensure weights sum to 1
        total_weight = self.reid_weight + self.clip_weight
        self.reid_weight /= total_weight
        self.clip_weight /= total_weight
        
        # Initialize models
        self.clip_model = None
        self.clip_preprocess = None
        self.reid_model = None
        self.reid_transform = None
        
        self._initialize_models()
        
        logger.info(f"PersonEmbeddingGenerator initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  CLIP model: {self.clip_model_name}")
        logger.info(f"  Use ReID: {self.use_reid}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Weights - CLIP: {self.clip_weight:.2f}, ReID: {self.reid_weight:.2f}")
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best device to use."""
        if device is not None:
            return device
        
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _initialize_models(self):
        """Initialize CLIP and ReID models."""
        try:
            # Initialize CLIP
            if CLIP_AVAILABLE:
                logger.info("Loading CLIP model...")
                self.clip_model, self.clip_preprocess = clip.load(
                    self.clip_model_name, 
                    device=self.device
                )
                self.clip_model.eval()
                logger.info("✅ CLIP model loaded successfully")
            else:
                logger.warning("❌ CLIP not available - falling back to basic features")
            
            # Initialize Person ReID model (using MobileNetV3 as a lightweight alternative)
            if self.use_reid and TORCHVISION_AVAILABLE:
                logger.info("Loading Person ReID model...")
                self.reid_model = mobilenet_v3_large(pretrained=True)
                # Remove classification head to get feature representations
                self.reid_model.classifier = torch.nn.Identity()
                self.reid_model = self.reid_model.to(self.device)
                self.reid_model.eval()
                
                # ReID preprocessing pipeline
                self.reid_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 128)),  # Standard person ReID size
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                logger.info("✅ Person ReID model loaded successfully")
            else:
                logger.warning("❌ Person ReID not available - using CLIP only")
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def generate_embedding(self, crop_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate combined embedding for a single person crop.
        
        Args:
            crop_image: Person crop as numpy array (H, W, C) in BGR format
            
        Returns:
            Combined embedding vector or None if generation failed
        """
        try:
            embeddings = []
            weights = []
            
            # Convert BGR to RGB for processing
            if len(crop_image.shape) == 3 and crop_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = crop_image
            
            # Generate CLIP embedding
            if self.clip_model is not None:
                clip_emb = self._generate_clip_embedding(rgb_image)
                if clip_emb is not None:
                    embeddings.append(clip_emb)
                    weights.append(self.clip_weight)
            
            # Generate ReID embedding
            if self.reid_model is not None:
                reid_emb = self._generate_reid_embedding(rgb_image)
                if reid_emb is not None:
                    embeddings.append(reid_emb)
                    weights.append(self.reid_weight)
            
            # Combine embeddings
            if embeddings:
                combined = self._combine_embeddings(embeddings, weights)
                return combined
            else:
                logger.warning("No embeddings could be generated")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    def _generate_clip_embedding(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate CLIP visual embedding."""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Preprocess and get embedding
            with torch.no_grad():
                image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
                image_features = self.clip_model.encode_image(image_input)
                image_features = F.normalize(image_features, dim=-1)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"CLIP embedding generation failed: {e}")
            return None
    
    def _generate_reid_embedding(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate Person ReID embedding."""
        try:
            # Preprocess image
            tensor_image = self.reid_transform(rgb_image).unsqueeze(0).to(self.device)
            
            # Generate features
            with torch.no_grad():
                features = self.reid_model(tensor_image)
                features = F.normalize(features, dim=-1)
                
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"ReID embedding generation failed: {e}")
            return None
    
    def _combine_embeddings(self, embeddings: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Combine multiple embeddings with weights."""
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Resize all embeddings to target dimension
        resized_embeddings = []
        for emb in embeddings:
            if len(emb) != self.embedding_dim:
                # Simple resize by repetition or truncation
                if len(emb) < self.embedding_dim:
                    # Pad with repetition
                    repeat_factor = self.embedding_dim // len(emb) + 1
                    repeated = np.tile(emb, repeat_factor)[:self.embedding_dim]
                    resized_embeddings.append(repeated)
                else:
                    # Truncate
                    resized_embeddings.append(emb[:self.embedding_dim])
            else:
                resized_embeddings.append(emb)
        
        # Weighted combination
        combined = np.zeros(self.embedding_dim)
        for emb, weight in zip(resized_embeddings, weights):
            combined += weight * emb
        
        # Normalize final embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined
    
    def generate_embeddings_for_tracks(self, person_tracks: List[PersonTrack]) -> List[PersonEmbedding]:
        """
        Generate embeddings for all crops in person tracks.
        
        Args:
            person_tracks: List of PersonTrack objects with crop images
            
        Returns:
            List of PersonEmbedding objects
        """
        logger.info(f"Generating embeddings for {len(person_tracks)} person tracks...")
        
        embeddings = []
        total_crops = sum(len(track.crops) for track in person_tracks)
        processed_crops = 0
        
        for track in person_tracks:
            logger.info(f"Processing track {track.track_id} with {len(track.crops)} crops")
            
            for crop in track.crops:
                if crop.crop_image is not None:
                    # Generate embedding
                    embedding_vector = self.generate_embedding(crop.crop_image)
                    
                    if embedding_vector is not None:
                        # Calculate embedding quality based on image properties
                        quality = self._assess_embedding_quality(crop.crop_image, embedding_vector)
                        
                        # Create PersonEmbedding object
                        person_embedding = PersonEmbedding(
                            track_id=crop.track_id,
                            frame_number=crop.frame_number,
                            video_filename=crop.video_filename,
                            embedding=embedding_vector,
                            confidence=crop.confidence,
                            embedding_quality=quality
                        )
                        
                        embeddings.append(person_embedding)
                    
                    processed_crops += 1
                    if processed_crops % 10 == 0:
                        logger.info(f"Processed {processed_crops}/{total_crops} crops")
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings from {processed_crops} crops")
        return embeddings
    
    def _assess_embedding_quality(self, crop_image: np.ndarray, embedding: np.ndarray) -> float:
        """
        Assess the quality of an embedding based on image and embedding properties.
        
        Args:
            crop_image: Original crop image
            embedding: Generated embedding vector
            
        Returns:
            Quality score between 0 and 1
        """
        quality_factors = []
        
        # Image size factor (larger crops generally better)
        h, w = crop_image.shape[:2]
        size_score = min(1.0, (h * w) / (224 * 224))  # Normalize to 224x224
        quality_factors.append(size_score)
        
        # Image sharpness (using Laplacian variance)
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY) if len(crop_image.shape) == 3 else crop_image
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 100.0)  # Normalize
        quality_factors.append(sharpness_score)
        
        # Embedding magnitude (well-normalized embeddings should have unit norm)
        emb_norm = np.linalg.norm(embedding)
        norm_score = 1.0 - abs(1.0 - emb_norm)  # Penalty for deviation from unit norm
        quality_factors.append(max(0.0, norm_score))
        
        # Combine factors
        overall_quality = np.mean(quality_factors)
        return float(overall_quality)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Ensure embeddings are normalized
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0
    
    def compute_similarity_matrix(self, embeddings: List[PersonEmbedding]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of embeddings.
        
        Args:
            embeddings: List of PersonEmbedding objects
            
        Returns:
            NxN similarity matrix
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_similarity(embeddings[i].embedding, embeddings[j].embedding)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric matrix
        
        return similarity_matrix
    
    def cluster_embeddings(self, 
                          embeddings: List[PersonEmbedding], 
                          similarity_threshold: float = 0.8) -> List[List[int]]:
        """
        Cluster embeddings based on similarity threshold.
        
        Args:
            embeddings: List of PersonEmbedding objects
            similarity_threshold: Minimum similarity to consider as same person
            
        Returns:
            List of clusters, where each cluster is a list of embedding indices
        """
        n = len(embeddings)
        if n == 0:
            return []
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Simple clustering based on similarity threshold
        clusters = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
                
            # Start new cluster
            cluster = [i]
            assigned.add(i)
            
            # Find similar embeddings
            for j in range(i + 1, n):
                if j not in assigned and similarity_matrix[i, j] >= similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        logger.info(f"Clustered {n} embeddings into {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            logger.info(f"  Cluster {i}: {len(cluster)} embeddings")
        
        return clusters
    
    def save_embeddings(self, embeddings: List[PersonEmbedding], output_file: str):
        """
        Save embeddings to JSON file.
        
        Args:
            embeddings: List of PersonEmbedding objects
            output_file: Path to output JSON file
        """
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_embeddings': len(embeddings),
                'embedding_dim': self.embedding_dim,
                'embedding_type': 'clip+reid' if self.use_reid else 'clip',
                'clip_model': self.clip_model_name,
                'device': self.device
            },
            'embeddings': [emb.to_dict() for emb in embeddings]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {output_file}")
    
    @classmethod
    def load_embeddings(cls, input_file: str) -> List[PersonEmbedding]:
        """
        Load embeddings from JSON file.
        
        Args:
            input_file: Path to JSON file containing embeddings
            
        Returns:
            List of PersonEmbedding objects
        """
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        embeddings = [PersonEmbedding.from_dict(emb_data) for emb_data in data['embeddings']]
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {input_file}")
        return embeddings
    
    def get_track_representative_embedding(self, track_embeddings: List[PersonEmbedding]) -> Optional[PersonEmbedding]:
        """
        Get the most representative embedding for a track.
        
        Args:
            track_embeddings: List of embeddings from the same track
            
        Returns:
            Best representative embedding or None
        """
        if not track_embeddings:
            return None
        
        if len(track_embeddings) == 1:
            return track_embeddings[0]
        
        # Score embeddings based on quality and confidence
        best_embedding = None
        best_score = -1
        
        for emb in track_embeddings:
            # Combined score: quality + confidence + frame centrality
            score = (emb.embedding_quality * 0.5 + 
                    emb.confidence * 0.3 + 
                    0.2)  # Base score for having embedding
            
            if score > best_score:
                best_score = score
                best_embedding = emb
        
        return best_embedding