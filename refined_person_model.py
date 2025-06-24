#!/usr/bin/env python3
"""
Refined Person Recognition Model

This module implements an advanced person recognition model that learns from
manually labeled mega-clusters to provide accurate auto-labeling of new detections.
It builds upon the conservative clustering results and human-verified labels to
create a robust identification system.

Key Features:
- Siamese neural network for person similarity learning
- Meta-learning from labeled mega-clusters
- Real-time auto-labeling of new person detections
- Confidence-based prediction with uncertainty estimation
- Integration with existing person recognition pipeline

Architecture:
1. Load training data from labeled mega-clusters
2. Train refined similarity model using Siamese networks
3. Create person-specific embeddings and thresholds
4. Provide auto-labeling functionality for new detections
5. Continuous learning from user feedback

Usage:
    # Train the refined model
    python refined_person_model.py --train
    
    # Auto-label new detections
    python refined_person_model.py --predict
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pickle
from pathlib import Path

from person_embedding import PersonEmbedding, PersonEmbeddingGenerator
from cluster_labeling_tool import MegaCluster
from logging_config import create_logger

logger = create_logger(__name__)


@dataclass
class PersonProfile:
    """Profile for a known person with learned characteristics."""
    
    person_name: str
    mega_cluster_id: str
    representative_embedding: np.ndarray
    embedding_mean: np.ndarray
    embedding_std: np.ndarray
    similarity_threshold: float
    confidence_threshold: float = 0.8
    total_training_samples: int = 0
    last_updated: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'person_name': self.person_name,
            'mega_cluster_id': self.mega_cluster_id,
            'representative_embedding': self.representative_embedding.tolist(),
            'embedding_mean': self.embedding_mean.tolist(),
            'embedding_std': self.embedding_std.tolist(),
            'similarity_threshold': float(self.similarity_threshold),
            'confidence_threshold': float(self.confidence_threshold),
            'total_training_samples': self.total_training_samples,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonProfile':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            person_name=data['person_name'],
            mega_cluster_id=data['mega_cluster_id'],
            representative_embedding=np.array(data['representative_embedding']),
            embedding_mean=np.array(data['embedding_mean']),
            embedding_std=np.array(data['embedding_std']),
            similarity_threshold=data['similarity_threshold'],
            confidence_threshold=data.get('confidence_threshold', 0.8),
            total_training_samples=data.get('total_training_samples', 0),
            last_updated=data.get('last_updated', '')
        )


class SiameseNetwork(nn.Module):
    """
    Siamese neural network for learning person similarity.
    
    Takes two embeddings and outputs a similarity score between 0 and 1.
    """
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Shared embedding transformation
        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Similarity computation
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Siamese network.
        
        Args:
            embedding1: First embedding tensor [batch_size, embedding_dim]
            embedding2: Second embedding tensor [batch_size, embedding_dim]
            
        Returns:
            Similarity scores [batch_size, 1]
        """
        # Transform embeddings
        emb1_transformed = self.embedding_net(embedding1)
        emb2_transformed = self.embedding_net(embedding2)
        
        # Compute element-wise differences and products
        diff = torch.abs(emb1_transformed - emb2_transformed)
        prod = emb1_transformed * emb2_transformed
        
        # Concatenate features
        combined = torch.cat([diff, prod], dim=1)
        
        # Compute similarity
        similarity = self.similarity_net(combined)
        
        return similarity


class PersonSimilarityDataset(Dataset):
    """Dataset for training person similarity model."""
    
    def __init__(self, training_data: Dict[str, Any]):
        """
        Initialize dataset from training data.
        
        Args:
            training_data: Dictionary containing positive and negative pairs
        """
        self.pairs = []
        self.labels = []
        
        # Add positive pairs (same person)
        for pair in training_data['positive_pairs']:
            emb1 = np.array(pair['embedding1']['embedding'])
            emb2 = np.array(pair['embedding2']['embedding'])
            self.pairs.append((emb1, emb2))
            self.labels.append(1.0)
        
        # Add negative pairs (different people)
        for pair in training_data['negative_pairs']:
            emb1 = np.array(pair['embedding1']['embedding'])
            emb2 = np.array(pair['embedding2']['embedding'])
            self.pairs.append((emb1, emb2))
            self.labels.append(0.0)
        
        logger.info(f"Dataset created with {len(self.pairs)} pairs")
        logger.info(f"  Positive pairs: {sum(self.labels)}")
        logger.info(f"  Negative pairs: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb1, emb2 = self.pairs[idx]
        label = self.labels[idx]
        
        return (
            torch.FloatTensor(emb1),
            torch.FloatTensor(emb2),
            torch.FloatTensor([label])
        )


class RefinedPersonRecognitionModel:
    """
    Refined person recognition model with auto-labeling capabilities.
    
    Combines Siamese neural networks with statistical analysis to provide
    accurate person identification for new detections.
    """
    
    def __init__(self, 
                 model_dir: str,
                 embedding_dim: int = 512,
                 device: Optional[str] = None):
        """
        Initialize the refined person recognition model.
        
        Args:
            model_dir: Directory to save/load model files
            embedding_dim: Dimension of input embeddings
            device: Device to run model on
        """
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        self.device = self._get_device(device)
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.siamese_model = SiameseNetwork(embedding_dim)
        self.siamese_model.to(self.device)
        
        # Person profiles
        self.person_profiles: Dict[str, PersonProfile] = {}
        
        # File paths
        self.model_file = os.path.join(model_dir, "siamese_model.pth")
        self.profiles_file = os.path.join(model_dir, "person_profiles.json")
        self.training_log_file = os.path.join(model_dir, "training_log.json")
        
        logger.info(f"RefinedPersonRecognitionModel initialized:")
        logger.info(f"  Model dir: {model_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Embedding dim: {embedding_dim}")
    
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
    
    def train_from_labeled_data(self, 
                              training_data_file: str,
                              epochs: int = 50,
                              batch_size: int = 32,
                              learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train the refined model from labeled training data.
        
        Args:
            training_data_file: Path to training data JSON file
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            
        Returns:
            Training metrics and results
        """
        logger.info("=" * 60)
        logger.info("TRAINING REFINED PERSON RECOGNITION MODEL")
        logger.info("=" * 60)
        
        # Load training data
        with open(training_data_file, 'r') as f:
            training_data = json.load(f)
        
        logger.info(f"Loaded training data:")
        logger.info(f"  Positive pairs: {training_data['metadata']['positive_pairs']}")
        logger.info(f"  Negative pairs: {training_data['metadata']['negative_pairs']}")
        logger.info(f"  Mega-clusters: {training_data['metadata']['total_mega_clusters']}")
        
        # Create dataset and dataloader
        dataset = PersonSimilarityDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.siamese_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training loop
        training_log = []
        best_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.siamese_model.train()
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, (emb1, emb2, labels) in enumerate(dataloader):
                emb1 = emb1.to(self.device)
                emb2 = emb2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.siamese_model(emb1, emb2)
                loss = criterion(predictions, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                predicted_labels = (predictions > 0.5).float()
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)
            
            scheduler.step()
            
            # Calculate metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            training_log.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.siamese_model.state_dict(), self.model_file)
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                           f"Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Save training log
        log_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'best_loss': best_loss,
                'final_accuracy': training_log[-1]['accuracy']
            },
            'training_log': training_log
        }
        
        with open(self.training_log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Training completed!")
        logger.info(f"  Best loss: {best_loss:.4f}")
        logger.info(f"  Final accuracy: {training_log[-1]['accuracy']:.4f}")
        logger.info(f"  Model saved: {self.model_file}")
        
        # Create person profiles
        self._create_person_profiles(training_data)
        
        return log_data
    
    def _create_person_profiles(self, training_data: Dict[str, Any]):
        """Create person profiles from training data."""
        logger.info("Creating person profiles...")
        
        # Group embeddings by mega-cluster
        mega_embeddings = {}
        for mega_id, info in training_data['mega_clusters_summary'].items():
            mega_embeddings[mega_id] = {
                'person_name': info['person_name'],
                'embeddings': []
            }
        
        # Collect embeddings
        for pair in training_data['positive_pairs']:
            mega_id = pair['mega_cluster_id']
            emb1 = np.array(pair['embedding1']['embedding'])
            emb2 = np.array(pair['embedding2']['embedding'])
            mega_embeddings[mega_id]['embeddings'].extend([emb1, emb2])
        
        # Create profiles
        self.person_profiles = {}
        
        for mega_id, data in mega_embeddings.items():
            if len(data['embeddings']) > 0:
                embeddings = np.array(data['embeddings'])
                
                # Calculate statistics
                mean_embedding = np.mean(embeddings, axis=0)
                std_embedding = np.std(embeddings, axis=0)
                
                # Calculate similarity threshold using intra-cluster distances
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(sim)
                
                # Use conservative threshold (mean - 2*std)
                if similarities:
                    sim_mean = np.mean(similarities)
                    sim_std = np.std(similarities)
                    threshold = max(0.7, sim_mean - 2 * sim_std)  # Minimum threshold of 0.7
                else:
                    threshold = 0.8  # Default threshold
                
                # Create profile
                profile = PersonProfile(
                    person_name=data['person_name'],
                    mega_cluster_id=mega_id,
                    representative_embedding=mean_embedding,
                    embedding_mean=mean_embedding,
                    embedding_std=std_embedding,
                    similarity_threshold=threshold,
                    total_training_samples=len(embeddings),
                    last_updated=datetime.now().isoformat()
                )
                
                self.person_profiles[data['person_name']] = profile
        
        # Save profiles
        self.save_person_profiles()
        
        logger.info(f"Created {len(self.person_profiles)} person profiles")
        for name, profile in self.person_profiles.items():
            logger.info(f"  {name}: threshold={profile.similarity_threshold:.3f}, "
                       f"samples={profile.total_training_samples}")
    
    def save_person_profiles(self):
        """Save person profiles to file."""
        profiles_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_profiles': len(self.person_profiles)
            },
            'profiles': {
                name: profile.to_dict() 
                for name, profile in self.person_profiles.items()
            }
        }
        
        with open(self.profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)
        
        logger.info(f"Saved {len(self.person_profiles)} profiles to {self.profiles_file}")
    
    def load_person_profiles(self):
        """Load person profiles from file."""
        if os.path.exists(self.profiles_file):
            with open(self.profiles_file, 'r') as f:
                profiles_data = json.load(f)
            
            self.person_profiles = {}
            for name, profile_data in profiles_data['profiles'].items():
                self.person_profiles[name] = PersonProfile.from_dict(profile_data)
            
            logger.info(f"Loaded {len(self.person_profiles)} person profiles")
        else:
            logger.warning("No person profiles file found")
    
    def load_trained_model(self):
        """Load trained Siamese model."""
        if os.path.exists(self.model_file):
            self.siamese_model.load_state_dict(torch.load(self.model_file, map_location=self.device))
            self.siamese_model.eval()
            logger.info(f"Loaded trained model from {self.model_file}")
        else:
            logger.warning("No trained model file found")
    
    def predict_person(self, embedding: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Predict the person for a given embedding.
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Tuple of (predicted_person_name, confidence, all_similarities)
        """
        if not self.person_profiles:
            return None, 0.0, {}
        
        self.siamese_model.eval()
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        
        similarities = {}
        best_person = None
        best_similarity = 0.0
        
        with torch.no_grad():
            for person_name, profile in self.person_profiles.items():
                # Get representative embedding
                profile_embedding = torch.FloatTensor(profile.representative_embedding).unsqueeze(0).to(self.device)
                
                # Compute similarity using Siamese network
                similarity = self.siamese_model(embedding_tensor, profile_embedding).item()
                similarities[person_name] = similarity
                
                # Check if this is the best match above threshold
                if similarity > profile.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_person = person_name
        
        # Calculate confidence based on margin to second-best match
        sorted_sims = sorted(similarities.values(), reverse=True)
        if len(sorted_sims) >= 2:
            margin = sorted_sims[0] - sorted_sims[1]
            confidence = min(1.0, best_similarity + margin)
        else:
            confidence = best_similarity
        
        return best_person, confidence, similarities
    
    def auto_label_embeddings(self, embeddings: List[PersonEmbedding]) -> List[Dict[str, Any]]:
        """
        Auto-label a list of embeddings.
        
        Args:
            embeddings: List of PersonEmbedding objects
            
        Returns:
            List of prediction results
        """
        results = []
        
        logger.info(f"Auto-labeling {len(embeddings)} embeddings...")
        
        for embedding in embeddings:
            predicted_person, confidence, similarities = self.predict_person(embedding.embedding)
            
            result = {
                'embedding_id': f"{embedding.video_filename}_{embedding.frame_number}_{embedding.track_id}",
                'predicted_person': predicted_person,
                'confidence': confidence,
                'similarities': similarities,
                'embedding': embedding.to_dict()
            }
            
            results.append(result)
        
        # Summary statistics
        total = len(results)
        labeled = sum(1 for r in results if r['predicted_person'] is not None)
        high_confidence = sum(1 for r in results if r['confidence'] > 0.8)
        
        logger.info(f"Auto-labeling results:")
        logger.info(f"  Total embeddings: {total}")
        logger.info(f"  Successfully labeled: {labeled} ({labeled/total*100:.1f}%)")
        logger.info(f"  High confidence (>0.8): {high_confidence} ({high_confidence/total*100:.1f}%)")
        
        return results


def main():
    """Run refined person recognition model training or prediction."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Refined Person Recognition Model')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                       help='Mode to run: train or predict')
    parser.add_argument('--training-data', 
                       default='/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/cluster_labeling/training_data.json',
                       help='Path to training data file')
    parser.add_argument('--model-dir',
                       default='/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/refined_model',
                       help='Directory for model files')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    try:
        # Initialize model
        model = RefinedPersonRecognitionModel(args.model_dir)
        
        if args.mode == 'train':
            if not os.path.exists(args.training_data):
                logger.error(f"Training data file not found: {args.training_data}")
                logger.info("Please run cluster labeling tool first to generate training data")
                return 1
            
            # Train model
            training_results = model.train_from_labeled_data(
                args.training_data,
                epochs=args.epochs
            )
            
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Model saved to: {args.model_dir}")
            logger.info(f"Final accuracy: {training_results['metadata']['final_accuracy']:.4f}")
            logger.info(f"Person profiles: {len(model.person_profiles)}")
            
        elif args.mode == 'predict':
            # Load trained model and profiles
            model.load_trained_model()
            model.load_person_profiles()
            
            if not model.person_profiles:
                logger.error("No person profiles found. Please train the model first.")
                return 1
            
            logger.info("=" * 60)
            logger.info("REFINED PERSON RECOGNITION MODEL READY")
            logger.info("=" * 60)
            logger.info(f"Loaded {len(model.person_profiles)} person profiles:")
            for name, profile in model.person_profiles.items():
                logger.info(f"  {name}: threshold={profile.similarity_threshold:.3f}")
            
            logger.info("\nModel is ready for auto-labeling new detections!")
            logger.info("Integrate with your video processing pipeline using:")
            logger.info("  predicted_person, confidence, _ = model.predict_person(embedding)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Refined model operation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())