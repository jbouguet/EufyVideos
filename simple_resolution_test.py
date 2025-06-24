#!/usr/bin/env python3
"""
Simple Resolution Test

This script manually tests higher resolution embeddings by:
1. Taking some existing person crops
2. Resizing them to different resolutions (224x224 vs 384x384)  
3. Generating embeddings for both resolutions
4. Comparing clustering quality

This avoids the complexity of running the full pipeline.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json

from person_embedding import PersonEmbeddingGenerator, PersonEmbedding
from enhanced_person_clustering import EnhancedPersonClusterer
from logging_config import create_logger

logger = create_logger(__name__)


def find_sample_crops(crops_dir: str, max_crops: int = 100) -> List[str]:
    """Find sample crop images for testing."""
    crop_files = []
    
    for video_dir in Path(crops_dir).iterdir():
        if video_dir.is_dir():
            for crop_file in video_dir.glob("*.jpg"):
                crop_files.append(str(crop_file))
                if len(crop_files) >= max_crops:
                    break
            if len(crop_files) >= max_crops:
                break
    
    logger.info(f"Found {len(crop_files)} sample crop images")
    return crop_files


def resize_and_generate_embeddings(crop_files: List[str], 
                                 resolution: Tuple[int, int],
                                 embedder: PersonEmbeddingGenerator) -> List[PersonEmbedding]:
    """Resize crops to specified resolution and generate embeddings."""
    logger.info(f"Generating embeddings at {resolution[0]}x{resolution[1]} resolution...")
    
    embeddings = []
    
    for i, crop_file in enumerate(crop_files):
        try:
            # Load image
            image = cv2.imread(crop_file)
            if image is None:
                continue
            
            # Resize to target resolution
            resized = cv2.resize(image, resolution)
            
            # Convert BGR to RGB for embedding generation
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Generate embedding
            embedding_vector = embedder.generate_embedding(rgb_image)
            
            if embedding_vector is not None:
                # Extract metadata from filename
                filename = Path(crop_file).name
                # Parse filename like: video_frame-123456_track-001.jpg
                parts = filename.replace('.jpg', '').split('_')
                frame_num = 0
                track_id = i  # Use index as track ID
                
                try:
                    for part in parts:
                        if 'frame-' in part:
                            frame_num = int(part.split('-')[1])
                        elif 'track-' in part:
                            track_id = int(part.split('-')[1])
                        elif 'detect-' in part:
                            track_id = int(part.split('-')[1])
                except:
                    pass  # Use defaults
                
                # Create embedding object
                embedding_obj = PersonEmbedding(
                    track_id=track_id,
                    frame_number=frame_num,
                    video_filename=Path(crop_file).parent.name,
                    embedding=embedding_vector,
                    embedding_type=f"clip+reid_{resolution[0]}x{resolution[1]}",
                    confidence=0.8,  # Dummy value
                    embedding_quality=0.8  # Dummy value, will be computed later
                )
                
                embeddings.append(embedding_obj)
        
        except Exception as e:
            logger.debug(f"Failed to process {crop_file}: {e}")
            continue
    
    logger.info(f"Generated {len(embeddings)} embeddings at {resolution[0]}x{resolution[1]}")
    return embeddings


def calculate_embedding_quality(embeddings: List[PersonEmbedding]) -> List[PersonEmbedding]:
    """Calculate quality scores for embeddings based on their characteristics."""
    if not embeddings:
        return embeddings
    
    # Calculate embedding variance as a proxy for quality
    for embedding in embeddings:
        if embedding.embedding is not None:
            # Higher variance = more informative embedding
            variance = np.var(embedding.embedding)
            # Normalize to 0-1 range (this is heuristic)
            quality = min(1.0, variance * 10)  # Scale factor is heuristic
            embedding.embedding_quality = quality
    
    return embeddings


def cluster_and_analyze(embeddings: List[PersonEmbedding], resolution_name: str) -> dict:
    """Cluster embeddings and return analysis results."""
    logger.info(f"Clustering {len(embeddings)} embeddings for {resolution_name}")
    
    if not embeddings:
        return {
            'resolution': resolution_name,
            'num_embeddings': 0,
            'num_clusters': 0,
            'avg_cluster_size': 0,
            'avg_quality': 0,
            'min_quality': 0,
            'max_quality': 0,
            'largest_cluster': 0
        }
    
    # Calculate quality scores
    embeddings = calculate_embedding_quality(embeddings)
    
    # Use conservative clustering settings
    clusterer = EnhancedPersonClusterer(
        similarity_threshold=0.88,  # Slightly more permissive for this test
        quality_threshold=0.3,     # Lower quality threshold for test data
        use_dbscan=False,          # Use hierarchical
        min_cluster_size=2
    )
    
    # Create temporary directory and save embeddings
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings_file = os.path.join(temp_dir, "test_embeddings.json")
        
        # Save embeddings
        embeddings_data = {
            'embeddings': [emb.to_dict() for emb in embeddings],
            'metadata': {
                'resolution': resolution_name,
                'total_embeddings': len(embeddings)
            }
        }
        
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        # Load and cluster
        loaded_embeddings = PersonEmbeddingGenerator.load_embeddings(embeddings_file)
        clusters = clusterer.cluster_embeddings_enhanced(loaded_embeddings)
    
    if not clusters:
        logger.warning(f"No clusters found for {resolution_name}")
        return {
            'resolution': resolution_name,
            'num_embeddings': len(embeddings),
            'num_clusters': 0,
            'avg_cluster_size': 0,
            'avg_quality': 0,
            'min_quality': 0,
            'max_quality': 0,
            'largest_cluster': 0
        }
    
    # Calculate statistics
    total_clustered = sum(c.cluster_size for c in clusters)
    avg_cluster_size = total_clustered / len(clusters)
    qualities = [c.quality_score for c in clusters]
    sizes = [c.cluster_size for c in clusters]
    
    results = {
        'resolution': resolution_name,
        'num_embeddings': len(embeddings),
        'num_clusters': len(clusters),
        'avg_cluster_size': avg_cluster_size,
        'avg_quality': sum(qualities) / len(qualities),
        'min_quality': min(qualities),
        'max_quality': max(qualities),
        'largest_cluster': max(sizes)
    }
    
    logger.info(f"{resolution_name} clustering results:")
    logger.info(f"  Embeddings: {results['num_embeddings']}")
    logger.info(f"  Clusters: {results['num_clusters']}")
    logger.info(f"  Avg quality: {results['avg_quality']:.3f}")
    logger.info(f"  Min quality: {results['min_quality']:.3f}")
    
    return results


def compare_resolutions():
    """Compare clustering results at different resolutions."""
    logger.info("=" * 60)
    logger.info("SIMPLE RESOLUTION COMPARISON TEST")
    logger.info("=" * 60)
    
    crops_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/crops"
    
    if not os.path.exists(crops_dir):
        logger.error(f"Crops directory not found: {crops_dir}")
        return
    
    # Find sample crops
    crop_files = find_sample_crops(crops_dir, max_crops=50)  # Test with subset
    
    if not crop_files:
        logger.error("No crop files found for testing")
        return
    
    # Initialize embedding generator
    try:
        embedder = PersonEmbeddingGenerator(
            device="mps",  # Use Mac GPU
            embedding_dim=512,
            reid_weight=0.3,
            clip_weight=0.7
        )
        logger.info("Embedding generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {e}")
        return
    
    # Test different resolutions
    resolutions = [
        (224, 224),  # Current baseline
        (384, 384),  # Higher resolution
    ]
    
    results = []
    
    for resolution in resolutions:
        logger.info(f"\nTesting resolution: {resolution[0]}x{resolution[1]}")
        
        try:
            # Generate embeddings at this resolution
            embeddings = resize_and_generate_embeddings(crop_files, resolution, embedder)
            
            # Cluster and analyze
            result = cluster_and_analyze(embeddings, f"{resolution[0]}x{resolution[1]}")
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to test resolution {resolution}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    if len(results) >= 2:
        analyze_comparison(results[0], results[1])
    else:
        logger.error("Not enough results to compare")


def analyze_comparison(baseline, high_res):
    """Analyze and report comparison between resolutions."""
    logger.info("\n" + "=" * 70)
    logger.info("RESOLUTION COMPARISON ANALYSIS")
    logger.info("=" * 70)
    
    # Comparison table
    logger.info(f"\n{'Metric':<20} {'224x224':<15} {'384x384':<15} {'Change':<15}")
    logger.info("-" * 70)
    
    logger.info(f"{'Embeddings':<20} {baseline['num_embeddings']:<15} {high_res['num_embeddings']:<15} "
               f"{high_res['num_embeddings'] - baseline['num_embeddings']:+d}")
    
    logger.info(f"{'Clusters':<20} {baseline['num_clusters']:<15} {high_res['num_clusters']:<15} "
               f"{high_res['num_clusters'] - baseline['num_clusters']:+d}")
    
    # Avoid division by zero
    if baseline['avg_quality'] > 0 and high_res['avg_quality'] > 0:
        logger.info(f"{'Avg Quality':<20} {baseline['avg_quality']:<15.3f} {high_res['avg_quality']:<15.3f} "
                   f"{high_res['avg_quality'] - baseline['avg_quality']:+.3f}")
        
        logger.info(f"{'Min Quality':<20} {baseline['min_quality']:<15.3f} {high_res['min_quality']:<15.3f} "
                   f"{high_res['min_quality'] - baseline['min_quality']:+.3f}")
    
    logger.info(f"{'Largest Cluster':<20} {baseline['largest_cluster']:<15} {high_res['largest_cluster']:<15} "
               f"{high_res['largest_cluster'] - baseline['largest_cluster']:+d}")
    
    # Analysis
    logger.info("\n" + "=" * 50)
    logger.info("ANALYSIS & RECOMMENDATIONS")
    logger.info("=" * 50)
    
    if high_res['avg_quality'] > 0 and baseline['avg_quality'] > 0:
        quality_improvement = high_res['avg_quality'] - baseline['avg_quality']
        precision_improvement = high_res['min_quality'] - baseline['min_quality']
        
        if quality_improvement > 0.01:  # More than 1% improvement
            logger.info(f"✅ QUALITY IMPROVEMENT: +{quality_improvement:.3f}")
            logger.info(f"✅ Higher resolution (384x384) shows better clustering")
            logger.info(f"✅ Recommendation: Consider switching to 384x384")
            
        elif quality_improvement > 0.005:  # Small improvement
            logger.info(f"⚖️  SMALL IMPROVEMENT: +{quality_improvement:.3f}")
            logger.info(f"⚖️  Higher resolution provides modest benefit")
            
        else:
            logger.info(f"📊 NO SIGNIFICANT IMPROVEMENT: {quality_improvement:+.3f}")
            logger.info(f"📊 224x224 resolution appears sufficient")
    
    # Cluster analysis
    cluster_change = high_res['num_clusters'] - baseline['num_clusters']
    if cluster_change > 0:
        logger.info(f"\n📈 MORE CLUSTERS: +{cluster_change} clusters")
        logger.info(f"   Higher resolution enables finer discrimination")
    elif cluster_change < 0:
        logger.info(f"\n📉 FEWER CLUSTERS: {cluster_change} clusters") 
        logger.info(f"   Higher resolution groups more crops together")
    
    # Final recommendation
    logger.info(f"\n🎯 TEST CONCLUSION:")
    if high_res['avg_quality'] > baseline['avg_quality'] + 0.01:
        logger.info(f"   Higher resolution shows meaningful improvement")
        logger.info(f"   Consider updating to person_crop_size: [384, 384]")
    else:
        logger.info(f"   Current 224x224 resolution is adequate")
        logger.info(f"   No strong evidence for switching to higher resolution")
    
    logger.info(f"\n📝 NOTE: This is a limited test with {baseline['num_embeddings']} crops")
    logger.info(f"   Results may vary with your full dataset")


def main():
    """Run simple resolution comparison test."""
    
    logger.info("🔬 SIMPLE RESOLUTION TEST")
    logger.info("=" * 50)
    logger.info("Testing crop resolution impact on clustering")
    logger.info("This uses existing crops resized to different resolutions")
    logger.info("=" * 50)
    
    try:
        compare_resolutions()
        
        logger.info(f"\n✅ RESOLUTION TEST COMPLETED")
        
        return 0
        
    except Exception as e:
        logger.error(f"Resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())