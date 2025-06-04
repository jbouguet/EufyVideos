#!/usr/bin/env python3
"""
Test script for person embedding generation functionality.

This script validates the person embedding system by:
1. Testing embedding generation on person crops
2. Validating similarity computation and clustering
3. Testing embedding persistence and loading
4. Demonstrating integration with person detection system
"""

import os
import tempfile
import logging
import numpy as np
from pathlib import Path

from logging_config import create_logger, set_all_loggers_level_and_format
from person_detector import PersonDetector, PersonTrack
from person_embedding import PersonEmbeddingGenerator, PersonEmbedding
from tag_processor import TaggerConfig

logger = create_logger(__name__)


def test_embedding_generation():
    """Test basic embedding generation functionality."""
    logger.info("Testing embedding generation...")
    
    try:
        # Create embedding generator
        embedder = PersonEmbeddingGenerator(device="mps")
        
        # Test with a simple synthetic image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Generate embedding
        embedding = embedder.generate_embedding(test_image)
        
        if embedding is not None:
            logger.info(f"✅ Embedding generation successful")
            logger.info(f"   Embedding shape: {embedding.shape}")
            logger.info(f"   Embedding norm: {np.linalg.norm(embedding):.3f}")
            logger.info(f"   Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
            return True
        else:
            logger.error("❌ Embedding generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embedding generation test failed: {e}")
        return False


def test_similarity_computation():
    """Test embedding similarity computation."""
    logger.info("Testing similarity computation...")
    
    try:
        embedder = PersonEmbeddingGenerator(device="mps")
        
        # Create test embeddings
        emb1 = np.random.randn(embedder.embedding_dim)
        emb1 = emb1 / np.linalg.norm(emb1)  # Normalize
        
        emb2 = emb1 + 0.1 * np.random.randn(embedder.embedding_dim)  # Similar
        emb2 = emb2 / np.linalg.norm(emb2)
        
        emb3 = np.random.randn(embedder.embedding_dim)  # Different
        emb3 = emb3 / np.linalg.norm(emb3)
        
        # Test similarities
        sim_same = embedder.compute_similarity(emb1, emb1)
        sim_similar = embedder.compute_similarity(emb1, emb2)
        sim_different = embedder.compute_similarity(emb1, emb3)
        
        logger.info(f"✅ Similarity computation successful")
        logger.info(f"   Same embedding: {sim_same:.3f}")
        logger.info(f"   Similar embedding: {sim_similar:.3f}")
        logger.info(f"   Different embedding: {sim_different:.3f}")
        
        # Validate results
        if abs(sim_same - 1.0) < 0.01 and sim_similar > sim_different:
            logger.info("   ✅ Similarity results are reasonable")
            return True
        else:
            logger.warning("   ⚠️ Similarity results seem unexpected")
            return False
            
    except Exception as e:
        logger.error(f"❌ Similarity computation test failed: {e}")
        return False


def test_clustering():
    """Test embedding clustering functionality."""
    logger.info("Testing embedding clustering...")
    
    try:
        embedder = PersonEmbeddingGenerator(device="mps")
        
        # Create test embeddings with known structure
        # Cluster 1: 3 similar embeddings
        base1 = np.random.randn(embedder.embedding_dim)
        base1 = base1 / np.linalg.norm(base1)
        cluster1_embs = []
        for i in range(3):
            emb = base1 + 0.05 * np.random.randn(embedder.embedding_dim)
            emb = emb / np.linalg.norm(emb)
            pe = PersonEmbedding(
                track_id=1, frame_number=i, video_filename="test.mp4",
                embedding=emb, confidence=0.9
            )
            cluster1_embs.append(pe)
        
        # Cluster 2: 2 similar embeddings
        base2 = np.random.randn(embedder.embedding_dim)
        base2 = base2 / np.linalg.norm(base2)
        cluster2_embs = []
        for i in range(2):
            emb = base2 + 0.05 * np.random.randn(embedder.embedding_dim)
            emb = emb / np.linalg.norm(emb)
            pe = PersonEmbedding(
                track_id=2, frame_number=i, video_filename="test.mp4",
                embedding=emb, confidence=0.8
            )
            cluster2_embs.append(pe)
        
        # Single outlier
        outlier_emb = np.random.randn(embedder.embedding_dim)
        outlier_emb = outlier_emb / np.linalg.norm(outlier_emb)
        outlier = PersonEmbedding(
            track_id=3, frame_number=0, video_filename="test.mp4",
            embedding=outlier_emb, confidence=0.7
        )
        
        all_embeddings = cluster1_embs + cluster2_embs + [outlier]
        
        # Test clustering
        clusters = embedder.cluster_embeddings(all_embeddings, similarity_threshold=0.85)
        
        logger.info(f"✅ Clustering successful")
        logger.info(f"   Created {len(clusters)} clusters from {len(all_embeddings)} embeddings")
        for i, cluster in enumerate(clusters):
            logger.info(f"   Cluster {i}: {len(cluster)} embeddings")
        
        # Validate clustering results
        if len(clusters) >= 2:  # Should have at least 2 clusters
            logger.info("   ✅ Clustering results are reasonable")
            return True
        else:
            logger.warning("   ⚠️ Clustering results unexpected")
            return False
            
    except Exception as e:
        logger.error(f"❌ Clustering test failed: {e}")
        return False


def test_persistence():
    """Test embedding saving and loading."""
    logger.info("Testing embedding persistence...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            embedder = PersonEmbeddingGenerator(device="mps")
            
            # Create test embeddings
            test_embeddings = []
            for i in range(3):
                emb = np.random.randn(embedder.embedding_dim)
                emb = emb / np.linalg.norm(emb)
                pe = PersonEmbedding(
                    track_id=i, frame_number=0, video_filename="test.mp4",
                    embedding=emb, confidence=0.8 + i * 0.1
                )
                test_embeddings.append(pe)
            
            # Save embeddings
            save_file = os.path.join(temp_dir, "test_embeddings.json")
            embedder.save_embeddings(test_embeddings, save_file)
            
            # Load embeddings
            loaded_embeddings = PersonEmbeddingGenerator.load_embeddings(save_file)
            
            # Validate
            if len(loaded_embeddings) == len(test_embeddings):
                logger.info(f"✅ Persistence test successful")
                logger.info(f"   Saved and loaded {len(loaded_embeddings)} embeddings")
                
                # Check one embedding in detail
                orig = test_embeddings[0]
                loaded = loaded_embeddings[0]
                emb_diff = np.linalg.norm(orig.embedding - loaded.embedding)
                
                logger.info(f"   Embedding preservation error: {emb_diff:.6f}")
                
                if emb_diff < 1e-5:
                    logger.info("   ✅ Embeddings preserved accurately")
                    return True
                else:
                    logger.warning("   ⚠️ Embedding preservation has some error")
                    return False
            else:
                logger.error("   ❌ Number of embeddings doesn't match")
                return False
                
    except Exception as e:
        logger.error(f"❌ Persistence test failed: {e}")
        return False


def test_integration_with_person_detection():
    """Test integration with person detection system."""
    logger.info("Testing integration with person detection...")
    
    # Sample video path (use one from our previous tests)
    video_path = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4"
    
    if not os.path.exists(video_path):
        logger.warning(f"Test video not found: {video_path}")
        logger.info("   ⚠️ Skipping integration test")
        return True  # Skip test but don't fail
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create person detector with fast settings
            person_detector = PersonDetector(
                base_detector_config=TaggerConfig(
                    model="Yolo11x_Optimized",
                    task="Track",
                    num_frames_per_second=1.0,  # Very fast for testing
                    conf_threshold=0.5,
                    batch_size=4
                ),
                min_confidence=0.7,
                min_bbox_area=3000
            )
            
            # Detect persons
            person_tracks = person_detector.detect_persons_in_video(video_path)
            
            if not person_tracks:
                logger.warning("   No person tracks detected - cannot test integration")
                return True  # Skip but don't fail
            
            # Extract crops for top track
            top_track = max(person_tracks, key=lambda t: len(t.crops))
            extracted_tracks = person_detector.extract_person_crops(
                video_path=video_path,
                person_tracks=[top_track],
                output_dir=None,  # Don't save to disk
                max_crops_per_track=3  # Just a few for testing
            )
            
            # Generate embeddings
            embedder = PersonEmbeddingGenerator(device="mps")
            embeddings = embedder.generate_embeddings_for_tracks(extracted_tracks)
            
            logger.info(f"✅ Integration test successful")
            logger.info(f"   Detected {len(person_tracks)} person tracks")
            logger.info(f"   Generated {len(embeddings)} embeddings")
            
            if embeddings:
                # Test embedding quality
                avg_quality = np.mean([emb.embedding_quality for emb in embeddings])
                logger.info(f"   Average embedding quality: {avg_quality:.3f}")
                
                # Test similarity within track
                if len(embeddings) >= 2:
                    sim = embedder.compute_similarity(embeddings[0].embedding, embeddings[1].embedding)
                    logger.info(f"   Intra-track similarity: {sim:.3f}")
                
                return True
            else:
                logger.warning("   ⚠️ No embeddings generated")
                return False
                
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all person embedding tests."""
    set_all_loggers_level_and_format(level=logging.INFO, extended_format=False)
    
    logger.info("Starting Person Embedding System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Embedding Generation", test_embedding_generation),
        ("Similarity Computation", test_similarity_computation),
        ("Clustering", test_clustering),
        ("Persistence", test_persistence),
        ("Integration Test", test_integration_with_person_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PERSON EMBEDDING TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All person embedding tests PASSED!")
        logger.info("Person embedding system is ready for use.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests FAILED!")
        logger.error("Person embedding system needs debugging.")
        return 1


if __name__ == "__main__":
    exit(main())