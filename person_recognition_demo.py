#!/usr/bin/env python3
"""
Person Recognition System Demo

This script demonstrates the complete person recognition pipeline:
1. Person detection with crop extraction
2. Embedding generation using CLIP + ReID
3. Person database management and labeling
4. Similarity computation and clustering
5. Full pipeline integration

Example Usage:
    python person_recognition_demo.py --video path/to/video.mp4 --output output_dir
"""

import os
import argparse
import tempfile
import logging
from pathlib import Path
import json

from logging_config import create_logger, set_all_loggers_level_and_format
from person_detector import PersonDetector, PersonTrack
from person_embedding import PersonEmbeddingGenerator, PersonEmbedding
from person_database import PersonDatabase
from tag_processor import TaggerConfig

logger = create_logger(__name__)


def demonstrate_person_recognition_pipeline(video_path: str, output_dir: str):
    """
    Demonstrate the complete person recognition pipeline.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output files
    """
    logger.info("🚀 Starting Person Recognition Pipeline Demo")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Person Detection with Crop Extraction
    logger.info("\n📹 STEP 1: Person Detection & Crop Extraction")
    logger.info("-" * 60)
    
    person_detector = PersonDetector(
        base_detector_config=TaggerConfig(
            model="Yolo11x_Optimized",
            task="Track",
            num_frames_per_second=2.0,  # Sample every 0.5 seconds
            conf_threshold=0.3,
            batch_size=8
        ),
        crop_size=(224, 224),
        min_confidence=0.6,
        min_bbox_area=2000
    )
    
    # Detect persons in video
    logger.info(f"Processing video: {os.path.basename(video_path)}")
    person_tracks = person_detector.detect_persons_in_video(video_path)
    
    logger.info(f"✅ Found {len(person_tracks)} person tracks:")
    for track in person_tracks:
        logger.info(f"   Track {track.track_id}: {len(track.crops)} detections, "
                   f"avg confidence: {track.confidence_avg:.2f}, "
                   f"duration: {track.duration_frames} frames")
    
    if not person_tracks:
        logger.error("❌ No person tracks found - cannot continue demo")
        return False
    
    # Extract crops for all tracks
    crop_dir = os.path.join(output_dir, "person_crops")
    extracted_tracks = person_detector.extract_person_crops(
        video_path=video_path,
        person_tracks=person_tracks,
        output_dir=crop_dir,
        max_crops_per_track=10
    )
    
    # Save person tracks data
    tracks_file = os.path.join(output_dir, "person_tracks.json")
    person_detector.save_person_tracks(extracted_tracks, tracks_file)
    logger.info(f"💾 Saved person tracks to {tracks_file}")
    
    # Step 2: Embedding Generation
    logger.info("\n🧠 STEP 2: Person Embedding Generation")
    logger.info("-" * 60)
    
    embedder = PersonEmbeddingGenerator(device="mps")
    
    # Generate embeddings for all person tracks
    embeddings = embedder.generate_embeddings_for_tracks(extracted_tracks)
    
    logger.info(f"✅ Generated {len(embeddings)} embeddings:")
    track_embedding_counts = {}
    for emb in embeddings:
        track_embedding_counts[emb.track_id] = track_embedding_counts.get(emb.track_id, 0) + 1
    
    for track_id, count in track_embedding_counts.items():
        avg_quality = sum(emb.embedding_quality for emb in embeddings 
                         if emb.track_id == track_id) / count
        logger.info(f"   Track {track_id}: {count} embeddings, "
                   f"avg quality: {avg_quality:.3f}")
    
    # Save embeddings
    embeddings_file = os.path.join(output_dir, "person_embeddings.json")
    embedder.save_embeddings(embeddings, embeddings_file)
    logger.info(f"💾 Saved embeddings to {embeddings_file}")
    
    # Step 3: Similarity Analysis & Clustering
    logger.info("\n🔍 STEP 3: Similarity Analysis & Clustering")
    logger.info("-" * 60)
    
    # Compute similarity matrix between tracks
    track_ids = list(set(emb.track_id for emb in embeddings))
    
    logger.info("Computing inter-track similarities...")
    similarities = {}
    for i, track1 in enumerate(track_ids):
        for j, track2 in enumerate(track_ids[i+1:], i+1):
            # Get representative embeddings for each track
            track1_embs = [emb for emb in embeddings if emb.track_id == track1]
            track2_embs = [emb for emb in embeddings if emb.track_id == track2]
            
            # Use best quality embedding from each track
            best1 = max(track1_embs, key=lambda e: e.embedding_quality)
            best2 = max(track2_embs, key=lambda e: e.embedding_quality)
            
            similarity = embedder.compute_similarity(best1.embedding, best2.embedding)
            similarities[(track1, track2)] = similarity
            
            logger.info(f"   Track {track1} vs Track {track2}: {similarity:.3f}")
    
    # Cluster embeddings
    clusters = embedder.cluster_embeddings(embeddings, similarity_threshold=0.75)
    
    logger.info(f"✅ Created {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        track_ids_in_cluster = list(set(embeddings[idx].track_id for idx in cluster))
        logger.info(f"   Cluster {i}: {len(cluster)} embeddings from tracks {track_ids_in_cluster}")
    
    # Step 4: Person Database Management
    logger.info("\n👥 STEP 4: Person Database Management")
    logger.info("-" * 60)
    
    # Create person database
    db_file = os.path.join(output_dir, "persons.json")
    db = PersonDatabase(db_file)
    
    # Add sample family members (demo data)
    logger.info("Adding sample person identities...")
    jean_yves_id = db.add_person(
        "Jean-Yves Bouguet", 
        description="Father, homeowner",
        aliases=["JY", "Dad"]
    )
    
    chittra_id = db.add_person(
        "Chittra Chaivorapol",
        description="Mother", 
        aliases=["Mom", "Chittra"]
    )
    
    lucas_id = db.add_person(
        "Lucas Bouguet",
        description="Son",
        aliases=["Luke"]
    )
    
    # Demonstrate manual labeling (simulated)
    video_filename = os.path.basename(video_path)
    if len(person_tracks) >= 1:
        db.label_track(video_filename, person_tracks[0].track_id, jean_yves_id,
                      notes="Demo labeling - first track")
        logger.info(f"   Labeled track {person_tracks[0].track_id} as Jean-Yves")
    
    if len(person_tracks) >= 2:
        db.label_track(video_filename, person_tracks[1].track_id, chittra_id,
                      confidence=0.85, labeled_by="manual",
                      notes="Demo labeling - second track")
        logger.info(f"   Labeled track {person_tracks[1].track_id} as Chittra")
    
    # Get database statistics
    stats = db.get_database_statistics()
    logger.info(f"✅ Database statistics:")
    logger.info(f"   Total persons: {stats['total_persons']}")
    logger.info(f"   Total labeled tracks: {stats['total_tracks']}")
    logger.info(f"   Manual labels: {stats['manual_labels']}")
    logger.info(f"   Persons: {', '.join(stats['persons_list'])}")
    
    # Step 5: Complete Pipeline Summary
    logger.info("\n📊 STEP 5: Pipeline Summary & Results")
    logger.info("-" * 60)
    
    # Create summary report
    summary = {
        'video_file': os.path.basename(video_path),
        'processing_results': {
            'person_tracks_detected': len(person_tracks),
            'total_detections': sum(len(track.crops) for track in person_tracks),
            'embeddings_generated': len(embeddings),
            'clusters_formed': len(clusters),
            'persons_in_database': stats['total_persons'],
            'labeled_tracks': stats['total_tracks']
        },
        'quality_metrics': {
            'average_detection_confidence': sum(track.confidence_avg for track in person_tracks) / len(person_tracks),
            'average_embedding_quality': sum(emb.embedding_quality for emb in embeddings) / len(embeddings),
            'highest_inter_track_similarity': max(similarities.values()) if similarities else 0.0
        },
        'files_created': {
            'person_tracks': os.path.basename(tracks_file),
            'embeddings': os.path.basename(embeddings_file),
            'database': os.path.basename(db_file),
            'crop_images': f"{len(list(Path(crop_dir).glob('*.jpg')))} files" if os.path.exists(crop_dir) else "0 files"
        }
    }
    
    # Save summary report
    summary_file = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Pipeline completed successfully!")
    logger.info(f"   Processed: {summary['processing_results']['person_tracks_detected']} person tracks")
    logger.info(f"   Generated: {summary['processing_results']['embeddings_generated']} embeddings")
    logger.info(f"   Quality: {summary['quality_metrics']['average_embedding_quality']:.3f} avg")
    logger.info(f"   Output: {len(os.listdir(output_dir))} files in {output_dir}")
    
    logger.info(f"\n📁 Output Files Created:")
    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"   {file:<25} ({size_mb:.2f} MB)")
        else:
            file_count = len(os.listdir(file_path)) if os.path.isdir(file_path) else 0
            logger.info(f"   {file}/<25 ({file_count} files)")
    
    return True


def main():
    """Main function for person recognition demo."""
    parser = argparse.ArgumentParser(description="Person Recognition System Demo")
    parser.add_argument("--video", type=str, 
                       default="/Users/jbouguet/Documents/EufySecurityVideos/record/Batch022/T8600P102338033E_20240930085536.mp4",
                       help="Path to input video file")
    parser.add_argument("--output", type=str, 
                       default="person_recognition_demo_output",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    set_all_loggers_level_and_format(level=log_level, extended_format=False)
    
    # Validate input video
    if not os.path.exists(args.video):
        logger.error(f"❌ Video file not found: {args.video}")
        logger.error("Please provide a valid video file path with --video")
        return 1
    
    # Run demo
    try:
        success = demonstrate_person_recognition_pipeline(args.video, args.output)
        
        if success:
            logger.info("\n🎉 Person Recognition Demo Completed Successfully!")
            logger.info(f"🔍 Check output files in: {args.output}")
            logger.info("📋 Review pipeline_summary.json for detailed results")
            return 0
        else:
            logger.error("\n💥 Person Recognition Demo Failed!")
            return 1
            
    except Exception as e:
        logger.error(f"\n💥 Demo failed with exception: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())