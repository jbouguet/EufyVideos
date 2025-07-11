#!/usr/bin/env python3
"""
Track-Level Clustering for Manual Labeling

This module creates individual clusters for each unique track_id to enable 
manual labeling and grouping of person detections. Each track becomes its own
cluster, providing thousands of individual clusters that can be manually labeled
with person names to create bigger clusters for better model training.

Key Features:
- One cluster per unique track_id (no track mixing)
- Maintains track integrity from original detections
- Enables granular manual labeling at track level
- Provides foundation for creating person-named mega-clusters
- Supports batch processing and cluster export

Workflow:
1. Load person embeddings from all videos
2. Create one cluster per unique (video, track_id) combination
3. Generate cluster thumbnails and metadata
4. Export clusters for manual labeling tools
5. Support later grouping into person-named mega-clusters

This approach ensures maximum granularity for manual labeling while maintaining
the integrity of the original tracking results.
"""

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from person_embedding import PersonEmbedding, PersonEmbeddingGenerator
from logging_config import create_logger

logger = create_logger(__name__)


@dataclass
class TrackLevelCluster:
    """Individual cluster representing one unique track."""
    
    cluster_id: str  # Unique identifier
    video_filename: str
    track_id: str
    embeddings: List[PersonEmbedding] = field(default_factory=list)
    representative_embedding: Optional[PersonEmbedding] = None
    cluster_size: int = 0
    avg_confidence: float = 0.0
    avg_embedding_quality: float = 0.0
    frame_range: Tuple[int, int] = (0, 0)
    
    # Manual labeling fields
    person_label: Optional[str] = None  # Manual label (e.g., "John", "Jane", "Unknown")
    verified: bool = False  # Whether manually reviewed
    mega_cluster_id: Optional[str] = None  # For grouping into larger clusters
    notes: str = ""  # Manual notes
    
    def __post_init__(self):
        if self.embeddings:
            self.cluster_size = len(self.embeddings)
            
            # Calculate frame range
            frames = [emb.frame_number for emb in self.embeddings]
            self.frame_range = (min(frames), max(frames))
            
            # Calculate average metrics
            self.avg_confidence = np.mean([emb.confidence for emb in self.embeddings])
            self.avg_embedding_quality = np.mean([emb.embedding_quality for emb in self.embeddings])
            
            # Select best representative (highest quality * confidence)
            if self.embeddings:
                self.representative_embedding = max(
                    self.embeddings,
                    key=lambda x: x.embedding_quality * x.confidence
                )


class TrackLevelClusterer:
    """Creates individual clusters for each unique track_id."""
    
    def __init__(self, 
                 min_embedding_quality: float = 0.3,
                 min_confidence: float = 0.3):
        """
        Initialize track-level clusterer.
        
        Args:
            min_embedding_quality: Minimum embedding quality threshold
            min_confidence: Minimum detection confidence threshold
        """
        self.min_embedding_quality = min_embedding_quality
        self.min_confidence = min_confidence
        
        logger.info("TrackLevelClusterer initialized:")
        logger.info("  Min embedding quality: %.2f", min_embedding_quality)
        logger.info("  Min confidence: %.2f", min_confidence)
    
    def load_and_filter_embeddings(self, embeddings_dir: str) -> List[PersonEmbedding]:
        """Load embeddings with quality filtering."""
        logger.info("Loading embeddings from %s", embeddings_dir)
        
        all_embeddings = []
        embeddings_path = Path(embeddings_dir)
        json_files = list(embeddings_path.glob("*_embeddings.json"))
        
        logger.info("Found %d embedding files", len(json_files))
        
        for json_file in json_files:
            try:
                embeddings = PersonEmbeddingGenerator.load_embeddings(str(json_file))
                all_embeddings.extend(embeddings)
                logger.info("Loaded %d embeddings from %s", len(embeddings), json_file.name)
            except Exception as e:
                logger.warning("Failed to load %s: %s", json_file.name, e)
        
        logger.info("Total embeddings before filtering: %d", len(all_embeddings))
        
        # Apply quality filtering
        filtered_embeddings = [
            emb for emb in all_embeddings
            if (emb.embedding_quality >= self.min_embedding_quality and
                emb.confidence >= self.min_confidence and
                emb.embedding is not None and
                len(emb.embedding) > 0)
        ]
        
        logger.info("Total embeddings after filtering: %d", len(filtered_embeddings))
        logger.info("Filtered out %d low-quality embeddings", 
                   len(all_embeddings) - len(filtered_embeddings))
        
        return filtered_embeddings
    
    def create_track_level_clusters(self, embeddings: List[PersonEmbedding]) -> List[TrackLevelCluster]:
        """
        Create one cluster per unique (video, track_id) combination.
        
        Args:
            embeddings: List of PersonEmbedding objects
            
        Returns:
            List of TrackLevelCluster objects (one per track)
        """
        if not embeddings:
            logger.warning("No embeddings for clustering")
            return []
        
        logger.info("Creating track-level clusters from %d embeddings", len(embeddings))
        
        # Group embeddings by unique (video, track_id) combination
        track_groups = defaultdict(list)
        
        for emb in embeddings:
            # Create unique key for each track
            key = (emb.video_filename, emb.track_id)
            track_groups[key].append(emb)
        
        logger.info("Found %d unique tracks across videos", len(track_groups))
        
        # Create one cluster per track
        clusters = []
        for cluster_idx, ((video_filename, track_id), group_embeddings) in enumerate(track_groups.items()):
            # Generate unique cluster ID
            cluster_id = f"track_{cluster_idx:04d}_{video_filename.replace('.mp4', '')}_{track_id}"
            
            cluster = TrackLevelCluster(
                cluster_id=cluster_id,
                video_filename=video_filename,
                track_id=track_id,
                embeddings=group_embeddings
            )
            clusters.append(cluster)
        
        logger.info("Created %d track-level clusters", len(clusters))
        
        # Sort clusters by video filename and track_id for consistency
        clusters.sort(key=lambda x: (x.video_filename, x.track_id))
        
        return clusters
    
    def save_clusters(self, clusters: List[TrackLevelCluster], output_file: str):
        """Save track-level clusters to JSON file."""
        cluster_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_clusters": len(clusters),
                "clustering_type": "track_level",
                "clusterer_config": {
                    "min_embedding_quality": self.min_embedding_quality,
                    "min_confidence": self.min_confidence
                }
            },
            "clusters": []
        }
        
        for cluster in clusters:
            cluster_info = {
                "cluster_id": cluster.cluster_id,
                "video_filename": cluster.video_filename,
                "track_id": cluster.track_id,
                "cluster_size": cluster.cluster_size,
                "avg_confidence": float(cluster.avg_confidence),
                "avg_embedding_quality": float(cluster.avg_embedding_quality),
                "frame_range": list(cluster.frame_range),
                
                # Manual labeling fields
                "person_label": cluster.person_label,
                "verified": cluster.verified,
                "mega_cluster_id": cluster.mega_cluster_id,
                "notes": cluster.notes,
                
                # Representative embedding
                "representative": (
                    {
                        "track_id": cluster.representative_embedding.track_id,
                        "frame_number": cluster.representative_embedding.frame_number,
                        "video_filename": cluster.representative_embedding.video_filename,
                        "confidence": float(cluster.representative_embedding.confidence),
                        "embedding_quality": float(cluster.representative_embedding.embedding_quality),
                    }
                    if cluster.representative_embedding
                    else None
                ),
                
                # All embeddings metadata
                "embeddings": [
                    {
                        "track_id": emb.track_id,
                        "frame_number": emb.frame_number,
                        "video_filename": emb.video_filename,
                        "confidence": float(emb.confidence),
                        "embedding_quality": float(emb.embedding_quality),
                    }
                    for emb in cluster.embeddings
                ]
            }
            cluster_data["clusters"].append(cluster_info)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Track-level clusters saved to %s", output_file)
    
    def load_clusters(self, clusters_file: str) -> List[TrackLevelCluster]:
        """Load track-level clusters from JSON file."""
        logger.info("Loading track-level clusters from %s", clusters_file)
        
        with open(clusters_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        clusters = []
        for cluster_info in cluster_data['clusters']:
            # Note: embeddings are not fully restored (just metadata)
            cluster = TrackLevelCluster(
                cluster_id=cluster_info['cluster_id'],
                video_filename=cluster_info['video_filename'],
                track_id=cluster_info['track_id'],
                embeddings=[],  # Would need separate loading if needed
                cluster_size=cluster_info['cluster_size'],
                avg_confidence=cluster_info['avg_confidence'],
                avg_embedding_quality=cluster_info['avg_embedding_quality'],
                frame_range=tuple(cluster_info['frame_range']),
                person_label=cluster_info.get('person_label'),
                verified=cluster_info.get('verified', False),
                mega_cluster_id=cluster_info.get('mega_cluster_id'),
                notes=cluster_info.get('notes', '')
            )
            clusters.append(cluster)
        
        logger.info("Loaded %d track-level clusters", len(clusters))
        return clusters
    
    def create_cluster_report(self, clusters: List[TrackLevelCluster], output_dir: str):
        """Create detailed track-level clustering report."""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "track_level_cluster_report.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("TRACK-LEVEL CLUSTERING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total track-level clusters: {len(clusters)}\n\n")
            
            if clusters:
                # Summary statistics
                total_embeddings = sum(c.cluster_size for c in clusters)
                videos = set(c.video_filename for c in clusters)
                avg_confidence = np.mean([c.avg_confidence for c in clusters])
                avg_quality = np.mean([c.avg_embedding_quality for c in clusters])
                
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total embeddings: {total_embeddings}\n")
                f.write(f"Unique videos: {len(videos)}\n")
                f.write(f"Average confidence: {avg_confidence:.3f}\n")
                f.write(f"Average embedding quality: {avg_quality:.3f}\n")
                f.write(f"Single-embedding clusters: {sum(1 for c in clusters if c.cluster_size == 1)}\n")
                f.write(f"Multi-embedding clusters: {sum(1 for c in clusters if c.cluster_size > 1)}\n\n")
                
                # Video breakdown
                video_stats = defaultdict(int)
                for cluster in clusters:
                    video_stats[cluster.video_filename] += 1
                
                f.write("VIDEO BREAKDOWN\n")
                f.write("-" * 30 + "\n")
                for video, count in sorted(video_stats.items()):
                    f.write(f"{video}: {count} clusters\n")
                f.write("\n")
                
                # Manual labeling status
                labeled_clusters = [c for c in clusters if c.person_label]
                verified_clusters = [c for c in clusters if c.verified]
                mega_clustered = [c for c in clusters if c.mega_cluster_id]
                
                f.write("MANUAL LABELING STATUS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Labeled clusters: {len(labeled_clusters)} ({len(labeled_clusters)/len(clusters)*100:.1f}%)\n")
                f.write(f"Verified clusters: {len(verified_clusters)} ({len(verified_clusters)/len(clusters)*100:.1f}%)\n")
                f.write(f"Mega-clustered: {len(mega_clustered)} ({len(mega_clustered)/len(clusters)*100:.1f}%)\n\n")
                
                # Cluster details (first 20)
                f.write("CLUSTER DETAILS (First 20)\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'ID':<15} {'Video':<25} {'Track':<15} {'Size':<4} {'Conf':<6} {'Qual':<6} {'Label'}\n")
                f.write("-" * 80 + "\n")
                
                for cluster in clusters[:20]:
                    video_short = cluster.video_filename[-25:] if len(cluster.video_filename) > 25 else cluster.video_filename
                    track_short = str(cluster.track_id)[-15:] if len(str(cluster.track_id)) > 15 else str(cluster.track_id)
                    label = cluster.person_label or "Unlabeled"
                    
                    f.write(f"{cluster.cluster_id[:15]:<15} {video_short:<25} {track_short:<15} "
                           f"{cluster.cluster_size:<4} {cluster.avg_confidence:<6.3f} "
                           f"{cluster.avg_embedding_quality:<6.3f} {label}\n")
                
                if len(clusters) > 20:
                    f.write(f"\n... and {len(clusters) - 20} more clusters\n")
        
        logger.info("Track-level clustering report saved: %s", report_path)


def main():
    """Run track-level clustering."""
    
    # Paths
    embeddings_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/embeddings"
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/track_level_clustering"
    
    logger.info("=" * 80)
    logger.info("TRACK-LEVEL CLUSTERING FOR MANUAL LABELING")
    logger.info("=" * 80)
    
    if not os.path.exists(embeddings_dir):
        logger.error("Embeddings directory not found: %s", embeddings_dir)
        return 1
    
    try:
        # Create track-level clusterer
        clusterer = TrackLevelClusterer(
            min_embedding_quality=0.3,
            min_confidence=0.3
        )
        
        # Load and filter embeddings
        embeddings = clusterer.load_and_filter_embeddings(embeddings_dir)
        
        if not embeddings:
            logger.error("No valid embeddings found after filtering")
            return 1
        
        # Create track-level clusters
        clusters = clusterer.create_track_level_clusters(embeddings)
        
        if not clusters:
            logger.warning("No clusters created")
            return 1
        
        # Create output directory and save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save clusters
        clusters_file = os.path.join(output_dir, "track_level_clusters.json")
        clusterer.save_clusters(clusters, clusters_file)
        
        # Create analysis report
        clusterer.create_cluster_report(clusters, output_dir)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TRACK-LEVEL CLUSTERING RESULTS")
        logger.info("=" * 80)
        logger.info("Total track-level clusters created: %d", len(clusters))
        
        videos = set(c.video_filename for c in clusters)
        total_embeddings = sum(c.cluster_size for c in clusters)
        
        logger.info("Videos processed: %d", len(videos))
        logger.info("Total embeddings clustered: %d", total_embeddings)
        logger.info("Embeddings per cluster: %.1f", total_embeddings / len(clusters))
        
        # Show sample clusters
        logger.info("\nSample clusters:")
        for i, cluster in enumerate(clusters[:5]):
            logger.info("  %d. %s (video=%s, track=%s, size=%d)", 
                       i+1, cluster.cluster_id, cluster.video_filename[-20:], 
                       cluster.track_id, cluster.cluster_size)
        
        logger.info("\nResults saved to: %s", output_dir)
        logger.info("Next steps:")
        logger.info("  1. Review cluster grids for visual inspection")
        logger.info("  2. Use labeling tools to assign person names")
        logger.info("  3. Group clusters into mega-clusters by person")
        logger.info("  4. Export training data for refined model")
        
        return 0
        
    except Exception as e:
        logger.error("Track-level clustering failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())