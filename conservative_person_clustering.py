#!/usr/bin/env python3
"""
Conservative Person Clustering Module

This module provides ultra-conservative clustering specifically designed to group
only near-duplicate detections that are essentially the same person detection
captured in consecutive frames or very close temporal proximity.

Key Features:
- Track-based clustering: Uses existing track IDs when available
- Temporal clustering: Groups detections close in time from same video
- Minimal false positives: Prefers over-segmentation to avoid mixing people
- Simple and fast: Lightweight compared to similarity-based clustering

Usage:
    python conservative_person_clustering.py
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from logging_config import create_logger
from person_embedding import PersonEmbedding, PersonEmbeddingGenerator

logger = create_logger(__name__)


@dataclass
class ConservativePersonCluster:
    """Conservative cluster containing only near-duplicate detections."""

    cluster_id: int
    embeddings: List[PersonEmbedding] = field(default_factory=list)
    representative_embedding: Optional[PersonEmbedding] = None
    cluster_size: int = 0
    cluster_type: str = "track"  # "track" or "temporal"
    video_filename: str = ""
    track_id: Optional[int] = None
    frame_range: Tuple[int, int] = (0, 0)
    temporal_span_seconds: float = 0.0
    avg_similarity: float = 1.0  # Conservative clusters should be very similar

    def __post_init__(self):
        if self.embeddings:
            self.cluster_size = len(self.embeddings)
            frames = [emb.frame_number for emb in self.embeddings]
            self.frame_range = (min(frames), max(frames))
            
            # Calculate temporal span (assuming 30fps)
            frame_span = self.frame_range[1] - self.frame_range[0]
            self.temporal_span_seconds = frame_span / 30.0
            
            # Select best representative (highest quality * confidence)
            self.representative_embedding = max(
                self.embeddings, 
                key=lambda x: x.embedding_quality * x.confidence
            )
            
            # Set video filename from first embedding
            self.video_filename = self.embeddings[0].video_filename


class ConservativePersonClusterer:
    """
    Ultra-conservative person clustering that only groups near-duplicates.
    
    This clusterer uses two strategies:
    1. Track-based: Group embeddings with same track_id
    2. Temporal: Group embeddings very close in time (same video)
    """

    def __init__(
        self,
        max_temporal_gap_seconds: float = 2.0,
        min_cluster_size: int = 2,
        quality_threshold: float = 0.3,
    ):
        """
        Initialize conservative clustering system.

        Args:
            max_temporal_gap_seconds: Maximum time gap between detections in temporal clusters
            min_cluster_size: Minimum embeddings per cluster
            quality_threshold: Minimum embedding quality to include
        """
        self.max_temporal_gap_seconds = max_temporal_gap_seconds
        self.min_cluster_size = min_cluster_size
        self.quality_threshold = quality_threshold

        logger.info("ConservativePersonClusterer initialized:")
        logger.info("  Max temporal gap: %.1f seconds", max_temporal_gap_seconds)
        logger.info("  Min cluster size: %d", min_cluster_size)
        logger.info("  Quality threshold: %.2f", quality_threshold)

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
            if (emb.embedding_quality >= self.quality_threshold and
                emb.confidence >= 0.3 and
                emb.embedding is not None and
                len(emb.embedding) > 0)
        ]

        logger.info("Total embeddings after filtering: %d", len(filtered_embeddings))
        logger.info("Filtered out %d low-quality embeddings", 
                   len(all_embeddings) - len(filtered_embeddings))

        return filtered_embeddings

    def cluster_embeddings_conservative(
        self, embeddings: List[PersonEmbedding]
    ) -> List[ConservativePersonCluster]:
        """
        Perform ultra-conservative clustering of person embeddings.
        
        This creates one cluster per unique track_id to ensure no tracks are mixed.

        Args:
            embeddings: List of PersonEmbedding objects to cluster

        Returns:
            List of ConservativePersonCluster objects (one per track)
        """
        if len(embeddings) < 1:
            logger.warning("No embeddings for clustering")
            return []

        logger.info("Conservative clustering of %d embeddings...", len(embeddings))
        logger.info("Creating one cluster per track_id to avoid mixing tracks")

        # Group embeddings by unique (video, track_id) combination
        track_groups = defaultdict(list)
        
        for emb in embeddings:
            # Create unique key for each track
            key = (emb.video_filename, emb.track_id)
            track_groups[key].append(emb)
        
        logger.info("Found %d unique tracks across videos", len(track_groups))
        
        # Create one cluster per track
        all_clusters = []
        for (video_filename, track_id), group_embeddings in track_groups.items():
            cluster = ConservativePersonCluster(
                cluster_id=-1,  # Will be assigned later
                embeddings=group_embeddings,
                cluster_type="track",
                track_id=track_id
            )
            all_clusters.append(cluster)
        
        # Filter by minimum size
        valid_clusters = [
            cluster for cluster in all_clusters 
            if cluster.cluster_size >= self.min_cluster_size
        ]

        # Assign cluster IDs
        for i, cluster in enumerate(valid_clusters):
            cluster.cluster_id = i

        logger.info("Conservative clustering completed:")
        logger.info("  Total unique tracks: %d", len(track_groups))
        logger.info("  Valid clusters (size >= %d): %d", 
                   self.min_cluster_size, len(valid_clusters))
        logger.info("  Filtered out %d small tracks (< %d embeddings)", 
                   len(track_groups) - len(valid_clusters), self.min_cluster_size)

        return valid_clusters

    def _cluster_by_tracks(self, embeddings: List[PersonEmbedding]) -> List[ConservativePersonCluster]:
        """Group embeddings by video and track_id."""
        track_groups = defaultdict(list)
        
        for emb in embeddings:
            if emb.track_id is not None:
                # Group by (video, track_id) combination
                key = (emb.video_filename, emb.track_id)
                track_groups[key].append(emb)
        
        clusters = []
        for (video_filename, track_id), group_embeddings in track_groups.items():
            if len(group_embeddings) >= self.min_cluster_size:
                cluster = ConservativePersonCluster(
                    cluster_id=-1,  # Will be assigned later
                    embeddings=group_embeddings,
                    cluster_type="track",
                    track_id=track_id
                )
                clusters.append(cluster)
        
        logger.info("Created %d track-based clusters", len(clusters))
        return clusters

    def _cluster_by_temporal_proximity(self, embeddings: List[PersonEmbedding]) -> List[ConservativePersonCluster]:
        """Group embeddings by temporal proximity within the same video."""
        # Group by video first
        video_groups = defaultdict(list)
        for emb in embeddings:
            video_groups[emb.video_filename].append(emb)
        
        clusters = []
        for video_filename, video_embeddings in video_groups.items():
            # Sort by frame number
            video_embeddings.sort(key=lambda x: x.frame_number)
            
            # Group consecutive embeddings within temporal threshold
            current_group = []
            max_frame_gap = self.max_temporal_gap_seconds * 30  # Assuming 30fps
            
            for emb in video_embeddings:
                if (not current_group or 
                    emb.frame_number - current_group[-1].frame_number <= max_frame_gap):
                    current_group.append(emb)
                else:
                    # Finalize current group if large enough
                    if len(current_group) >= self.min_cluster_size:
                        cluster = ConservativePersonCluster(
                            cluster_id=-1,  # Will be assigned later
                            embeddings=current_group.copy(),
                            cluster_type="temporal"
                        )
                        clusters.append(cluster)
                    
                    # Start new group
                    current_group = [emb]
            
            # Don't forget the last group
            if len(current_group) >= self.min_cluster_size:
                cluster = ConservativePersonCluster(
                    cluster_id=-1,  # Will be assigned later
                    embeddings=current_group.copy(),
                    cluster_type="temporal"
                )
                clusters.append(cluster)
        
        logger.info("Created %d temporal clusters", len(clusters))
        return clusters

    def create_conservative_report(self, clusters: List[ConservativePersonCluster], output_dir: str):
        """Create detailed conservative clustering report."""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "conservative_cluster_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("CONSERVATIVE PERSON CLUSTERING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total clusters: {len(clusters)}\n\n")

            # Summary statistics
            if clusters:
                track_clusters = [c for c in clusters if c.cluster_type == "track"]
                temporal_clusters = [c for c in clusters if c.cluster_type == "temporal"]
                total_embeddings = sum(c.cluster_size for c in clusters)

                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total embeddings clustered: {total_embeddings}\n")
                f.write(f"Track-based clusters: {len(track_clusters)}\n")
                f.write(f"Temporal clusters: {len(temporal_clusters)}\n")
                f.write(f"Average cluster size: {total_embeddings / len(clusters):.1f}\n")
                f.write(f"Largest cluster: {max(c.cluster_size for c in clusters)}\n\n")

            # Cluster details
            f.write("CLUSTER DETAILS\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'ID':<4} {'Type':<8} {'Size':<6} {'Video':<30} {'Track':<8} {'Frames':<15} {'Duration'}\n")
            f.write("-" * 80 + "\n")

            for cluster in sorted(clusters, key=lambda x: x.cluster_size, reverse=True):
                track_str = str(cluster.track_id) if cluster.track_id is not None else "N/A"
                frame_str = f"{cluster.frame_range[0]}-{cluster.frame_range[1]}"
                duration_str = f"{cluster.temporal_span_seconds:.1f}s"
                video_short = cluster.video_filename[-30:] if len(cluster.video_filename) > 30 else cluster.video_filename
                
                f.write(f"{cluster.cluster_id:<4} {cluster.cluster_type:<8} {cluster.cluster_size:<6} "
                       f"{video_short:<30} {track_str:<8} {frame_str:<15} {duration_str}\n")

        logger.info("Conservative clustering report saved: %s", report_path)


def save_conservative_clusters(clusters: List[ConservativePersonCluster], output_file: str):
    """Save conservative clustering results."""
    
    cluster_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_clusters": len(clusters),
            "clustering_type": "conservative",
        },
        "clusters": [],
    }

    for cluster in clusters:
        cluster_info = {
            "cluster_id": cluster.cluster_id,
            "cluster_size": cluster.cluster_size,
            "cluster_type": cluster.cluster_type,
            "video_filename": cluster.video_filename,
            "track_id": cluster.track_id,
            "frame_range": list(cluster.frame_range),
            "temporal_span_seconds": cluster.temporal_span_seconds,
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
            "embeddings": [
                {
                    "track_id": emb.track_id,
                    "frame_number": emb.frame_number,
                    "video_filename": emb.video_filename,
                    "confidence": float(emb.confidence),
                    "embedding_quality": float(emb.embedding_quality),
                }
                for emb in cluster.embeddings
            ],
        }
        cluster_data["clusters"].append(cluster_info)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=2)

    logger.info("Conservative clusters saved to %s", output_file)


def main():
    """Run conservative person clustering."""

    # Paths
    embeddings_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/embeddings"
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/conservative_clustering"

    logger.info("=" * 60)
    logger.info("CONSERVATIVE PERSON CLUSTERING")
    logger.info("=" * 60)

    if not os.path.exists(embeddings_dir):
        logger.error("Embeddings directory not found: %s", embeddings_dir)
        return 1

    try:
        # Create conservative clusterer
        clusterer = ConservativePersonClusterer(
            max_temporal_gap_seconds=2.0,  # Not used in track-only mode
            min_cluster_size=1,  # Keep all tracks, even single detections
            quality_threshold=0.3,
        )

        # Load and filter embeddings
        embeddings = clusterer.load_and_filter_embeddings(embeddings_dir)

        if not embeddings:
            logger.error("No valid embeddings found after filtering")
            return 1

        # Perform conservative clustering
        clusters = clusterer.cluster_embeddings_conservative(embeddings)

        if not clusters:
            logger.warning("No clusters found")
            return 1

        # Create analysis
        os.makedirs(output_dir, exist_ok=True)
        clusterer.create_conservative_report(clusters, output_dir)

        # Save cluster data
        conservative_clusters_file = os.path.join(output_dir, "conservative_clusters.json")
        save_conservative_clusters(clusters, conservative_clusters_file)

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("CONSERVATIVE CLUSTERING RESULTS")
        logger.info("=" * 60)
        logger.info("Ultra-conservative clusters discovered: %d", len(clusters))

        track_clusters = [c for c in clusters if c.cluster_type == "track"]
        temporal_clusters = [c for c in clusters if c.cluster_type == "temporal"]
        
        logger.info("Track-based clusters: %d", len(track_clusters))
        logger.info("Temporal clusters: %d", len(temporal_clusters))

        for i, cluster in enumerate(clusters[:5]):
            logger.info("Cluster %d: size=%d, type=%s, video=%s", 
                       i+1, cluster.cluster_size, cluster.cluster_type, 
                       cluster.video_filename[-20:])

        logger.info("\nResults saved to: %s", output_dir)
        return 0

    except Exception as e:
        logger.error("Conservative clustering failed: %s", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())