#!/usr/bin/env python3
"""
Track Data Structure Analyzer

This script analyzes the current track data structure to understand:
1. How many unique track_ids exist in the embeddings
2. The structure of track data for clustering
3. Distribution of embeddings per track
4. Current clustering approach analysis

This analysis will inform the creation of track-level clusters for manual labeling.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
import numpy as np

from person_embedding import PersonEmbedding, PersonEmbeddingGenerator
from conservative_person_clustering import ConservativePersonClusterer
from logging_config import create_logger

logger = create_logger(__name__)


class TrackDataAnalyzer:
    """Analyzer for track-based data structure and clustering preparation."""
    
    def __init__(self, embeddings_dir: str):
        """Initialize analyzer with embeddings directory."""
        self.embeddings_dir = embeddings_dir
        self.embeddings: List[PersonEmbedding] = []
        self.track_stats: Dict[str, Dict] = {}
        
    def load_all_embeddings(self) -> List[PersonEmbedding]:
        """Load all embeddings from the embeddings directory."""
        logger.info(f"Loading embeddings from {self.embeddings_dir}")
        
        embeddings_path = Path(self.embeddings_dir)
        if not embeddings_path.exists():
            logger.error(f"Embeddings directory not found: {self.embeddings_dir}")
            return []
            
        json_files = list(embeddings_path.glob("*_embeddings.json"))
        logger.info(f"Found {len(json_files)} embedding files")
        
        all_embeddings = []
        for json_file in json_files:
            try:
                embeddings = PersonEmbeddingGenerator.load_embeddings(str(json_file))
                all_embeddings.extend(embeddings)
                logger.info(f"Loaded {len(embeddings)} embeddings from {json_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {json_file.name}: {e}")
        
        self.embeddings = all_embeddings
        logger.info(f"Total embeddings loaded: {len(all_embeddings)}")
        return all_embeddings
    
    def analyze_track_structure(self) -> Dict[str, any]:
        """Analyze the track structure and return comprehensive statistics."""
        if not self.embeddings:
            logger.warning("No embeddings loaded for analysis")
            return {}
        
        # Group by (video, track_id) combinations
        track_groups = defaultdict(list)
        video_track_counts = defaultdict(set)
        
        for emb in self.embeddings:
            track_key = (emb.video_filename, emb.track_id)
            track_groups[track_key].append(emb)
            video_track_counts[emb.video_filename].add(emb.track_id)
        
        # Analyze track statistics
        track_sizes = [len(embeddings) for embeddings in track_groups.values()]
        track_size_distribution = Counter(track_sizes)
        
        # Video-level statistics
        video_stats = {}
        for video, track_ids in video_track_counts.items():
            video_embeddings = [emb for emb in self.embeddings if emb.video_filename == video]
            video_stats[video] = {
                'unique_tracks': len(track_ids),
                'total_embeddings': len(video_embeddings),
                'avg_embeddings_per_track': len(video_embeddings) / len(track_ids) if track_ids else 0
            }
        
        # Overall statistics
        stats = {
            'total_embeddings': len(self.embeddings),
            'total_unique_tracks': len(track_groups),
            'unique_videos': len(video_track_counts),
            'avg_embeddings_per_track': np.mean(track_sizes) if track_sizes else 0,
            'median_embeddings_per_track': np.median(track_sizes) if track_sizes else 0,
            'min_embeddings_per_track': min(track_sizes) if track_sizes else 0,
            'max_embeddings_per_track': max(track_sizes) if track_sizes else 0,
            'track_size_distribution': dict(track_size_distribution),
            'video_stats': video_stats,
            'track_groups': track_groups  # For further analysis
        }
        
        self.track_stats = stats
        return stats
    
    def print_analysis_report(self):
        """Print a comprehensive analysis report."""
        if not self.track_stats:
            logger.error("No track statistics available. Run analyze_track_structure() first.")
            return
        
        stats = self.track_stats
        
        print("=" * 80)
        print("TRACK DATA STRUCTURE ANALYSIS REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().isoformat()}")
        print()
        
        print("OVERALL STATISTICS")
        print("-" * 40)
        print(f"Total embeddings: {stats['total_embeddings']:,}")
        print(f"Total unique tracks: {stats['total_unique_tracks']:,}")
        print(f"Unique videos: {stats['unique_videos']}")
        print(f"Average embeddings per track: {stats['avg_embeddings_per_track']:.1f}")
        print(f"Median embeddings per track: {stats['median_embeddings_per_track']:.1f}")
        print(f"Min embeddings per track: {stats['min_embeddings_per_track']}")
        print(f"Max embeddings per track: {stats['max_embeddings_per_track']}")
        print()
        
        print("TRACK SIZE DISTRIBUTION")
        print("-" * 40)
        for size, count in sorted(stats['track_size_distribution'].items()):
            percentage = (count / stats['total_unique_tracks']) * 100
            print(f"Tracks with {size:2d} embeddings: {count:4d} ({percentage:5.1f}%)")
        print()
        
        print("VIDEO-LEVEL STATISTICS")
        print("-" * 40)
        print(f"{'Video':<50} {'Tracks':<8} {'Embeddings':<12} {'Avg/Track'}")
        print("-" * 80)
        for video, vstats in stats['video_stats'].items():
            video_short = video[-50:] if len(video) > 50 else video
            print(f"{video_short:<50} {vstats['unique_tracks']:<8} {vstats['total_embeddings']:<12} {vstats['avg_embeddings_per_track']:.1f}")
        print()
        
        print("TRACK-LEVEL CLUSTERING IMPLICATIONS")
        print("-" * 40)
        print(f"If creating one cluster per track:")
        print(f"  • Total clusters: {stats['total_unique_tracks']:,}")
        print(f"  • Clusters for manual labeling: {stats['total_unique_tracks']:,}")
        print(f"  • Single-embedding tracks: {stats['track_size_distribution'].get(1, 0):,}")
        print(f"  • Multi-embedding tracks: {stats['total_unique_tracks'] - stats['track_size_distribution'].get(1, 0):,}")
        print()
        
        # Analyze for different minimum cluster sizes
        print("CLUSTERING WITH MINIMUM SIZE FILTERS")
        print("-" * 40)
        for min_size in [1, 2, 3, 5, 10]:
            qualifying_tracks = sum(
                count for size, count in stats['track_size_distribution'].items() 
                if size >= min_size
            )
            percentage = (qualifying_tracks / stats['total_unique_tracks']) * 100
            print(f"Min {min_size:2d} embeddings: {qualifying_tracks:4d} clusters ({percentage:5.1f}%)")
        print()
    
    def compare_with_conservative_clustering(self):
        """Compare track-level clustering with conservative clustering approach."""
        if not self.embeddings:
            logger.warning("No embeddings loaded for comparison")
            return
        
        print("CONSERVATIVE CLUSTERING COMPARISON")
        print("-" * 40)
        
        # Run conservative clusterer
        clusterer = ConservativePersonClusterer(
            max_temporal_gap_seconds=2.0,
            min_cluster_size=1,  # Include all tracks
            quality_threshold=0.3
        )
        
        # Filter embeddings like the conservative clusterer does
        filtered_embeddings = [
            emb for emb in self.embeddings
            if (emb.embedding_quality >= 0.3 and
                emb.confidence >= 0.3 and
                emb.embedding is not None and
                len(emb.embedding) > 0)
        ]
        
        # Create conservative clusters
        conservative_clusters = clusterer.cluster_embeddings_conservative(filtered_embeddings)
        
        print(f"Original embeddings: {len(self.embeddings)}")
        print(f"Filtered embeddings: {len(filtered_embeddings)}")
        print(f"Conservative clusters: {len(conservative_clusters)}")
        print(f"Clusters match unique tracks: {len(conservative_clusters) == self.track_stats['total_unique_tracks']}")
        print()
        
        return conservative_clusters
    
    def save_track_analysis(self, output_file: str):
        """Save track analysis results to JSON file."""
        if not self.track_stats:
            logger.error("No track statistics to save")
            return
        
        # Create serializable version (remove track_groups which contains objects)
        save_stats = {k: v for k, v in self.track_stats.items() if k != 'track_groups'}
        save_stats['analysis_timestamp'] = datetime.now().isoformat()
        save_stats['embeddings_directory'] = self.embeddings_dir
        
        with open(output_file, 'w') as f:
            json.dump(save_stats, f, indent=2)
        
        logger.info(f"Track analysis saved to: {output_file}")


def main():
    """Run track data analysis."""
    
    # Configuration
    embeddings_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/embeddings"
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/track_analysis"
    
    logger.info("=" * 80)
    logger.info("TRACK DATA STRUCTURE ANALYZER")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = TrackDataAnalyzer(embeddings_dir)
        
        # Load and analyze embeddings
        embeddings = analyzer.load_all_embeddings()
        if not embeddings:
            logger.error("No embeddings found")
            return 1
        
        # Analyze track structure
        stats = analyzer.analyze_track_structure()
        
        # Print comprehensive report
        analyzer.print_analysis_report()
        
        # Compare with conservative clustering
        analyzer.compare_with_conservative_clustering()
        
        # Save results
        analysis_file = os.path.join(output_dir, "track_analysis.json")
        analyzer.save_track_analysis(analysis_file)
        
        print(f"\nAnalysis complete. Results saved to: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Track analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())