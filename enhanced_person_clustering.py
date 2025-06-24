#!/usr/bin/env python3
"""
Enhanced Person Clustering Module

This module provides improved clustering with better precision for person identification.
Key improvements over basic clustering:
1. Higher similarity thresholds for better precision
2. Multi-pass clustering with quality filtering
3. Better handling of poor-quality embeddings
4. Face-based filtering for more reliable clustering
5. Temporal consistency checks

Usage:
    python enhanced_person_clustering.py
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from logging_config import create_logger
from person_embedding import PersonEmbedding, PersonEmbeddingGenerator

logger = create_logger(__name__)


@dataclass
class EnhancedPersonCluster:
    """Enhanced cluster with quality metrics."""

    cluster_id: int
    embeddings: List[PersonEmbedding] = field(default_factory=list)
    representative_embedding: Optional[PersonEmbedding] = None
    cluster_size: int = 0
    avg_similarity: float = 0.0
    min_similarity: float = 1.0
    quality_score: float = 0.0
    videos_spanned: List[str] = field(default_factory=list)
    frame_range: Tuple[int, int] = (0, 0)
    temporal_consistency: float = 0.0

    def __post_init__(self):
        if self.embeddings:
            self.cluster_size = len(self.embeddings)
            self.videos_spanned = list(
                set(emb.video_filename for emb in self.embeddings)
            )
            frames = [emb.frame_number for emb in self.embeddings]
            self.frame_range = (min(frames), max(frames))

            # Calculate quality metrics
            self._calculate_quality_metrics()

            # Select best representative
            self.representative_embedding = max(
                self.embeddings, key=lambda x: x.embedding_quality * x.confidence
            )

    def _calculate_quality_metrics(self):
        """Calculate cluster quality metrics."""
        if len(self.embeddings) < 2:
            self.avg_similarity = 1.0
            self.min_similarity = 1.0
            self.quality_score = (
                self.embeddings[0].embedding_quality if self.embeddings else 0.0
            )
            return

        # Calculate pairwise similarities within cluster
        embeddings_matrix = np.array([emb.embedding for emb in self.embeddings])
        similarity_matrix = cosine_similarity(embeddings_matrix)

        # Remove diagonal and get upper triangle
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix[mask]

        self.avg_similarity = similarities.mean()
        self.min_similarity = similarities.min()

        # Quality score combines avg similarity, min similarity, and embedding quality
        avg_embedding_quality = np.mean(
            [emb.embedding_quality for emb in self.embeddings]
        )
        avg_confidence = np.mean([emb.confidence for emb in self.embeddings])

        self.quality_score = (
            0.4 * self.avg_similarity
            + 0.3 * self.min_similarity
            + 0.2 * avg_embedding_quality
            + 0.1 * avg_confidence
        )


class EnhancedPersonClusterer:
    """
    Enhanced clustering system with improved precision and quality filtering.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,  # Conservative but not ultra-conservative
        min_cluster_size: int = 2,
        quality_threshold: float = 0.5,  # Balanced quality threshold
        use_dbscan: bool = False,  # Use hierarchical for more control
        eps: float = 0.12,  # DBSCAN parameter (1 - similarity_threshold)
        min_samples: int = 2,
    ):
        """
        Initialize enhanced clustering system.

        Args:
            similarity_threshold: Higher threshold for better precision
            min_cluster_size: Minimum embeddings per cluster
            quality_threshold: Minimum embedding quality to include
            use_dbscan: Whether to use DBSCAN instead of hierarchical clustering
            eps: DBSCAN epsilon parameter (distance threshold)
            min_samples: DBSCAN minimum samples per cluster
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.quality_threshold = quality_threshold
        self.use_dbscan = use_dbscan
        self.eps = eps
        self.min_samples = min_samples

        logger.info(f"EnhancedPersonClusterer initialized:")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Quality threshold: {quality_threshold}")
        logger.info(f"  Algorithm: {'DBSCAN' if use_dbscan else 'Hierarchical'}")
        logger.info(f"  Min cluster size: {min_cluster_size}")
        logger.info(f"  Min samples: {min_samples}")

    def load_and_filter_embeddings(self, embeddings_dir: str) -> List[PersonEmbedding]:
        """
        Load embeddings with enhanced quality filtering.

        Args:
            embeddings_dir: Directory containing embedding JSON files

        Returns:
            List of high-quality PersonEmbedding objects
        """
        logger.info(f"Loading and filtering embeddings from {embeddings_dir}")

        all_embeddings = []

        # Find all JSON files
        embeddings_path = Path(embeddings_dir)
        json_files = list(embeddings_path.glob("*_embeddings.json"))

        logger.info(f"Found {len(json_files)} embedding files")

        for json_file in json_files:
            try:
                embeddings = PersonEmbeddingGenerator.load_embeddings(str(json_file))
                all_embeddings.extend(embeddings)
                logger.info(
                    f"  Loaded {len(embeddings)} embeddings from {json_file.name}"
                )
            except Exception as e:
                logger.warning(f"  Failed to load {json_file.name}: {e}")

        logger.info(f"Total embeddings before filtering: {len(all_embeddings)}")

        # Apply quality filtering
        filtered_embeddings = self._filter_low_quality_embeddings(all_embeddings)

        logger.info(f"Total embeddings after filtering: {len(filtered_embeddings)}")
        logger.info(
            f"Filtered out {len(all_embeddings) - len(filtered_embeddings)} low-quality embeddings"
        )

        return filtered_embeddings

    def _filter_low_quality_embeddings(
        self, embeddings: List[PersonEmbedding]
    ) -> List[PersonEmbedding]:
        """Filter out low-quality embeddings that can hurt clustering."""

        filtered = []

        for emb in embeddings:
            # Quality filters
            quality_ok = emb.embedding_quality >= self.quality_threshold
            confidence_ok = emb.confidence >= 0.3  # Minimum confidence
            embedding_ok = emb.embedding is not None and len(emb.embedding) > 0

            # Embedding variance check (avoid degenerate embeddings)
            variance_ok = True
            if emb.embedding is not None:
                embedding_std = np.std(emb.embedding)
                variance_ok = embedding_std > 0.01  # Minimum variance

            if quality_ok and confidence_ok and embedding_ok and variance_ok:
                filtered.append(emb)
            else:
                logger.debug(
                    f"Filtered out embedding: quality={emb.embedding_quality:.3f}, "
                    f"confidence={emb.confidence:.3f}, std={embedding_std:.4f}"
                )

        return filtered

    def cluster_embeddings_enhanced(
        self, embeddings: List[PersonEmbedding]
    ) -> List[EnhancedPersonCluster]:
        """
        Perform enhanced clustering with improved precision.

        Args:
            embeddings: List of PersonEmbedding objects to cluster

        Returns:
            List of EnhancedPersonCluster objects
        """
        if len(embeddings) < 2:
            logger.warning("Not enough embeddings for clustering")
            return []

        logger.info(f"Enhanced clustering of {len(embeddings)} embeddings...")

        # Create embedding matrix
        embedding_matrix = np.array([emb.embedding for emb in embeddings])

        # Normalize embeddings (important for cosine similarity)
        embedding_matrix = embedding_matrix / np.linalg.norm(
            embedding_matrix, axis=1, keepdims=True
        )

        if self.use_dbscan:
            clusters = self._cluster_with_dbscan(embedding_matrix, embeddings)
        else:
            clusters = self._cluster_hierarchical(embedding_matrix, embeddings)

        # Filter clusters by minimum size and quality
        valid_clusters = []
        for cluster in clusters:
            if (
                cluster.cluster_size >= self.min_cluster_size
                and cluster.quality_score > 0.3
            ):  # Minimum quality threshold
                valid_clusters.append(cluster)

        # Sort by quality score (best first)
        valid_clusters.sort(key=lambda x: x.quality_score, reverse=True)

        logger.info(f"Enhanced clustering completed:")
        logger.info(f"  Valid clusters: {len(valid_clusters)}")
        logger.info(
            f"  Average cluster quality: {np.mean([c.quality_score for c in valid_clusters]):.3f}"
        )

        return valid_clusters

    def _cluster_with_dbscan(
        self, embedding_matrix: np.ndarray, embeddings: List[PersonEmbedding]
    ) -> List[EnhancedPersonCluster]:
        """Cluster using DBSCAN algorithm."""
        logger.info(
            f"Using DBSCAN clustering (eps={self.eps}, min_samples={self.min_samples})"
        )

        # Use DBSCAN with cosine distance
        # Convert similarity to distance, ensuring no negative values
        similarity_matrix = cosine_similarity(embedding_matrix)

        # Clamp similarities to [0, 1] to avoid floating point precision issues
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix

        # Ensure distance matrix is symmetric and has zero diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # Ensure non-negative (should already be, but safety check)
        distance_matrix = np.abs(distance_matrix)

        clusterer = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="precomputed"
        )

        cluster_labels = clusterer.fit_predict(distance_matrix)

        return self._create_clusters_from_labels(cluster_labels, embeddings)

    def _cluster_hierarchical(
        self, embedding_matrix: np.ndarray, embeddings: List[PersonEmbedding]
    ) -> List[EnhancedPersonCluster]:
        """Cluster using hierarchical clustering."""
        logger.info(
            f"Using hierarchical clustering (threshold={self.similarity_threshold})"
        )

        # Compute distance matrix with safety checks
        similarity_matrix = cosine_similarity(embedding_matrix)

        # Clamp similarities to [0, 1] to avoid floating point precision issues
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix

        # Ensure distance matrix is symmetric and has zero diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # More conservative distance threshold
        distance_threshold = 1 - self.similarity_threshold

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="complete",  # Use complete linkage for more conservative clustering
        )

        cluster_labels = clusterer.fit_predict(distance_matrix)

        return self._create_clusters_from_labels(cluster_labels, embeddings)

    def _create_clusters_from_labels(
        self, cluster_labels: np.ndarray, embeddings: List[PersonEmbedding]
    ) -> List[EnhancedPersonCluster]:
        """Create cluster objects from clustering labels."""

        # Group embeddings by cluster label
        cluster_groups = defaultdict(list)
        for emb, label in zip(embeddings, cluster_labels):
            if label != -1:  # -1 indicates noise in DBSCAN
                cluster_groups[label].append(emb)

        # Create EnhancedPersonCluster objects
        clusters = []
        for cluster_id, cluster_embeddings in cluster_groups.items():
            cluster = EnhancedPersonCluster(
                cluster_id=cluster_id, embeddings=cluster_embeddings
            )
            clusters.append(cluster)

        return clusters

    def analyze_cluster_quality(
        self, clusters: List[EnhancedPersonCluster], output_dir: str
    ):
        """Create detailed quality analysis of clusters."""
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Creating enhanced cluster analysis in {output_dir}")

        # 1. Quality score distribution
        self._plot_quality_distribution(clusters, output_dir)

        # 2. Similarity analysis
        self._plot_similarity_analysis(clusters, output_dir)

        # 3. Create detailed report
        self._create_enhanced_report(clusters, output_dir)

    def _plot_quality_distribution(
        self, clusters: List[EnhancedPersonCluster], output_dir: str
    ):
        """Plot cluster quality score distribution."""
        quality_scores = [cluster.quality_score for cluster in clusters]

        plt.figure(figsize=(12, 8))

        # Quality score histogram
        plt.subplot(2, 2, 1)
        plt.hist(quality_scores, bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Cluster Quality Score")
        plt.ylabel("Number of Clusters")
        plt.title("Cluster Quality Distribution")
        plt.grid(True, alpha=0.3)

        # Quality vs Size scatter
        sizes = [cluster.cluster_size for cluster in clusters]
        plt.subplot(2, 2, 2)
        plt.scatter(sizes, quality_scores, alpha=0.6)
        plt.xlabel("Cluster Size")
        plt.ylabel("Quality Score")
        plt.title("Quality vs Size")
        plt.grid(True, alpha=0.3)

        # Min similarity distribution
        min_similarities = [
            cluster.min_similarity for cluster in clusters if cluster.cluster_size > 1
        ]
        plt.subplot(2, 2, 3)
        plt.hist(min_similarities, bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Minimum Intra-Cluster Similarity")
        plt.ylabel("Number of Clusters")
        plt.title("Minimum Similarity Distribution")
        plt.axvline(
            x=self.similarity_threshold,
            color="red",
            linestyle="--",
            label=f"Threshold: {self.similarity_threshold}",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Avg similarity distribution
        avg_similarities = [
            cluster.avg_similarity for cluster in clusters if cluster.cluster_size > 1
        ]
        plt.subplot(2, 2, 4)
        plt.hist(avg_similarities, bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Average Intra-Cluster Similarity")
        plt.ylabel("Number of Clusters")
        plt.title("Average Similarity Distribution")
        plt.axvline(
            x=self.similarity_threshold,
            color="red",
            linestyle="--",
            label=f"Threshold: {self.similarity_threshold}",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "enhanced_cluster_quality.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_similarity_analysis(
        self, clusters: List[EnhancedPersonCluster], output_dir: str
    ):
        """Create detailed similarity analysis plots."""

        # Collect similarity data
        all_avg_sims = []
        all_min_sims = []
        cluster_sizes = []

        for cluster in clusters:
            if cluster.cluster_size > 1:
                all_avg_sims.append(cluster.avg_similarity)
                all_min_sims.append(cluster.min_similarity)
                cluster_sizes.append(cluster.cluster_size)

        if not all_avg_sims:
            return

        plt.figure(figsize=(15, 5))

        # Average vs minimum similarity
        plt.subplot(1, 3, 1)
        plt.scatter(all_min_sims, all_avg_sims, alpha=0.6, s=50)
        plt.xlabel("Minimum Similarity")
        plt.ylabel("Average Similarity")
        plt.title("Min vs Avg Similarity")
        plt.plot([0, 1], [0, 1], "r--", alpha=0.5)
        plt.grid(True, alpha=0.3)

        # Similarity vs cluster size
        plt.subplot(1, 3, 2)
        plt.scatter(
            cluster_sizes, all_avg_sims, alpha=0.6, s=50, label="Avg Similarity"
        )
        plt.scatter(
            cluster_sizes, all_min_sims, alpha=0.6, s=50, label="Min Similarity"
        )
        plt.xlabel("Cluster Size")
        plt.ylabel("Similarity")
        plt.title("Similarity vs Cluster Size")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Similarity difference distribution
        sim_diffs = np.array(all_avg_sims) - np.array(all_min_sims)
        plt.subplot(1, 3, 3)
        plt.hist(sim_diffs, bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Avg - Min Similarity")
        plt.ylabel("Number of Clusters")
        plt.title("Similarity Spread Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "similarity_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_enhanced_report(
        self, clusters: List[EnhancedPersonCluster], output_dir: str
    ):
        """Create detailed enhanced clustering report."""
        report_path = os.path.join(output_dir, "enhanced_cluster_report.txt")

        with open(report_path, "w") as f:
            f.write("ENHANCED PERSON CLUSTERING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Algorithm: {'DBSCAN' if self.use_dbscan else 'Hierarchical'}\n")
            f.write(f"Similarity threshold: {self.similarity_threshold}\n")
            f.write(f"Quality threshold: {self.quality_threshold}\n")
            f.write(f"Total clusters: {len(clusters)}\n\n")

            # Summary statistics
            if clusters:
                total_embeddings = sum(c.cluster_size for c in clusters)
                avg_quality = np.mean([c.quality_score for c in clusters])
                avg_similarity = np.mean(
                    [c.avg_similarity for c in clusters if c.cluster_size > 1]
                )

                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total embeddings clustered: {total_embeddings}\n")
                f.write(
                    f"Average cluster size: {total_embeddings / len(clusters):.1f}\n"
                )
                f.write(f"Average quality score: {avg_quality:.3f}\n")
                f.write(f"Average intra-cluster similarity: {avg_similarity:.3f}\n")
                f.write(f"Largest cluster: {max(c.cluster_size for c in clusters)}\n")
                f.write(
                    f"Highest quality cluster: {max(c.quality_score for c in clusters):.3f}\n\n"
                )

            # ALL clusters by quality
            sorted_clusters = sorted(
                clusters, key=lambda x: x.quality_score, reverse=True
            )
            f.write("ALL CLUSTERS BY QUALITY\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Rank':<6} {'ID':<4} {'Size':<6} {'Quality':<8} {'Avg Sim':<8} {'Min Sim':<8} {'Videos':<7} {'Representative'}\n")
            f.write("-" * 90 + "\n")
            
            for i, cluster in enumerate(sorted_clusters):
                rep_info = ""
                if cluster.representative_embedding:
                    rep = cluster.representative_embedding
                    rep_info = f"{rep.video_filename} frame {rep.frame_number}"
                
                f.write(f"{i+1:<6} {cluster.cluster_id:<4} {cluster.cluster_size:<6} "
                       f"{cluster.quality_score:<8.3f} {cluster.avg_similarity:<8.3f} "
                       f"{cluster.min_similarity:<8.3f} {len(cluster.videos_spanned):<7} {rep_info}\n")
            
            f.write("\n")
            
            # Detailed breakdown for top 10
            f.write("DETAILED BREAKDOWN - TOP 10 CLUSTERS\n")
            f.write("-" * 50 + "\n")
            for i, cluster in enumerate(sorted_clusters[:10]):
                f.write(f"Rank {i+1}: Cluster {cluster.cluster_id}\n")
                f.write(f"  Size: {cluster.cluster_size} embeddings\n")
                f.write(f"  Quality score: {cluster.quality_score:.3f}\n")
                f.write(f"  Avg similarity: {cluster.avg_similarity:.3f}\n")
                f.write(f"  Min similarity: {cluster.min_similarity:.3f}\n")
                f.write(f"  Videos spanned: {len(cluster.videos_spanned)}\n")
                if cluster.representative_embedding:
                    rep = cluster.representative_embedding
                    f.write(
                        f"  Representative: {rep.video_filename} frame {rep.frame_number}\n"
                    )
                f.write("\n")


def main():
    """Run enhanced person clustering."""

    # Paths
    embeddings_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/embeddings"
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/enhanced_clustering"

    logger.info("=" * 60)
    logger.info("ENHANCED PERSON CLUSTERING")
    logger.info("=" * 60)

    if not os.path.exists(embeddings_dir):
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return 1

    try:
        # Create enhanced clusterer with balanced conservative settings
        clusterer = EnhancedPersonClusterer(
            similarity_threshold=0.88,  # Conservative but allows more granular clusters
            quality_threshold=0.5,  # Balanced quality filter
            use_dbscan=False,  # Use hierarchical for better control
            eps=0.12,  # Allow slightly larger clusters
            min_samples=2,
            min_cluster_size=2,
        )

        # Load and filter embeddings
        embeddings = clusterer.load_and_filter_embeddings(embeddings_dir)

        if not embeddings:
            logger.error("No valid embeddings found after filtering")
            return 1

        # Perform enhanced clustering
        clusters = clusterer.cluster_embeddings_enhanced(embeddings)

        if not clusters:
            logger.warning("No clusters found - try lowering thresholds")
            return 1

        # Create analysis
        os.makedirs(output_dir, exist_ok=True)
        clusterer.analyze_cluster_quality(clusters, output_dir)

        # Save enhanced cluster data
        enhanced_clusters_file = os.path.join(output_dir, "enhanced_clusters.json")
        save_enhanced_clusters(clusters, enhanced_clusters_file)

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("ENHANCED CLUSTERING RESULTS")
        logger.info("=" * 60)
        logger.info(f"High-quality clusters discovered: {len(clusters)}")

        for i, cluster in enumerate(clusters[:5]):
            logger.info(
                f"Cluster {i+1}: size={cluster.cluster_size}, "
                f"quality={cluster.quality_score:.3f}, "
                f"similarity={cluster.avg_similarity:.3f}"
            )

        logger.info(f"\nResults saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Enhanced clustering failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def save_enhanced_clusters(clusters: List[EnhancedPersonCluster], output_file: str):
    """Save enhanced clustering results."""

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    cluster_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_clusters": len(clusters),
            "clustering_type": "enhanced",
        },
        "clusters": [],
    }

    for cluster in clusters:
        cluster_info = {
            "cluster_id": convert_numpy_types(cluster.cluster_id),
            "cluster_size": convert_numpy_types(cluster.cluster_size),
            "quality_score": convert_numpy_types(cluster.quality_score),
            "avg_similarity": convert_numpy_types(cluster.avg_similarity),
            "min_similarity": convert_numpy_types(cluster.min_similarity),
            "videos_spanned": cluster.videos_spanned,
            "frame_range": [
                convert_numpy_types(cluster.frame_range[0]),
                convert_numpy_types(cluster.frame_range[1]),
            ],
            "representative": (
                {
                    "track_id": convert_numpy_types(
                        cluster.representative_embedding.track_id
                    ),
                    "frame_number": convert_numpy_types(
                        cluster.representative_embedding.frame_number
                    ),
                    "video_filename": cluster.representative_embedding.video_filename,
                    "confidence": convert_numpy_types(
                        cluster.representative_embedding.confidence
                    ),
                    "embedding_quality": convert_numpy_types(
                        cluster.representative_embedding.embedding_quality
                    ),
                }
                if cluster.representative_embedding
                else None
            ),
            "embeddings": [
                {
                    "track_id": convert_numpy_types(emb.track_id),
                    "frame_number": convert_numpy_types(emb.frame_number),
                    "video_filename": emb.video_filename,
                    "confidence": convert_numpy_types(emb.confidence),
                    "embedding_quality": convert_numpy_types(emb.embedding_quality),
                }
                for emb in cluster.embeddings
            ],
        }
        cluster_data["clusters"].append(cluster_info)

    with open(output_file, "w") as f:
        json.dump(cluster_data, f, indent=2)

    logger.info(f"Enhanced clusters saved to {output_file}")


if __name__ == "__main__":
    exit(main())
