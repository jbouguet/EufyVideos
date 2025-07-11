#!/usr/bin/env python3
"""
Comprehensive Parameter Tuning System for Person Clustering

This script helps you systematically test different parameter combinations
to find the optimal settings for high-precision person clustering.

The goal is to find settings that minimize false positives (different people
in same cluster) while accepting some false negatives (same person split).

Usage:
    python parameter_tuning_system.py
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from person_clustering import PersonClusterer, PersonCluster
from logging_config import create_logger

logger = create_logger(__name__)


@dataclass
class ParameterSet:
    """Configuration for a parameter test run."""
    name: str
    similarity_threshold: float
    quality_threshold: float
    use_dbscan: bool
    eps: float
    min_samples: int
    min_cluster_size: int
    description: str


@dataclass 
class ClusteringResults:
    """Results from a clustering run."""
    parameter_set: ParameterSet
    num_clusters: int
    total_embeddings: int
    avg_cluster_size: float
    avg_quality: float
    min_quality: float
    max_quality: float
    largest_cluster: int
    smallest_cluster: int
    clusters: List[PersonCluster]


class ParameterTuningSystem:
    """Systematic parameter tuning for person clustering."""
    
    def __init__(self, embeddings_dir: str, output_dir: str):
        """
        Initialize the tuning system.
        
        Args:
            embeddings_dir: Directory containing embedding files
            output_dir: Directory to save tuning results
        """
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.embeddings = None  # Cache embeddings
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def load_embeddings_once(self):
        """Load embeddings once and cache them for all tests."""
        if self.embeddings is None:
            logger.info("Loading embeddings for parameter tuning...")
            
            # Use a basic clusterer just to load embeddings
            basic_clusterer = PersonClusterer(quality_threshold=0.0)  # Load all
            self.embeddings = basic_clusterer.load_and_filter_embeddings(self.embeddings_dir)
            
            logger.info(f"Loaded {len(self.embeddings)} embeddings for testing")
    
    def test_parameter_set(self, param_set: ParameterSet) -> ClusteringResults:
        """
        Test a single parameter configuration.
        
        Args:
            param_set: Parameter configuration to test
            
        Returns:
            ClusteringResults with outcome
        """
        logger.info(f"Testing: {param_set.name}")
        
        # Create clusterer with specified parameters
        clusterer = PersonClusterer(
            similarity_threshold=param_set.similarity_threshold,
            quality_threshold=param_set.quality_threshold,
            use_dbscan=param_set.use_dbscan,
            eps=param_set.eps,
            min_samples=param_set.min_samples,
            min_cluster_size=param_set.min_cluster_size
        )
        
        # Filter embeddings based on quality threshold
        filtered_embeddings = clusterer._filter_low_quality_embeddings(self.embeddings)
        
        if not filtered_embeddings:
            logger.warning(f"No embeddings passed quality filter for {param_set.name}")
            return ClusteringResults(
                parameter_set=param_set,
                num_clusters=0,
                total_embeddings=0,
                avg_cluster_size=0,
                avg_quality=0,
                min_quality=0,
                max_quality=0,
                largest_cluster=0,
                smallest_cluster=0,
                clusters=[]
            )
        
        # Perform clustering
        clusters = clusterer.cluster_embeddings(filtered_embeddings)
        
        if not clusters:
            logger.warning(f"No clusters found for {param_set.name}")
            return ClusteringResults(
                parameter_set=param_set,
                num_clusters=0,
                total_embeddings=len(filtered_embeddings),
                avg_cluster_size=0,
                avg_quality=0,
                min_quality=0,
                max_quality=0,
                largest_cluster=0,
                smallest_cluster=0,
                clusters=[]
            )
        
        # Calculate statistics
        total_clustered = sum(c.cluster_size for c in clusters)
        avg_cluster_size = total_clustered / len(clusters)
        qualities = [c.quality_score for c in clusters]
        sizes = [c.cluster_size for c in clusters]
        
        results = ClusteringResults(
            parameter_set=param_set,
            num_clusters=len(clusters),
            total_embeddings=len(filtered_embeddings),
            avg_cluster_size=avg_cluster_size,
            avg_quality=np.mean(qualities),
            min_quality=np.min(qualities),
            max_quality=np.max(qualities),
            largest_cluster=np.max(sizes),
            smallest_cluster=np.min(sizes),
            clusters=clusters
        )
        
        logger.info(f"  → {len(clusters)} clusters, avg size: {avg_cluster_size:.1f}, "
                   f"avg quality: {np.mean(qualities):.3f}")
        
        return results
    
    def define_parameter_sets(self) -> List[ParameterSet]:
        """
        Define parameter sets to test.
        
        Returns comprehensive range from ultra-conservative to permissive.
        """
        parameter_sets = []
        
        # Ultra-conservative: Maximum precision
        parameter_sets.append(ParameterSet(
            name="ultra_conservative",
            similarity_threshold=0.95,
            quality_threshold=0.8,
            use_dbscan=True,
            eps=0.05,
            min_samples=1,
            min_cluster_size=2,
            description="Maximum precision - may over-split same person"
        ))
        
        # Very conservative: High precision
        parameter_sets.append(ParameterSet(
            name="very_conservative", 
            similarity_threshold=0.93,
            quality_threshold=0.7,
            use_dbscan=True,
            eps=0.07,
            min_samples=1,
            min_cluster_size=2,
            description="High precision with some flexibility"
        ))
        
        # Conservative: Recommended starting point
        parameter_sets.append(ParameterSet(
            name="conservative",
            similarity_threshold=0.91,
            quality_threshold=0.6,
            use_dbscan=True,
            eps=0.09,
            min_samples=1,
            min_cluster_size=2,
            description="Good precision-recall balance"
        ))
        
        # Moderate: Balanced approach
        parameter_sets.append(ParameterSet(
            name="moderate",
            similarity_threshold=0.88,
            quality_threshold=0.5,
            use_dbscan=True,
            eps=0.12,
            min_samples=2,
            min_cluster_size=2,
            description="Balanced clustering"
        ))
        
        # Permissive: Fewer clusters
        parameter_sets.append(ParameterSet(
            name="permissive",
            similarity_threshold=0.85,
            quality_threshold=0.4,
            use_dbscan=True,
            eps=0.15,
            min_samples=2,
            min_cluster_size=3,
            description="Fewer clusters, may mix some people"
        ))
        
        # Test DBSCAN vs Hierarchical
        parameter_sets.append(ParameterSet(
            name="hierarchical_conservative",
            similarity_threshold=0.91,
            quality_threshold=0.6,
            use_dbscan=False,  # Use hierarchical
            eps=0.09,  # Not used for hierarchical
            min_samples=1,  # Not used for hierarchical
            min_cluster_size=2,
            description="Hierarchical clustering - conservative"
        ))
        
        # Test different min_samples values
        for min_samples in [1, 2, 3]:
            parameter_sets.append(ParameterSet(
                name=f"dbscan_samples_{min_samples}",
                similarity_threshold=0.90,
                quality_threshold=0.6,
                use_dbscan=True,
                eps=0.10,
                min_samples=min_samples,
                min_cluster_size=2,
                description=f"DBSCAN with min_samples={min_samples}"
            ))
        
        # Test range of similarity thresholds
        for threshold in [0.87, 0.89, 0.90, 0.92, 0.94]:
            parameter_sets.append(ParameterSet(
                name=f"similarity_{threshold:.2f}",
                similarity_threshold=threshold,
                quality_threshold=0.6,
                use_dbscan=True,
                eps=1.0 - threshold,  # Corresponding eps
                min_samples=1,
                min_cluster_size=2,
                description=f"Similarity threshold sweep: {threshold}"
            ))
        
        return parameter_sets
    
    def run_comprehensive_tuning(self) -> List[ClusteringResults]:
        """
        Run comprehensive parameter tuning.
        
        Returns:
            List of ClusteringResults for all tested parameter sets
        """
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE PARAMETER TUNING")
        logger.info("=" * 60)
        
        # Load embeddings once
        self.load_embeddings_once()
        
        # Define parameter sets to test
        parameter_sets = self.define_parameter_sets()
        
        logger.info(f"Testing {len(parameter_sets)} parameter configurations...")
        
        # Test each parameter set
        all_results = []
        for i, param_set in enumerate(parameter_sets):
            logger.info(f"\n[{i+1}/{len(parameter_sets)}] {param_set.description}")
            
            try:
                results = self.test_parameter_set(param_set)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to test {param_set.name}: {e}")
                continue
        
        return all_results
    
    def analyze_results(self, all_results: List[ClusteringResults]):
        """
        Analyze and report tuning results.
        
        Args:
            all_results: Results from all parameter tests
        """
        logger.info("\n" + "=" * 80)
        logger.info("PARAMETER TUNING ANALYSIS")
        logger.info("=" * 80)
        
        # Create summary table
        logger.info(f"\n{'Name':<25} {'Clusters':<8} {'Avg Size':<8} {'Quality':<8} {'Min Qual':<8} {'Max Size':<8}")
        logger.info("-" * 80)
        
        for result in all_results:
            if result.num_clusters > 0:
                logger.info(f"{result.parameter_set.name:<25} {result.num_clusters:<8} "
                           f"{result.avg_cluster_size:<8.1f} {result.avg_quality:<8.3f} "
                           f"{result.min_quality:<8.3f} {result.largest_cluster:<8}")
            else:
                logger.info(f"{result.parameter_set.name:<25} {'0':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
        
        # Recommendations
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 60)
        
        # For high precision (your use case)
        high_precision_candidates = [r for r in all_results 
                                   if r.num_clusters > 0 and r.avg_quality > 0.9 and r.largest_cluster < 100]
        
        if high_precision_candidates:
            best_precision = max(high_precision_candidates, key=lambda x: x.min_quality)
            logger.info(f"\nBest for HIGH PRECISION (your use case):")
            logger.info(f"  Parameter Set: {best_precision.parameter_set.name}")
            logger.info(f"  Similarity Threshold: {best_precision.parameter_set.similarity_threshold}")
            logger.info(f"  Quality Threshold: {best_precision.parameter_set.quality_threshold}")
            logger.info(f"  Results: {best_precision.num_clusters} clusters, "
                       f"min quality: {best_precision.min_quality:.3f}")
            logger.info(f"  Description: {best_precision.parameter_set.description}")
        
        # For balanced approach
        balanced_candidates = [r for r in all_results 
                             if r.num_clusters > 0 and 10 <= r.num_clusters <= 50]
        
        if balanced_candidates:
            best_balanced = max(balanced_candidates, key=lambda x: x.avg_quality)
            logger.info(f"\nBest for BALANCED approach:")
            logger.info(f"  Parameter Set: {best_balanced.parameter_set.name}")
            logger.info(f"  Similarity Threshold: {best_balanced.parameter_set.similarity_threshold}")
            logger.info(f"  Results: {best_balanced.num_clusters} clusters, "
                       f"avg quality: {best_balanced.avg_quality:.3f}")
            logger.info(f"  Description: {best_balanced.parameter_set.description}")
        
        # Warning about potential issues
        logger.info(f"\n⚠️  ANALYSIS NOTES:")
        logger.info(f"  - Higher cluster counts = more conservative (good for precision)")
        logger.info(f"  - Large clusters (>100) may contain multiple people")
        logger.info(f"  - Quality >0.9 suggests pure clusters")
        logger.info(f"  - Start with highest precision setting and relax as needed")
    
    def save_detailed_results(self, all_results: List[ClusteringResults]):
        """
        Save detailed results to files.
        
        Args:
            all_results: Results to save
        """
        # Save parameter comparison
        comparison_file = os.path.join(self.output_dir, "parameter_comparison.json")
        
        comparison_data = {
            "metadata": {
                "total_embeddings": len(self.embeddings) if self.embeddings else 0,
                "parameter_sets_tested": len(all_results),
                "timestamp": "2025-06-12"
            },
            "results": []
        }
        
        for result in all_results:
            result_data = {
                "parameter_set": {
                    "name": result.parameter_set.name,
                    "similarity_threshold": result.parameter_set.similarity_threshold,
                    "quality_threshold": result.parameter_set.quality_threshold,
                    "use_dbscan": result.parameter_set.use_dbscan,
                    "eps": result.parameter_set.eps,
                    "min_samples": result.parameter_set.min_samples,
                    "min_cluster_size": result.parameter_set.min_cluster_size,
                    "description": result.parameter_set.description
                },
                "results": {
                    "num_clusters": int(result.num_clusters),
                    "total_embeddings": int(result.total_embeddings),
                    "avg_cluster_size": float(result.avg_cluster_size),
                    "avg_quality": float(result.avg_quality),
                    "min_quality": float(result.min_quality),
                    "max_quality": float(result.max_quality),
                    "largest_cluster": int(result.largest_cluster),
                    "smallest_cluster": int(result.smallest_cluster)
                }
            }
            comparison_data["results"].append(result_data)
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Detailed results saved to: {comparison_file}")
        
        # Save recommendations file
        recommendations_file = os.path.join(self.output_dir, "tuning_recommendations.txt")
        
        with open(recommendations_file, 'w') as f:
            f.write("PARAMETER TUNING RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write("Based on your requirement for high precision (avoid mixing different people),\n")
            f.write("here are the key findings and recommendations:\n\n")
            
            # Find best settings
            high_precision = [r for r in all_results if r.num_clusters > 0 and r.avg_quality > 0.9]
            
            if high_precision:
                best = max(high_precision, key=lambda x: x.min_quality)
                
                f.write("RECOMMENDED SETTINGS (High Precision)\n")
                f.write("-" * 30 + "\n")
                f.write(f"similarity_threshold = {best.parameter_set.similarity_threshold}\n")
                f.write(f"quality_threshold = {best.parameter_set.quality_threshold}\n")
                f.write(f"use_dbscan = {best.parameter_set.use_dbscan}\n")
                f.write(f"eps = {best.parameter_set.eps}\n")
                f.write(f"min_samples = {best.parameter_set.min_samples}\n")
                f.write(f"min_cluster_size = {best.parameter_set.min_cluster_size}\n\n")
                
                f.write(f"Expected Results:\n")
                f.write(f"- {best.num_clusters} clusters\n")
                f.write(f"- Average quality: {best.avg_quality:.3f}\n")
                f.write(f"- Minimum quality: {best.min_quality:.3f}\n")
                f.write(f"- Largest cluster: {best.largest_cluster} crops\n\n")
            
            f.write("PARAMETER EFFECTS\n")
            f.write("-" * 20 + "\n")
            f.write("similarity_threshold: Higher = more conservative, more clusters\n")
            f.write("quality_threshold: Higher = only best embeddings, fewer total crops\n")
            f.write("eps: Lower = tighter clusters, more clusters\n")
            f.write("min_samples: Lower = allows smaller clusters, more total clusters\n")
            f.write("min_cluster_size: Filter for post-processing, keeps only larger clusters\n\n")
            
            f.write("NEXT STEPS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Start with recommended high precision settings\n")
            f.write("2. Generate galleries and review cluster purity\n")
            f.write("3. If clusters are too split, slightly lower similarity_threshold\n")
            f.write("4. If you see mixed people, increase similarity_threshold\n")
            f.write("5. Manual labeling will handle same-person clusters later\n")
        
        logger.info(f"Recommendations saved to: {recommendations_file}")


def main():
    """Run comprehensive parameter tuning."""
    
    # Paths
    embeddings_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/embeddings"
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/parameter_tuning"
    
    logger.info("🔧 PARAMETER TUNING SYSTEM")
    logger.info("=" * 60)
    logger.info("This will systematically test different parameter combinations")
    logger.info("to find optimal settings for high-precision person clustering.")
    logger.info("=" * 60)
    
    if not os.path.exists(embeddings_dir):
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        logger.error("Please run person recognition first to generate embeddings")
        return 1
    
    try:
        # Create tuning system
        tuning_system = ParameterTuningSystem(embeddings_dir, output_dir)
        
        # Run comprehensive tuning
        all_results = tuning_system.run_comprehensive_tuning()
        
        if not all_results:
            logger.error("No valid results from parameter tuning")
            return 1
        
        # Analyze results
        tuning_system.analyze_results(all_results)
        
        # Save detailed results
        tuning_system.save_detailed_results(all_results)
        
        logger.info(f"\n🎯 TUNING COMPLETED")
        logger.info(f"Results and recommendations saved to: {output_dir}")
        logger.info(f"Next step: Review recommendations and test selected parameters")
        
        return 0
        
    except Exception as e:
        logger.error(f"Parameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())