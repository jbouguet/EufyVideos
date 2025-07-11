#!/usr/bin/env python3
"""
Crop Resolution Comparison Test

This script tests different crop resolutions to determine their impact on 
embedding quality and clustering performance. It runs the same clustering
algorithm with different crop resolutions and compares the results.

The goal is to find the optimal resolution that provides the best trade-off
between clustering quality and computational cost.

Usage:
    python crop_resolution_comparison.py
"""

import os
import json
import shutil
import tempfile
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import yaml

from person_clustering import PersonClusterer, PersonCluster
from person_embedding import PersonEmbeddingGenerator, PersonEmbedding  
from logging_config import create_logger

logger = create_logger(__name__)


@dataclass
class ResolutionTestResult:
    """Results from testing a specific crop resolution."""
    resolution: Tuple[int, int]
    num_embeddings: int
    num_clusters: int
    avg_cluster_size: float
    avg_quality: float
    min_quality: float
    max_quality: float
    largest_cluster: int
    smallest_cluster: int
    processing_time: float
    clusters: List[PersonCluster]


class CropResolutionTester:
    """Test different crop resolutions and their impact on clustering."""
    
    def __init__(self, 
                 video_files: List[str],
                 config_template_path: str,
                 base_output_dir: str):
        """
        Initialize the resolution tester.
        
        Args:
            video_files: List of video files to process  
            config_template_path: Path to analysis config template
            base_output_dir: Directory to save all test results
        """
        self.video_files = video_files
        self.config_template_path = config_template_path
        self.base_output_dir = base_output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(base_output_dir, exist_ok=True)
        
        logger.info(f"Crop resolution tester initialized")
        logger.info(f"Video files: {len(video_files)}")
        logger.info(f"Output directory: {base_output_dir}")
    
    def load_config_template(self) -> Dict[str, Any]:
        """Load the analysis config template."""
        with open(self.config_template_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_resolution_config(self, 
                               resolution: Tuple[int, int], 
                               test_dir: str) -> str:
        """
        Create a modified config file for specific resolution.
        
        Args:
            resolution: (width, height) for person crops
            test_dir: Directory for this test's outputs
            
        Returns:
            Path to the created config file
        """
        config = self.load_config_template()
        
        # Update the person crop size
        config['stories'][0]['tag_processing_config']['person_crop_size'] = list(resolution)
        
        # Update output paths to be test-specific
        root_db = config['directories']['root_database']
        test_person_dir = os.path.join(test_dir, "person_recognition")
        
        # Update person recognition paths
        config['person_recognition']['database_file'] = os.path.join(test_person_dir, "persons.json")
        config['person_recognition']['embeddings_dir'] = os.path.join(test_person_dir, "embeddings") 
        config['person_recognition']['crops_dir'] = os.path.join(test_person_dir, "crops")
        
        # Update config paths in the story
        config['stories'][0]['tag_processing_config']['person_database_file'] = os.path.join(test_person_dir, "persons.json")
        config['stories'][0]['tag_processing_config']['person_embeddings_file'] = os.path.join(test_person_dir, "embeddings", "person_embeddings.json")
        config['stories'][0]['tag_processing_config']['person_crops_dir'] = os.path.join(test_person_dir, "crops")
        
        # Save modified config
        config_path = os.path.join(test_dir, f"config_resolution_{resolution[0]}x{resolution[1]}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def run_person_recognition(self, config_path: str) -> Tuple[str, float]:
        """
        Run person recognition with the specified config.
        
        Args:
            config_path: Path to config file
            
        Returns:
            (embeddings_dir, processing_time)
        """
        import time
        start_time = time.time()
        
        # Import and run the story creator
        from story_creator import StoryCreator
        
        try:
            creator = StoryCreator(config_path)
            creator.process_all_stories()
            processing_time = time.time() - start_time
            
            # Find the embeddings directory
            config = yaml.safe_load(open(config_path, 'r'))
            embeddings_dir = config['person_recognition']['embeddings_dir']
            
            return embeddings_dir, processing_time
            
        except Exception as e:
            logger.error(f"Failed to run person recognition: {e}")
            return "", 0.0
    
    def cluster_embeddings_at_resolution(self, 
                                       embeddings_dir: str,
                                       processing_time: float) -> ResolutionTestResult:
        """
        Cluster embeddings and return results.
        
        Args:
            embeddings_dir: Directory containing embeddings
            processing_time: Time taken for person recognition
            
        Returns:
            ResolutionTestResult with clustering outcome
        """
        # Use the best clustering parameters from our tuning
        clusterer = PersonClusterer(
            similarity_threshold=0.91,  # Conservative
            quality_threshold=0.6,
            use_dbscan=False,  # Use hierarchical (best from tuning)
            min_cluster_size=2
        )
        
        # Load embeddings
        embeddings = clusterer.load_and_filter_embeddings(embeddings_dir)
        
        if not embeddings:
            logger.warning(f"No embeddings found in {embeddings_dir}")
            return ResolutionTestResult(
                resolution=(0, 0),
                num_embeddings=0,
                num_clusters=0,
                avg_cluster_size=0,
                avg_quality=0,
                min_quality=0,
                max_quality=0,
                largest_cluster=0,
                smallest_cluster=0,
                processing_time=processing_time,
                clusters=[]
            )
        
        # Perform clustering
        clusters = clusterer.cluster_embeddings(embeddings)
        
        # Calculate statistics
        if clusters:
            total_clustered = sum(c.cluster_size for c in clusters)
            avg_cluster_size = total_clustered / len(clusters)
            qualities = [c.quality_score for c in clusters]
            sizes = [c.cluster_size for c in clusters]
            
            # Extract resolution from first embedding (all should be same resolution)
            # This is a bit of a hack since we don't directly store crop resolution
            resolution = (224, 224)  # Default, will be overridden by caller
            
            return ResolutionTestResult(
                resolution=resolution,
                num_embeddings=len(embeddings),
                num_clusters=len(clusters),
                avg_cluster_size=avg_cluster_size,
                avg_quality=sum(qualities) / len(qualities),
                min_quality=min(qualities),
                max_quality=max(qualities),
                largest_cluster=max(sizes),
                smallest_cluster=min(sizes),
                processing_time=processing_time,
                clusters=clusters
            )
        else:
            return ResolutionTestResult(
                resolution=(0, 0),
                num_embeddings=len(embeddings),
                num_clusters=0,
                avg_cluster_size=0,
                avg_quality=0,
                min_quality=0,
                max_quality=0,
                largest_cluster=0,
                smallest_cluster=0,
                processing_time=processing_time,
                clusters=[]
            )
    
    def test_resolution(self, resolution: Tuple[int, int]) -> ResolutionTestResult:
        """
        Test a specific crop resolution.
        
        Args:
            resolution: (width, height) for person crops
            
        Returns:
            ResolutionTestResult with complete results
        """
        logger.info(f"Testing resolution: {resolution[0]}x{resolution[1]}")
        
        # Create test directory
        test_dir = os.path.join(self.base_output_dir, f"resolution_{resolution[0]}x{resolution[1]}")
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # Create config for this resolution
            config_path = self.create_resolution_config(resolution, test_dir)
            logger.info(f"Created config: {config_path}")
            
            # Run person recognition
            embeddings_dir, processing_time = self.run_person_recognition(config_path)
            
            if not embeddings_dir or not os.path.exists(embeddings_dir):
                logger.error(f"No embeddings generated for resolution {resolution}")
                return ResolutionTestResult(
                    resolution=resolution,
                    num_embeddings=0,
                    num_clusters=0,
                    avg_cluster_size=0,
                    avg_quality=0,
                    min_quality=0,
                    max_quality=0,
                    largest_cluster=0,
                    smallest_cluster=0,
                    processing_time=processing_time,
                    clusters=[]
                )
            
            # Cluster embeddings
            result = self.cluster_embeddings_at_resolution(embeddings_dir, processing_time)
            result.resolution = resolution  # Override with actual resolution
            
            logger.info(f"Resolution {resolution[0]}x{resolution[1]} results:")
            logger.info(f"  Embeddings: {result.num_embeddings}")
            logger.info(f"  Clusters: {result.num_clusters}")
            logger.info(f"  Avg quality: {result.avg_quality:.3f}")
            logger.info(f"  Processing time: {result.processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to test resolution {resolution}: {e}")
            import traceback
            traceback.print_exc()
            
            return ResolutionTestResult(
                resolution=resolution,
                num_embeddings=0,
                num_clusters=0,
                avg_cluster_size=0,
                avg_quality=0,
                min_quality=0,
                max_quality=0,
                largest_cluster=0,
                smallest_cluster=0,
                processing_time=0,
                clusters=[]
            )
    
    def run_comprehensive_resolution_test(self) -> List[ResolutionTestResult]:
        """
        Test multiple resolutions comprehensively.
        
        Returns:
            List of ResolutionTestResult for all tested resolutions
        """
        # Define resolutions to test
        resolutions_to_test = [
            (224, 224),   # Current/baseline
            (320, 320),   # Moderate increase  
            (384, 384),   # High detail
            (512, 512),   # Very high detail
            # (640, 640),   # Ultra high (commented out - very slow)
        ]
        
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE CROP RESOLUTION COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Testing {len(resolutions_to_test)} different crop resolutions")
        logger.info(f"Resolutions: {resolutions_to_test}")
        
        results = []
        
        for i, resolution in enumerate(resolutions_to_test):
            logger.info(f"\n[{i+1}/{len(resolutions_to_test)}] Testing {resolution[0]}x{resolution[1]}")
            
            result = self.test_resolution(resolution)
            results.append(result)
            
            # Save intermediate results
            self.save_resolution_results(results)
        
        self.results = results
        return results
    
    def analyze_resolution_results(self, results: List[ResolutionTestResult]):
        """
        Analyze and report resolution comparison results.
        
        Args:
            results: List of resolution test results
        """
        logger.info("\n" + "=" * 80)
        logger.info("CROP RESOLUTION ANALYSIS")
        logger.info("=" * 80)
        
        # Create comparison table
        logger.info(f"\n{'Resolution':<12} {'Embeddings':<10} {'Clusters':<8} {'Quality':<8} {'Min Qual':<8} {'Time(s)':<8}")
        logger.info("-" * 70)
        
        for result in results:
            if result.num_embeddings > 0:
                logger.info(f"{result.resolution[0]}x{result.resolution[1]:<6} "
                           f"{result.num_embeddings:<10} {result.num_clusters:<8} "
                           f"{result.avg_quality:<8.3f} {result.min_quality:<8.3f} "
                           f"{result.processing_time:<8.1f}")
            else:
                logger.info(f"{result.resolution[0]}x{result.resolution[1]:<6} "
                           f"{'FAILED':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} "
                           f"{result.processing_time:<8.1f}")
        
        # Analysis and recommendations
        logger.info("\n" + "=" * 60)
        logger.info("RESOLUTION ANALYSIS")
        logger.info("=" * 60)
        
        valid_results = [r for r in results if r.num_embeddings > 0]
        
        if not valid_results:
            logger.error("No valid results to analyze!")
            return
        
        # Find best quality
        best_quality = max(valid_results, key=lambda x: x.avg_quality)
        logger.info(f"\nBest Quality: {best_quality.resolution[0]}x{best_quality.resolution[1]} "
                   f"(avg quality: {best_quality.avg_quality:.3f})")
        
        # Find best precision (highest min quality)
        best_precision = max(valid_results, key=lambda x: x.min_quality)
        logger.info(f"Best Precision: {best_precision.resolution[0]}x{best_precision.resolution[1]} "
                   f"(min quality: {best_precision.min_quality:.3f})")
        
        # Find fastest
        fastest = min(valid_results, key=lambda x: x.processing_time)
        logger.info(f"Fastest: {fastest.resolution[0]}x{fastest.resolution[1]} "
                   f"({fastest.processing_time:.1f}s)")
        
        # Quality vs time analysis
        baseline_224 = next((r for r in valid_results if r.resolution == (224, 224)), None)
        
        if baseline_224:
            logger.info(f"\nComparison to 224x224 baseline:")
            for result in valid_results:
                if result.resolution != (224, 224):
                    quality_improvement = result.avg_quality - baseline_224.avg_quality
                    time_factor = result.processing_time / baseline_224.processing_time
                    
                    logger.info(f"  {result.resolution[0]}x{result.resolution[1]}: "
                               f"Quality +{quality_improvement:+.3f}, "
                               f"Time {time_factor:.1f}x slower")
        
        # Recommendations
        logger.info(f"\n📋 RECOMMENDATIONS:")
        
        # For maximum quality
        if best_quality.avg_quality > baseline_224.avg_quality:
            logger.info(f"🏆 For MAXIMUM QUALITY: Use {best_quality.resolution[0]}x{best_quality.resolution[1]}")
            logger.info(f"   - Quality improvement: +{best_quality.avg_quality - baseline_224.avg_quality:.3f}")
            logger.info(f"   - Time cost: {best_quality.processing_time / baseline_224.processing_time:.1f}x slower")
        
        # For balanced approach
        quality_threshold = baseline_224.avg_quality + 0.01  # At least 1% improvement
        balanced_options = [r for r in valid_results 
                          if r.avg_quality > quality_threshold and 
                          r.processing_time < baseline_224.processing_time * 2]  # Max 2x slower
        
        if balanced_options:
            best_balanced = max(balanced_options, key=lambda x: x.avg_quality / (x.processing_time / baseline_224.processing_time))
            logger.info(f"⚖️  For BALANCED APPROACH: Use {best_balanced.resolution[0]}x{best_balanced.resolution[1]}")
            logger.info(f"   - Good quality improvement with reasonable time cost")
        
        # If no improvement
        if best_quality.avg_quality <= baseline_224.avg_quality:
            logger.info(f"📊 RESULT: 224x224 appears optimal for your data")
            logger.info(f"   - Higher resolutions did not improve clustering quality")
            logger.info(f"   - Stick with current 224x224 resolution")
    
    def save_resolution_results(self, results: List[ResolutionTestResult]):
        """Save resolution comparison results to files."""
        
        # Save JSON results
        results_file = os.path.join(self.base_output_dir, "resolution_comparison.json")
        
        results_data = {
            "metadata": {
                "test_date": "2025-06-13",
                "video_files": self.video_files,
                "total_resolutions_tested": len(results)
            },
            "results": []
        }
        
        for result in results:
            result_data = {
                "resolution": {
                    "width": result.resolution[0], 
                    "height": result.resolution[1]
                },
                "embeddings": int(result.num_embeddings),
                "clusters": int(result.num_clusters),
                "avg_cluster_size": float(result.avg_cluster_size),
                "avg_quality": float(result.avg_quality),
                "min_quality": float(result.min_quality),
                "max_quality": float(result.max_quality),
                "largest_cluster": int(result.largest_cluster),
                "smallest_cluster": int(result.smallest_cluster),
                "processing_time_seconds": float(result.processing_time)
            }
            results_data["results"].append(result_data)
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Resolution comparison results saved: {results_file}")
        
        # Save text summary
        summary_file = os.path.join(self.base_output_dir, "resolution_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("CROP RESOLUTION COMPARISON SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("TESTED RESOLUTIONS\n")
            f.write("-" * 20 + "\n")
            for result in results:
                if result.num_embeddings > 0:
                    f.write(f"{result.resolution[0]}x{result.resolution[1]}: "
                           f"{result.num_clusters} clusters, quality {result.avg_quality:.3f}, "
                           f"{result.processing_time:.1f}s\n")
                else:
                    f.write(f"{result.resolution[0]}x{result.resolution[1]}: FAILED\n")
            
            f.write("\nKEY FINDINGS\n")
            f.write("-" * 15 + "\n")
            
            valid_results = [r for r in results if r.num_embeddings > 0]
            if valid_results:
                best_quality = max(valid_results, key=lambda x: x.avg_quality)
                f.write(f"Best quality: {best_quality.resolution[0]}x{best_quality.resolution[1]} "
                       f"({best_quality.avg_quality:.3f})\n")
                
                fastest = min(valid_results, key=lambda x: x.processing_time)
                f.write(f"Fastest: {fastest.resolution[0]}x{fastest.resolution[1]} "
                       f"({fastest.processing_time:.1f}s)\n")
                
                baseline = next((r for r in valid_results if r.resolution == (224, 224)), None)
                if baseline and best_quality.avg_quality > baseline.avg_quality:
                    improvement = best_quality.avg_quality - baseline.avg_quality
                    f.write(f"Quality improvement over 224x224: +{improvement:.3f}\n")
                    f.write(f"Recommended: Use {best_quality.resolution[0]}x{best_quality.resolution[1]} for best results\n")
                else:
                    f.write("No significant improvement over 224x224 baseline\n")
                    f.write("Recommended: Keep current 224x224 resolution\n")
        
        logger.info(f"Resolution summary saved: {summary_file}")


def main():
    """Run comprehensive crop resolution comparison."""
    
    # Configuration
    video_files = [
        'T8600P1024260D5E_20241118084615.mp4',
        'T8600P1024260D5E_20241118084819.mp4', 
        'T8600P1024260D5E_20241118084902.mp4',
        'T8600P1024260D5E_20241118085102.mp4',
        'T8600P1024260D5E_20241118085306.mp4',
        'T8600P102338033E_20240930085536.mp4',
        'T8600P1024260D5E_20241119181809.mp4'
    ]
    
    config_template_path = "/Users/jbouguet/Documents/EufySecurityVideos/python/EufyVideos/analysis_config.yaml"
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/resolution_comparison"
    
    logger.info("🔍 CROP RESOLUTION COMPARISON")
    logger.info("=" * 60)
    logger.info("This test will compare different crop resolutions and their")
    logger.info("impact on embedding quality and clustering performance.")
    logger.info("=" * 60)
    
    try:
        # Create resolution tester
        tester = CropResolutionTester(video_files, config_template_path, output_dir)
        
        # Run comprehensive resolution test
        results = tester.run_comprehensive_resolution_test()
        
        if not results:
            logger.error("No resolution test results obtained")
            return 1
        
        # Analyze results
        tester.analyze_resolution_results(results)
        
        logger.info(f"\n🎯 RESOLUTION COMPARISON COMPLETED")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Check resolution_summary.txt for recommendations")
        
        return 0
        
    except Exception as e:
        logger.error(f"Resolution comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())