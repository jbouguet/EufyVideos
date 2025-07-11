#!/usr/bin/env python3
"""
Simple Higher Resolution Test

This script tests higher crop resolution by directly modifying the analysis_config.yaml
and re-running person recognition on a subset of videos.

We'll compare:
1. Current 224x224 (baseline from existing embeddings)
2. 384x384 (higher resolution test)

Usage:
    python test_higher_resolution.py
"""

import os
import shutil
import tempfile
from person_clustering import PersonClusterer
from logging_config import create_logger

logger = create_logger(__name__)


def backup_current_config():
    """Backup the current config file."""
    config_path = "/Users/jbouguet/Documents/EufySecurityVideos/python/EufyVideos/analysis_config.yaml"
    backup_path = config_path + ".backup"
    shutil.copy2(config_path, backup_path)
    logger.info(f"Config backed up to: {backup_path}")
    return backup_path


def restore_config(backup_path):
    """Restore config from backup."""
    config_path = "/Users/jbouguet/Documents/EufySecurityVideos/python/EufyVideos/analysis_config.yaml"
    shutil.copy2(backup_path, config_path)
    logger.info("Config restored from backup")


def modify_config_for_higher_resolution():
    """Modify config to use 384x384 resolution and clean output paths."""
    config_path = "/Users/jbouguet/Documents/EufySecurityVideos/python/EufyVideos/analysis_config.yaml"
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace person crop size
    content = content.replace('person_crop_size: [224, 224]', 'person_crop_size: [384, 384]')
    
    # Ensure we use a different output directory for high-res test
    content = content.replace(
        'person_recognition/embeddings', 
        'person_recognition_384/embeddings'
    )
    content = content.replace(
        'person_recognition/crops',
        'person_recognition_384/crops'  
    )
    content = content.replace(
        'person_recognition/persons.json',
        'person_recognition_384/persons.json'
    )
    
    with open(config_path, 'w') as f:
        f.write(content)
    
    logger.info("Config modified for 384x384 resolution test")


def clean_high_res_outputs():
    """Clean previous high-res test outputs."""
    high_res_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition_384"
    if os.path.exists(high_res_dir):
        shutil.rmtree(high_res_dir)
        logger.info("Cleaned previous high-res outputs")


def run_person_recognition():
    """Run person recognition with current config."""
    from story_creator import StoryCreator
    
    config_path = "/Users/jbouguet/Documents/EufySecurityVideos/python/EufyVideos/analysis_config.yaml"
    
    try:
        creator = StoryCreator(config_path)
        creator.process_all_stories()
        logger.info("Person recognition completed successfully")
        return True
    except Exception as e:
        logger.error(f"Person recognition failed: {e}")
        return False


def cluster_and_analyze(embeddings_dir, resolution_name):
    """Cluster embeddings and return results."""
    logger.info(f"Clustering embeddings for {resolution_name}")
    
    # Use the best settings from our parameter tuning
    clusterer = PersonClusterer(
        similarity_threshold=0.91,  # Hierarchical conservative
        quality_threshold=0.6,
        use_dbscan=False,  # Use hierarchical clustering
        min_cluster_size=2
    )
    
    # Load embeddings
    embeddings = clusterer.load_and_filter_embeddings(embeddings_dir)
    
    if not embeddings:
        logger.warning(f"No embeddings found in {embeddings_dir}")
        return None
    
    # Perform clustering  
    clusters = clusterer.cluster_embeddings(embeddings)
    
    if not clusters:
        logger.warning(f"No clusters found for {resolution_name}")
        return None
    
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
        'largest_cluster': max(sizes),
        'smallest_cluster': min(sizes),
        'clusters': clusters
    }
    
    logger.info(f"{resolution_name} results:")
    logger.info(f"  Embeddings: {results['num_embeddings']}")
    logger.info(f"  Clusters: {results['num_clusters']}")
    logger.info(f"  Avg quality: {results['avg_quality']:.3f}")
    logger.info(f"  Min quality: {results['min_quality']:.3f}")
    logger.info(f"  Largest cluster: {results['largest_cluster']}")
    
    return results


def compare_resolutions():
    """Compare clustering results between resolutions."""
    logger.info("=" * 60)
    logger.info("CROP RESOLUTION COMPARISON")
    logger.info("=" * 60)
    
    # Test baseline 224x224 (existing embeddings)
    baseline_embeddings = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/embeddings"
    baseline_results = cluster_and_analyze(baseline_embeddings, "224x224 (baseline)")
    
    if not baseline_results:
        logger.error("Could not analyze baseline 224x224 results")
        return
    
    # Backup current config
    backup_path = backup_current_config()
    
    try:
        # Clean previous high-res outputs
        clean_high_res_outputs()
        
        # Modify config for 384x384
        modify_config_for_higher_resolution()
        
        # Run person recognition at 384x384
        logger.info("Running person recognition at 384x384...")
        if run_person_recognition():
            # Test high-res results
            high_res_embeddings = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition_384/embeddings"
            high_res_results = cluster_and_analyze(high_res_embeddings, "384x384 (high-res)")
            
            if high_res_results:
                # Compare results
                analyze_comparison(baseline_results, high_res_results)
            else:
                logger.error("Could not analyze high-res results")
        else:
            logger.error("High-resolution person recognition failed")
    
    finally:
        # Always restore config
        restore_config(backup_path)


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
    
    logger.info(f"{'Avg Quality':<20} {baseline['avg_quality']:<15.3f} {high_res['avg_quality']:<15.3f} "
               f"{high_res['avg_quality'] - baseline['avg_quality']:+.3f}")
    
    logger.info(f"{'Min Quality':<20} {baseline['min_quality']:<15.3f} {high_res['min_quality']:<15.3f} "
               f"{high_res['min_quality'] - baseline['min_quality']:+.3f}")
    
    logger.info(f"{'Max Quality':<20} {baseline['max_quality']:<15.3f} {high_res['max_quality']:<15.3f} "
               f"{high_res['max_quality'] - baseline['max_quality']:+.3f}")
    
    logger.info(f"{'Largest Cluster':<20} {baseline['largest_cluster']:<15} {high_res['largest_cluster']:<15} "
               f"{high_res['largest_cluster'] - baseline['largest_cluster']:+d}")
    
    # Analysis
    logger.info("\n" + "=" * 50)
    logger.info("ANALYSIS & RECOMMENDATIONS")
    logger.info("=" * 50)
    
    quality_improvement = high_res['avg_quality'] - baseline['avg_quality']
    precision_improvement = high_res['min_quality'] - baseline['min_quality']
    
    if quality_improvement > 0.01:  # More than 1% improvement
        logger.info(f"✅ SIGNIFICANT QUALITY IMPROVEMENT: +{quality_improvement:.3f}")
        logger.info(f"✅ Higher resolution (384x384) provides better clustering")
        logger.info(f"✅ Recommended: Switch to 384x384 crop resolution")
        
        if precision_improvement > 0.01:
            logger.info(f"✅ PRECISION ALSO IMPROVED: +{precision_improvement:.3f}")
            logger.info(f"✅ Higher resolution reduces risk of mixed-person clusters")
    
    elif quality_improvement > 0.005:  # Small but meaningful improvement
        logger.info(f"⚖️  MODERATE QUALITY IMPROVEMENT: +{quality_improvement:.3f}")
        logger.info(f"⚖️  Higher resolution provides some benefit")
        logger.info(f"⚖️  Consider switching if computational cost is acceptable")
    
    else:
        logger.info(f"📊 NO SIGNIFICANT IMPROVEMENT: {quality_improvement:+.3f}")
        logger.info(f"📊 Current 224x224 resolution appears optimal")
        logger.info(f"📊 Recommended: Keep current 224x224 resolution")
    
    # Cluster count analysis
    cluster_change = high_res['num_clusters'] - baseline['num_clusters']
    if cluster_change > 0:
        logger.info(f"\n📈 MORE CLUSTERS: +{cluster_change} clusters")
        logger.info(f"   This suggests higher precision (more conservative clustering)")
    elif cluster_change < 0:
        logger.info(f"\n📉 FEWER CLUSTERS: {cluster_change} clusters")
        logger.info(f"   This suggests some over-clustering in baseline")
    else:
        logger.info(f"\n📊 SAME CLUSTER COUNT: No change in clustering behavior")
    
    # Final recommendation
    logger.info(f"\n🎯 FINAL RECOMMENDATION:")
    if quality_improvement > 0.01 or precision_improvement > 0.01:
        logger.info(f"   Use 384x384 crop resolution for better clustering quality")
        logger.info(f"   Update analysis_config.yaml: person_crop_size: [384, 384]")
    else:
        logger.info(f"   Keep current 224x224 crop resolution")
        logger.info(f"   Higher resolution does not provide meaningful benefit")
    
    # Save results
    save_comparison_results(baseline, high_res)


def save_comparison_results(baseline, high_res):
    """Save comparison results to file."""
    output_dir = "/Users/jbouguet/Documents/EufySecurityVideos/record/person_recognition/resolution_test"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "resolution_comparison_results.txt")
    
    with open(results_file, 'w') as f:
        f.write("CROP RESOLUTION COMPARISON RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 10 + "\n")
        f.write(f"224x224 (baseline): {baseline['num_clusters']} clusters, "
               f"quality {baseline['avg_quality']:.3f}\n")
        f.write(f"384x384 (high-res): {high_res['num_clusters']} clusters, "
               f"quality {high_res['avg_quality']:.3f}\n")
        
        f.write(f"\nQuality improvement: {high_res['avg_quality'] - baseline['avg_quality']:+.3f}\n")
        f.write(f"Precision improvement: {high_res['min_quality'] - baseline['min_quality']:+.3f}\n")
        f.write(f"Cluster count change: {high_res['num_clusters'] - baseline['num_clusters']:+d}\n")
        
        f.write("\nRECOMMENDATION\n")
        f.write("-" * 15 + "\n")
        
        quality_improvement = high_res['avg_quality'] - baseline['avg_quality']
        if quality_improvement > 0.01:
            f.write("Switch to 384x384 resolution for better clustering quality\n")
            f.write("Update config: person_crop_size: [384, 384]\n")
        else:
            f.write("Keep current 224x224 resolution\n")
            f.write("Higher resolution does not provide meaningful benefit\n")
    
    logger.info(f"Comparison results saved to: {results_file}")


def main():
    """Run higher resolution test."""
    
    logger.info("🔍 HIGHER RESOLUTION CLUSTERING TEST")
    logger.info("=" * 60)
    logger.info("Comparing 224x224 (current) vs 384x384 (higher resolution)")
    logger.info("This will test if higher resolution crops improve clustering")
    logger.info("=" * 60)
    
    try:
        compare_resolutions()
        
        logger.info(f"\n✅ RESOLUTION TEST COMPLETED")
        logger.info(f"Check the analysis above for recommendations")
        
        return 0
        
    except Exception as e:
        logger.error(f"Resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())