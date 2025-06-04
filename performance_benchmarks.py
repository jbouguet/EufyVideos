#!/usr/bin/env python3
"""
Performance benchmarking utilities for YOLO object detection optimization.

This module provides tools to:
1. Compare original vs optimized YOLO detector performance
2. Benchmark different batch sizes and configurations  
3. Test GPU acceleration benefits
4. Generate detailed performance reports
5. Validate optimization improvements

Usage:
    # Run comprehensive benchmark
    python performance_benchmarks.py --video /path/to/video.mp4 --frames 100
    
    # Compare detectors
    python performance_benchmarks.py --compare --video /path/to/video.mp4
    
    # Batch size optimization
    python performance_benchmarks.py --optimize-batch --video /path/to/video.mp4
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
from tqdm import tqdm

from logging_config import create_logger
from object_detector_yolo import YoloObjectDetector
from object_detector_yolo_optimized import OptimizedYoloObjectDetector

logger = create_logger(__name__)


@dataclass
class BenchmarkResult:
    """Performance benchmark result data."""
    detector_type: str
    model_name: str
    device: str
    batch_size: Optional[int]
    num_frames: int
    total_time: float
    fps: float
    detections_found: int
    memory_peak_mb: float
    gpu_available: bool
    video_path: str
    timestamp: str


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for YOLO detectors.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def benchmark_detector(self, 
                          detector, 
                          video_path: str, 
                          num_frames: int = 50,
                          task: str = "detect") -> BenchmarkResult:
        """
        Benchmark a single detector configuration.
        
        Args:
            detector: YOLO detector instance
            video_path: Path to test video
            num_frames: Number of frames to process
            task: 'detect' or 'track'
            
        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Benchmarking {type(detector).__name__} on {num_frames} frames")
        
        # Monitor memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run benchmark
        start_time = time.time()
        
        if task == "track":
            results = detector.track_objects(video_path, num_frames)
        else:
            results = detector.detect_objects(video_path, num_frames)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        fps = num_frames / total_time if total_time > 0 else 0
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_peak = memory_after - memory_before
        
        # Get detector info
        detector_type = type(detector).__name__
        model_name = detector.model_name
        
        # Device and batch size info
        device = getattr(detector, 'device', 'cpu')
        batch_size = getattr(detector, 'batch_size', None)
        gpu_available = getattr(detector, 'device', 'cpu') != 'cpu'
        
        result = BenchmarkResult(
            detector_type=detector_type,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            num_frames=num_frames,
            total_time=total_time,
            fps=fps,
            detections_found=len(results),
            memory_peak_mb=memory_peak,
            gpu_available=gpu_available,
            video_path=os.path.basename(video_path),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        
        logger.info(f"Benchmark completed:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  FPS: {fps:.1f}")
        logger.info(f"  Detections: {len(results)}")
        logger.info(f"  Memory peak: {memory_peak:.1f} MB")
        
        return result

    def compare_detectors(self, 
                         video_path: str, 
                         num_frames: int = 50,
                         model_name: str = "yolo11x.pt") -> Dict[str, BenchmarkResult]:
        """
        Compare original vs optimized YOLO detector performance.
        
        Args:
            video_path: Path to test video
            num_frames: Number of frames to process
            model_name: YOLO model to use
            
        Returns:
            Dictionary with benchmark results for each detector
        """
        logger.info(f"Comparing detector performance on {video_path}")
        
        results = {}
        
        # Benchmark original detector
        logger.info("Testing original YOLO detector...")
        original_detector = YoloObjectDetector(model_name=model_name)
        results['original'] = self.benchmark_detector(
            original_detector, video_path, num_frames, task="detect"
        )
        
        # Benchmark optimized detector
        logger.info("Testing optimized YOLO detector...")
        optimized_detector = OptimizedYoloObjectDetector(model_name=model_name)
        results['optimized'] = self.benchmark_detector(
            optimized_detector, video_path, num_frames, task="detect"
        )
        
        # Calculate improvement metrics
        self._log_comparison(results['original'], results['optimized'])
        
        return results

    def optimize_batch_size(self, 
                           video_path: str, 
                           num_frames: int = 50,
                           batch_sizes: Optional[List[int]] = None) -> Dict[int, BenchmarkResult]:
        """
        Find optimal batch size for the optimized detector.
        
        Args:
            video_path: Path to test video
            num_frames: Number of frames to process
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary mapping batch size to benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        logger.info(f"Optimizing batch size with options: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            try:
                detector = OptimizedYoloObjectDetector(
                    model_name="yolo11x.pt", 
                    batch_size=batch_size
                )
                result = self.benchmark_detector(detector, video_path, num_frames)
                results[batch_size] = result
                
            except Exception as e:
                logger.error(f"Failed to test batch size {batch_size}: {e}")
                continue
        
        # Find optimal batch size
        if results:
            best_batch_size = max(results.keys(), key=lambda x: results[x].fps)
            logger.info(f"Optimal batch size: {best_batch_size} ({results[best_batch_size].fps:.1f} FPS)")
        
        return results

    def benchmark_different_models(self, 
                                  video_path: str, 
                                  num_frames: int = 50) -> Dict[str, BenchmarkResult]:
        """
        Benchmark different YOLO model sizes.
        
        Args:
            video_path: Path to test video
            num_frames: Number of frames to process
            
        Returns:
            Dictionary mapping model name to benchmark results
        """
        models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        results = {}
        
        logger.info(f"Benchmarking different model sizes: {models}")
        
        for model_name in models:
            logger.info(f"Testing model: {model_name}")
            try:
                detector = OptimizedYoloObjectDetector(model_name=model_name)
                result = self.benchmark_detector(detector, video_path, num_frames)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to test model {model_name}: {e}")
                continue
        
        return results

    def _log_comparison(self, original: BenchmarkResult, optimized: BenchmarkResult):
        """Log detailed comparison between two benchmark results."""
        speed_improvement = optimized.fps / original.fps if original.fps > 0 else 0
        time_improvement = original.total_time / optimized.total_time if optimized.total_time > 0 else 0
        
        logger.info("="*60)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*60)
        logger.info(f"Original detector:")
        logger.info(f"  Time: {original.total_time:.2f}s")
        logger.info(f"  FPS: {original.fps:.1f}")
        logger.info(f"  Memory: {original.memory_peak_mb:.1f} MB")
        logger.info(f"  Device: {original.device}")
        logger.info("")
        logger.info(f"Optimized detector:")
        logger.info(f"  Time: {optimized.total_time:.2f}s") 
        logger.info(f"  FPS: {optimized.fps:.1f}")
        logger.info(f"  Memory: {optimized.memory_peak_mb:.1f} MB")
        logger.info(f"  Device: {optimized.device}")
        logger.info(f"  Batch size: {optimized.batch_size}")
        logger.info("")
        logger.info(f"IMPROVEMENTS:")
        logger.info(f"  Speed: {speed_improvement:.1f}x faster")
        logger.info(f"  Time: {time_improvement:.1f}x reduction")
        logger.info("="*60)

    def save_results(self, output_path: str):
        """Save benchmark results to JSON file."""
        results_data = [asdict(result) for result in self.results]
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate a human-readable performance report."""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("YOLO DETECTOR PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append("")
        
        for i, result in enumerate(self.results, 1):
            report.append(f"{i}. {result.detector_type}")
            report.append(f"   Model: {result.model_name}")
            report.append(f"   Device: {result.device}")
            if result.batch_size:
                report.append(f"   Batch size: {result.batch_size}")
            report.append(f"   Performance: {result.fps:.1f} FPS ({result.total_time:.2f}s)")
            report.append(f"   Detections: {result.detections_found}")
            report.append(f"   Memory: {result.memory_peak_mb:.1f} MB")
            report.append(f"   Video: {result.video_path}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Command-line interface for running benchmarks."""
    parser = argparse.ArgumentParser(description="YOLO Performance Benchmarking")
    parser.add_argument("--video", required=True, help="Path to test video file")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to process")
    parser.add_argument("--compare", action="store_true", help="Compare original vs optimized")
    parser.add_argument("--optimize-batch", action="store_true", help="Find optimal batch size")
    parser.add_argument("--test-models", action="store_true", help="Test different model sizes")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    benchmark = PerformanceBenchmark()
    
    if args.compare:
        benchmark.compare_detectors(args.video, args.frames)
    
    if args.optimize_batch:
        benchmark.optimize_batch_size(args.video, args.frames)
    
    if args.test_models:
        benchmark.benchmark_different_models(args.video, args.frames)
    
    # Generate and print report
    report = benchmark.generate_report()
    print(report)
    
    # Save results if requested
    if args.output:
        benchmark.save_results(args.output)


if __name__ == "__main__":
    main()