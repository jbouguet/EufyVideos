#!/usr/bin/env python3
"""
YOLO Performance Benchmark

Comprehensive benchmark comparing original YOLO vs optimized YOLO implementations
across different model sizes with fair comparison (batch_size=1, enable_gpu=True).

This benchmark helps understand when the optimized version provides meaningful 
performance improvements.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any
import gc
import torch

from logging_config import create_logger, set_all_loggers_level_and_format
from object_detector_yolo import YoloObjectDetector
from object_detector_yolo_optimized import OptimizedYoloObjectDetector
from video_metadata import VideoMetadata

# Configure logging to reduce noise
set_all_loggers_level_and_format(level="WARNING", extended_format=False)
logger = create_logger(__name__)

class YoloBenchmark:
    """Benchmark suite for comparing YOLO implementations."""
    
    def __init__(self, test_video_path: str):
        """
        Initialize benchmark with test video.
        
        Args:
            test_video_path: Path to test video file
        """
        self.test_video_path = test_video_path
        self.results = {}
        
        # Verify test video exists
        if not Path(test_video_path).exists():
            raise FileNotFoundError(f"Test video not found: {test_video_path}")
        
        # Get video metadata
        self.video_metadata = VideoMetadata.from_video_file(test_video_path)
        logger.info(f"Test video: {Path(test_video_path).name}")
        logger.info(f"Duration: {self.video_metadata.duration.total_seconds():.1f}s")
        
    def benchmark_detector(self, 
                          detector_class, 
                          model_name: str, 
                          implementation: str,
                          num_frames: int = 50) -> Dict[str, Any]:
        """
        Benchmark a specific detector implementation.
        
        Args:
            detector_class: YoloObjectDetector or OptimizedYoloObjectDetector
            model_name: YOLO model name (e.g., 'yolo11n.pt')
            implementation: 'original' or 'optimized'
            num_frames: Number of frames to process
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking {implementation} {model_name}")
        
        # Force garbage collection before test
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Initialize detector with fair comparison settings
        start_init = time.time()
        
        if implementation == 'original':
            detector = detector_class(
                model_name=model_name,
                conf_threshold=0.3,
                enable_gpu=True  # Fair comparison
            )
        else:  # optimized
            detector = detector_class(
                model_name=model_name,
                conf_threshold=0.3,
                batch_size=1,      # Fair comparison - no batch advantage
                enable_gpu=True    # Fair comparison
            )
        
        init_time = time.time() - start_init
        
        # Run detection benchmark
        start_detection = time.time()
        detections = detector.detect_objects(
            video_path=self.test_video_path,
            num_frames=num_frames
        )
        detection_time = time.time() - start_detection
        
        # Calculate metrics
        detection_fps = num_frames / detection_time if detection_time > 0 else 0
        
        # Run tracking benchmark
        start_tracking = time.time()
        tracks = detector.track_objects(
            video_path=self.test_video_path,
            num_frames=num_frames
        )
        tracking_time = time.time() - start_tracking
        
        # Calculate metrics
        tracking_fps = num_frames / tracking_time if tracking_time > 0 else 0
        
        results = {
            'model_name': model_name,
            'implementation': implementation,
            'num_frames': num_frames,
            'init_time': round(init_time, 3),
            'detection_time': round(detection_time, 3),
            'detection_fps': round(detection_fps, 2),
            'tracking_time': round(tracking_time, 3),
            'tracking_fps': round(tracking_fps, 2),
            'total_detections': len(detections),
            'total_tracks': len(tracks)
        }
        
        logger.info(f"  Init: {init_time:.3f}s")
        logger.info(f"  Detection: {detection_time:.3f}s ({detection_fps:.1f} FPS)")
        logger.info(f"  Tracking: {tracking_time:.3f}s ({tracking_fps:.1f} FPS)")
        logger.info(f"  Objects: {len(detections)} detections, {len(tracks)} tracks")
        
        return results
    
    def run_full_benchmark(self, num_frames: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run comprehensive benchmark across all YOLO models and implementations.
        
        Args:
            num_frames: Number of frames to process for each test
            
        Returns:
            Complete benchmark results
        """
        logger.info("=" * 80)
        logger.info("YOLO PERFORMANCE BENCHMARK")
        logger.info("=" * 80)
        logger.info(f"Test video: {Path(self.test_video_path).name}")
        logger.info(f"Frames per test: {num_frames}")
        logger.info(f"GPU acceleration: {'MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'None'}")
        logger.info("")
        
        # Model configurations to test (starting with smaller models)
        models = [
            ('yolo11n.pt', 'YOLOv11 Nano'),
            ('yolo11s.pt', 'YOLOv11 Small'), 
            ('yolo11m.pt', 'YOLOv11 Medium'),
            # ('yolo11l.pt', 'YOLOv11 Large'),      # Commented out for faster testing
            # ('yolo11x.pt', 'YOLOv11 Extra Large'), # Commented out for faster testing
        ]
        
        results = {
            'original': [],
            'optimized': [],
            'metadata': {
                'test_video': str(self.test_video_path),
                'video_duration': self.video_metadata.duration.total_seconds(),
                'num_frames': num_frames,
                'gpu_available': torch.backends.mps.is_available() or torch.cuda.is_available(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for model_file, model_desc in models:
            logger.info(f"\n--- {model_desc} ({model_file}) ---")
            
            try:
                # Test original implementation
                original_result = self.benchmark_detector(
                    YoloObjectDetector, model_file, 'original', num_frames
                )
                results['original'].append(original_result)
                
                # Test optimized implementation
                optimized_result = self.benchmark_detector(
                    OptimizedYoloObjectDetector, model_file, 'optimized', num_frames
                )
                results['optimized'].append(optimized_result)
                
                # Calculate speedup
                det_speedup = (optimized_result['detection_fps'] / original_result['detection_fps'] 
                              if original_result['detection_fps'] > 0 else 0)
                track_speedup = (optimized_result['tracking_fps'] / original_result['tracking_fps'] 
                               if original_result['tracking_fps'] > 0 else 0)
                
                logger.info(f"  Speedup - Detection: {det_speedup:.2f}x, Tracking: {track_speedup:.2f}x")
                
            except Exception as e:
                logger.error(f"  Failed to benchmark {model_file}: {e}")
                continue
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str = "yolo_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of benchmark results."""
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        
        print(f"{'Model':<15} {'Task':<10} {'Original':<12} {'Optimized':<12} {'Speedup':<10}")
        print("-" * 70)
        
        for i, original in enumerate(results['original']):
            if i < len(results['optimized']):
                optimized = results['optimized'][i]
                model = original['model_name'].replace('.pt', '').upper()
                
                # Detection comparison
                det_speedup = (optimized['detection_fps'] / original['detection_fps'] 
                              if original['detection_fps'] > 0 else 0)
                print(f"{model:<15} {'Detection':<10} {original['detection_fps']:<12.1f} "
                      f"{optimized['detection_fps']:<12.1f} {det_speedup:<10.2f}x")
                
                # Tracking comparison  
                track_speedup = (optimized['tracking_fps'] / original['tracking_fps'] 
                               if original['tracking_fps'] > 0 else 0)
                print(f"{'':<15} {'Tracking':<10} {original['tracking_fps']:<12.1f} "
                      f"{optimized['tracking_fps']:<12.1f} {track_speedup:<10.2f}x")
                print()


def main():
    """Run the YOLO performance benchmark."""
    # Test video path
    test_video = "/Users/jbouguet/Documents/EufySecurityVideos/record/Batch010/T8600P1024260D5E_20241118084615.mp4"
    
    try:
        # Create benchmark instance
        benchmark = YoloBenchmark(test_video)
        
        # Run comprehensive benchmark
        results = benchmark.run_full_benchmark(num_frames=20)  # 20 frames for faster testing
        
        # Save and display results
        benchmark.save_results(results)
        benchmark.print_summary(results)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Test video not found: {e}")
        logger.error("Please update the test_video path in the script")
        return 1
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())