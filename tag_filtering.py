"""
Tag Filtering Library

This module provides a set of tools to filter tags in videos.
Built on top of tag_processor.
"""

from collections import defaultdict
from typing import Any, Dict, List, Set

import numpy as np

from tag_processor import VideoTags

# tags are:
#  filename  -> frame -> hash -> tag
# Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]
#
# tracks are:
#  filename  -> track_id -> frame -> tag
# Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]


def filter_by_value(
    tags: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]], value: str
) -> Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]:
    tags_filt = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for filename, file_tags in tags.items():
        for frame_number, frame_tags in file_tags.items():
            for hash_key, tag in frame_tags.items():
                if tag["value"] == value:
                    new_tag = {k: v for k, v in tag.items()}
                    tags_filt[filename][frame_number][hash_key] = new_tag
    return dict(tags_filt)


def tags_to_tracks(
    tags: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]
) -> Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]:
    tracks = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for filename, file_tags in tags.items():
        for frame_number, frame_tags in file_tags.items():
            for hash_key, tag in frame_tags.items():
                if "track_id" in tag:
                    track_id = tag["track_id"]
                    new_tag = {k: v for k, v in tag.items() if k != "track_id"}
                    new_tag["hash"] = hash_key
                    tracks[filename][track_id][frame_number] = new_tag
    return dict(tracks)


def tracks_to_tags(
    tracks: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]
) -> Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]:
    tags = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for filename, file_tracks in tracks.items():
        for track_id, track_frames in file_tracks.items():
            for frame_number, tag in track_frames.items():
                hash_key = tag["hash"]
                new_tag = {k: v for k, v in tag.items() if k != "hash"}
                new_tag["track_id"] = track_id
                tags[filename][frame_number][hash_key] = new_tag
    return dict(tags)


def track_stats(
    tracks: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]
) -> Dict[str, int]:
    return {
        "num_tags": sum(
            sum(len(tracks[filename][track_id]) for track_id in tracks[filename].keys())
            for filename in tracks.keys()
        ),
        "num_tracks": sum(len(tracks[filename]) for filename in tracks.keys()),
        "num_videos": len(tracks),
    }


def tags_stats(tags: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]) -> Dict[str, int]:
    return {
        "num_tags": sum(
            sum(len(tags) for tags in frames.values()) for frames in tags.values()
        ),
        "num_frames": sum(len(frames) for frames in tags.values()),
        "num_videos": len(tags),
    }


def compute_iou(box1: dict, box2: dict) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Boxes are dictionaries with keys 'x1', 'y1', 'x2', 'y2'.
    Returns a value between 0 (no overlap) and 1 (perfect overlap).
    """
    # Calculate intersection coordinates
    x_left = max(box1["x1"], box2["x1"])
    y_top = max(box1["y1"], box2["y1"])
    x_right = min(box1["x2"], box2["x2"])
    y_bottom = min(box1["y2"], box2["y2"])

    # If there's no intersection, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def extract_clusters(similarity_matrix: np.ndarray) -> List[Set[int]]:
    """
    Extract clusters from a binary similarity matrix using depth-first search.

    Args:
        similarity_matrix: NxN numpy array where entry (i,j) is 1 if elements i and j
                         are in the same cluster, 0 otherwise.

    Returns:
        List of sets, where each set contains the indices of elements in a cluster.
    """
    N = similarity_matrix.shape[0]
    if N != similarity_matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Keep track of which elements we've already assigned to clusters
    visited = set()
    clusters = []

    def dfs(node: int, current_cluster: Set[int]):
        """Depth-first search to find all connected elements."""
        visited.add(node)
        current_cluster.add(node)

        # Check all possible connections
        for neighbor in range(N):
            if similarity_matrix[node, neighbor] == 1 and neighbor not in visited:
                dfs(neighbor, current_cluster)

    # Find clusters starting from each unvisited node
    for node in range(N):
        if node not in visited:
            current_cluster = set()
            dfs(node, current_cluster)
            clusters.append(current_cluster)

    return clusters


def remove_duplicates(video_tags: VideoTags, iou_thresh: float = 0.5) -> VideoTags:
    tracks = tags_to_tracks(video_tags.tags)
    tags_new = {}

    for filename in video_tags.tags.keys():
        if filename not in tags_new:
            tags_new[filename] = {}
        for frame_number in video_tags.tags[filename].keys():
            if frame_number not in tags_new[filename]:
                tags_new[filename][frame_number] = {}

            # Collect all the tags in the frame:
            all_tags_in_frame = []
            for hash_key, tag in video_tags.tags[filename][frame_number].items():
                track_id = tag["track_id"]
                track_len = len(tracks[filename][track_id])
                tag_info = {
                    "hash": hash_key,
                    "bbox": tag["bounding_box"],
                    "track_len": track_len,
                    "value": tag["value"],
                }
                all_tags_in_frame.append(tag_info)

            # Build a similarity matrix between all pairs of tags in the frame.
            sim_matrix = np.zeros((len(all_tags_in_frame), len(all_tags_in_frame)))
            for i, tag1 in enumerate(all_tags_in_frame):
                box1 = tag1["bbox"]
                value1 = tag1["value"]
                for j, tag2 in enumerate(all_tags_in_frame):
                    box2 = tag2["bbox"]
                    value2 = tag2["value"]
                    sim_matrix[i][j] = (
                        1
                        if (value1 == value2) and (compute_iou(box1, box2) > iou_thresh)
                        else 0
                    )

            # Extract clusters of near-duplicates and the associates track length.
            tag_clusters = extract_clusters(sim_matrix)
            track_length_clusters = []
            for cluster in tag_clusters:
                track_lengths = []
                for element_id in cluster:
                    tag_info = all_tags_in_frame[element_id]
                    track_length = tag_info["track_len"]
                    track_lengths.append(tag_info["track_len"])
                track_length_clusters.append(track_lengths)

            # Compute the max track len per cluster.
            max_track_length_per_cluster = []
            for track_lengths in track_length_clusters:
                max_track_length = 0
                for track_length in track_lengths:
                    if track_length > max_track_length:
                        max_track_length = track_length
                max_track_length_per_cluster.append(max_track_length)

            # Keep the one representative tag per cluster, the one with the largest track length.
            indices_to_keep = []
            for i, cluster in enumerate(tag_clusters):
                max_track_length = max_track_length_per_cluster[i]
                for element_id in cluster:
                    tag_info = all_tags_in_frame[element_id]
                    track_length = tag_info["track_len"]
                    if track_length == max_track_length:
                        indices_to_keep.append(element_id)

            # Add the tags to keep in the output tags collection.
            for index in indices_to_keep:
                hash_key = all_tags_in_frame[index]["hash"]
                tag = video_tags.tags[filename][frame_number][hash_key]
                new_tag = {k: v for k, v in tag.items()}
                tags_new[filename][frame_number][hash_key] = new_tag

    return VideoTags.from_tags(tags_new, video_tags.tagger_config)


if __name__ == "__main__":

    # Testing code for the module.
    import os
    import sys
    from typing import List

    from logging_config import setup_logger
    from tag_visualizer import TagVisualizer, TagVisualizerConfig
    from video_metadata import VideoMetadata

    logger = setup_logger(__name__)

    story_name: str = "track_testing"
    video_files: List[str] = [
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch010/T8600P1024260D5E_20241118084615.mp4",
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch010/T8600P1024260D5E_20241118084819.mp4",
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch010/T8600P1024260D5E_20241118084902.mp4",
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch010/T8600P1024260D5E_20241118085102.mp4",
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/Batch010/T8600P1024260D5E_20241118085306.mp4",
    ]
    out_dir: str = "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/stories"
    videos = VideoMetadata.clean_and_sort(
        [VideoMetadata.from_video_file(file) for file in video_files]
    )

    # Load the tag database
    tags_directory: str = (
        "/Users/jeanyves.bouguet/Documents/EufySecurityVideos/EufyVideos/record/tags_database"
    )
    tags_files = [
        os.path.join(tags_directory, file)
        for file in os.listdir(tags_directory)
        if file.lower().endswith((".json", ".tags"))
    ]
    video_tags_database: VideoTags = VideoTags.from_tags(tags={})
    for tags_file in tags_files:
        logger.info(f"Loading tags file {tags_file}")
        video_tags_database.merge(VideoTags.from_file(tags_file))

    # Subselect tags that are "person"
    # video_tags_database.tags = filter_by_value(video_tags_database.tags, "person")

    logger.info(f"Tags Databse: {video_tags_database.stats}")

    # Export tags to Videos to keep onlt the relevant tags present in the videos
    video_tags = VideoTags.from_videos(video_tags_database.to_videos(videos))

    logger.info(f"Video Tags: {video_tags.stats}")

    # Remove near duplicated of tags.
    iou_thresh: float = 0.5
    video_tags_new = remove_duplicates(video_tags, iou_thresh)

    tracks_new = tags_to_tracks(video_tags_new.tags)
    logger.info(f"Post near-duplicates removal: tags_new: {video_tags_new.stats}")
    logger.info(f"Post near-duplicates removal: tracks_new: {track_stats(tracks_new)}")

    videos_new = VideoMetadata.clean_and_sort(
        [VideoMetadata.from_video_file(file) for file in video_files]
    )
    video_tags_new.to_videos(videos_new)

    show_tags_video: bool = False
    if show_tags_video:
        # Quick visualization of the tags in the videos.
        tag_video_file = os.path.join(out_dir, f"{story_name}_tags.mp4")
        logger.info(f"Generating video tag file {tag_video_file}")
        TagVisualizer(
            TagVisualizerConfig(output_size={"width": 1600, "height": 900})
        ).run(videos, tag_video_file)

    show_tags_video_new: bool = True
    if show_tags_video_new:
        # Quick visualization of the tags in the videos.
        tag_video_file = os.path.join(out_dir, f"{story_name}_deduped_0.5_tags.mp4")
        logger.info(f"Generating video tag file {tag_video_file}")
        TagVisualizer(
            TagVisualizerConfig(output_size={"width": 1600, "height": 900})
        ).run(videos_new, tag_video_file)

    sys.exit()
