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


def remove_duplicates(video_tags: VideoTags, iou_thresh: float = 0.5) -> VideoTags:
    tags_new = defaultdict(lambda: defaultdict(dict))
    tracks = tags_to_tracks(video_tags.tags)

    for filename, file_tags in video_tags.tags.items():
        for frame_number, frame_tags in file_tags.items():
            # Group tags by value
            value_groups = defaultdict(list)
            for hash_key, tag in frame_tags.items():
                value_groups[tag["value"]].append((hash_key, tag))

            # Process each value group
            for group in value_groups.values():
                while group:
                    hash_key, tag = group.pop(0)
                    track_len = len(tracks[filename][tag["track_id"]])
                    best_tag = (hash_key, tag, track_len)

                    # Compare with remaining tags in the group
                    i = 0
                    while i < len(group):
                        other_hash, other_tag = group[i]
                        if (
                            compute_iou(tag["bounding_box"], other_tag["bounding_box"])
                            > iou_thresh
                        ):
                            other_track_len = len(
                                tracks[filename][other_tag["track_id"]]
                            )
                            if other_track_len > best_tag[2]:
                                best_tag = (other_hash, other_tag, other_track_len)
                            group.pop(i)
                        else:
                            i += 1

                    # Add the best tag to the new tags
                    tags_new[filename][frame_number][best_tag[0]] = best_tag[1]

    return VideoTags.from_tags(dict(tags_new), video_tags.tagger_config)


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

    logger.info(f"Tags Databse (pre de-dup)     : {video_tags_database.stats}")

    # First pass at removing duplicates in the entire database:
    video_tags_database = remove_duplicates(video_tags_database)

    logger.info(f"Tags Databse (post de-dup 1)  : {video_tags_database.stats}")

    video_tags_database = remove_duplicates(video_tags_database)

    logger.info(f"Tags Databse (post de-dup 2)  : {video_tags_database.stats}")

    video_tags_database = remove_duplicates(video_tags_database)

    logger.info(f"Tags Databse (post de-dup 3)  : {video_tags_database.stats}")

    # Export tags to Videos to keep onlt the relevant tags present in the videos
    video_tags = VideoTags.from_videos(video_tags_database.to_videos(videos))

    logger.info(f"Tags in videos                : {video_tags.stats}")

    video_tags = remove_duplicates(video_tags)

    logger.info(f"Tags in videos (post de-dup 1): {video_tags.stats}")

    video_tags = remove_duplicates(video_tags)

    logger.info(f"Tags in videos (post de-dup 2): {video_tags.stats}")

    videos_new = VideoMetadata.clean_and_sort(
        [VideoMetadata.from_video_file(file) for file in video_files]
    )
    video_tags.to_videos(videos_new)

    show_tags_video: bool = False
    if show_tags_video:
        # Quick visualization of the tags in the videos.
        tag_video_file = os.path.join(out_dir, f"{story_name}_deduped_0.5_2_tags.mp4")
        logger.info(f"Generating video tag file {tag_video_file}")
        TagVisualizer(
            TagVisualizerConfig(output_size={"width": 1600, "height": 900})
        ).run(videos_new, tag_video_file)

    sys.exit()
