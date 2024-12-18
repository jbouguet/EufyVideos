"""
Tag Filtering Library

This module provides a set of tools to filter tags in videos.
Built on top of tag_processor.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tag_processor import VideoTags


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

    # Restrict to person tags:
    # video_tags_database.tags = filter_by_value(video_tags_database.tags, "person")

    logger.info(f"Tags Database (pre de-dup)     : {video_tags_database.stats}")

    # Remove duplicate tags in the entire database:
    video_tags_database.remove_duplicates()

    logger.info(f"Tags Database (post de-dup 1)  : {video_tags_database.stats}")

    # Export and import tags to and from videos of interest to retain relevant tags present in the videos
    video_tags = VideoTags.from_videos(
        video_tags_database.to_videos(
            VideoMetadata.clean_and_sort(
                [VideoMetadata.from_video_file(file) for file in video_files]
            )
        )
    )

    logger.info(f"Tags in videos: {video_tags.stats}")

    tracks = tags_to_tracks(video_tags.tags)
    num_tracks = sum(len(tracks[filename]) for filename in tracks.keys())

    track_renames: Dict[int, int] = {}
    iou_thresh: float = 0.99

    for filename in video_tags.tags.keys():
        tracks_in_file = tracks[filename]
        tags_in_file = video_tags.tags[filename]
        frame_numbers = sorted(video_tags.tags[filename].keys())
        for current_frame, next_frame in zip(frame_numbers, frame_numbers[1:]):
            tags_current = tags_in_file[current_frame]
            tags_next = tags_in_file[next_frame]

            # Identify tracks in current_frame that are not present in next_frame
            track_ids_current = set()
            track_ids_next = set()
            for hash_key in tags_current:
                track_ids_current.add(tags_current[hash_key]["track_id"])
            for hash_key in tags_next:
                track_ids_next.add(tags_next[hash_key]["track_id"])

            tracks_lost = track_ids_current - track_ids_next
            tracks_new = track_ids_next - track_ids_current

            # Establish pairings between tracks_lost and tracks_new from an iou POV:
            track_pairings: List[Tuple[int, int]] = []
            for track_lost in tracks_lost:
                sim_list = []
                for track_new in tracks_new:
                    tag_lost = tracks_in_file[track_lost][current_frame]
                    tag_new = tracks_in_file[track_new][next_frame]
                    sim = 0
                    if VideoTags.compute_iou(tag_lost, tag_new) > iou_thresh:
                        sim = 1
                        track_pairings.append((track_lost, track_new))
                    sim_list.append(sim)

            # Commmit and publish a track_id unification for all elected pairs
            for track_pairing in track_pairings:
                track_to_rename = max(track_pairing)
                track_name_to_keep = min(track_pairing)
                if track_to_rename not in track_renames:
                    track_renames[track_to_rename] = track_name_to_keep

    # Collapse all of the track_id renaming into distinct clusters
    keep_collapsing_renaming_chains: bool = True
    while keep_collapsing_renaming_chains:
        logger.info("Collapsing the chains of merges...")
        keep_collapsing_renaming_chains = False
        for track_id in track_renames.keys():
            if track_renames[track_id] in track_renames:
                if track_renames[track_id] != track_renames[track_renames[track_id]]:
                    logger.info(
                        f"Closing chain: {track_id} -> {track_renames[track_id]} -> {track_renames[track_renames[track_id]]}"
                    )
                    track_renames[track_id] = track_renames[track_renames[track_id]]
                    keep_collapsing_renaming_chains = True

    logger.info(f"Number of tracks to be renamed: {len(track_renames.keys())}")

    # WIP: Experimentally, not all renames shoudl be committed. In pratice,
    # When long tracks are committed to merge together, bad things happen.
    # Is it best to take an approcah of voting. When many votes call for 2 tracks to merge
    # a merge should be called. Otherwise, no.

    if True:
        # Execute track collapsing:
        for filename, file_tags in video_tags.tags.items():
            for frame_number, frame_tags in file_tags.items():
                for hash_key, tag in frame_tags.items():
                    if tag["track_id"] in track_renames:
                        tag["track_id"] = track_renames[tag["track_id"]]

    tracks_collapsed_tracks = tags_to_tracks(video_tags.tags)
    num_tracks_collapsed = sum(
        len(tracks_collapsed_tracks[filename])
        for filename in tracks_collapsed_tracks.keys()
    )

    print(f"Original number of tracks: {num_tracks}")
    print(f"Number of tracks after collapse: {num_tracks_collapsed}")

    show_tags_video: bool = True
    if show_tags_video:
        tag_video_file = os.path.join(
            out_dir, f"{story_name}_deduped_tracks_collapsed_tags5.mp4"
        )
        logger.info(f"Generating video tag file {tag_video_file}")
        TagVisualizer(
            TagVisualizerConfig(output_size={"width": 1600, "height": 900})
        ).run(
            video_tags.to_videos(
                VideoMetadata.clean_and_sort(
                    [VideoMetadata.from_video_file(file) for file in video_files]
                )
            ),
            tag_video_file,
        )

    sys.exit()
