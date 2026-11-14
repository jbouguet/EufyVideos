#!/usr/bin/env python3

"""
Audio Enhancement Module

This module provides an interface to DeepFilterNet, a deep-learning based
speech enhancement model, to isolate human voices and suppress background
noise (traffic, wind, HVAC, insects, etc.) in the audio tracks of security
videos.

DeepFilterNet is imported lazily so that pipelines which never request audio
enhancement do not pay the cost of loading torch/DeepFilterNet, and do not
require the dependency to be installed at all.

Example Usage:
    ```python
    from audio_enhancer import AudioEnhancer, AudioEnhancementConfig

    enhancer = AudioEnhancer(AudioEnhancementConfig(atten_lim_db=20.0))
    enhancer.enhance_video(
        input_video_file="input.mp4",
        output_video_file="input_enhanced.mp4",
    )
    ```
"""

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

from logging_config import create_logger

logger = create_logger(__name__)

# DeepFilterNet models operate natively on mono 48kHz audio.
DEEPFILTERNET_SAMPLE_RATE = 48000


def _patch_torchaudio_backend_compat() -> None:
    """
    DeepFilterNet 0.5.6's df.io module targets an older torchaudio API than the
    one required elsewhere in this project (torchaudio>=2.0.0):
      - it imports torchaudio.backend.common.AudioMetaData, a submodule removed
        from modern torchaudio (used there purely as a type hint), and
      - it calls torchaudio.info(), a top-level function also since removed.
    Patch both in so `df.enhance` imports and runs without pinning torchaudio to
    an old version. Safe to call repeatedly; a no-op once already patched or if
    an older torchaudio that still ships these APIs natively is installed.
    """
    import sys
    import types
    import wave

    import torchaudio

    if not hasattr(torchaudio, "backend"):
        common_module = types.ModuleType("torchaudio.backend.common")

        class AudioMetaData:  # Stub type, never instantiated by df.io's logic.
            pass

        common_module.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

        backend_module = types.ModuleType("torchaudio.backend")
        backend_module.common = common_module  # type: ignore[attr-defined]

        sys.modules["torchaudio.backend"] = backend_module
        sys.modules["torchaudio.backend.common"] = common_module
        torchaudio.backend = backend_module  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "info"):

        def _info_stub(filepath: str, **kwargs: Any):
            # df.io.load_audio only reads .sample_rate off the returned object,
            # and only for PCM wav files that this module itself produces.
            with wave.open(filepath, "rb") as f:
                return types.SimpleNamespace(
                    sample_rate=f.getframerate(),
                    num_frames=f.getnframes(),
                    num_channels=f.getnchannels(),
                    bits_per_sample=f.getsampwidth() * 8,
                    encoding="PCM_S",
                )

        torchaudio.info = _info_stub  # type: ignore[attr-defined]


@dataclass
class AudioEnhancementConfig:
    """
    Configuration for DeepFilterNet-based audio enhancement.

    Attributes:
        atten_lim_db: Maximum attenuation applied to the estimated noise, in dB.
            None (default) applies DeepFilterNet's full noise suppression.
            Lower values (e.g. 10-20) mix in more of the original signal, which
            can sound more natural at the cost of leaving more residual noise.
        post_filter: Enable DeepFilterNet's post-filter, which further reduces
            residual noise at a small cost to speech quality.

    Example:
        ```python
        config = AudioEnhancementConfig(atten_lim_db=20.0, post_filter=True)
        ```
    """

    atten_lim_db: Optional[float] = None
    post_filter: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AudioEnhancementConfig":
        """Creates an AudioEnhancementConfig instance from a dictionary configuration."""
        return cls(
            atten_lim_db=config_dict.get("atten_lim_db"),
            post_filter=config_dict.get("post_filter", False),
        )


class AudioEnhancer:
    """
    Isolates human speech and removes background noise from video audio
    tracks using DeepFilterNet.

    The DeepFilterNet model is loaded lazily on first use and cached on the
    instance for reuse across multiple calls.

    Example:
        ```python
        enhancer = AudioEnhancer()
        enhancer.enhance_video("doorbell_clip.mp4", "doorbell_clip_enhanced.mp4")
        ```
    """

    def __init__(self, config: Optional[AudioEnhancementConfig] = None) -> None:
        """Initialize enhancer with optional configuration."""
        self.config = AudioEnhancementConfig() if config is None else config
        self._model: Optional[Any] = None
        self._df_state: Optional[Any] = None

    def _load_model(self) -> None:
        """Lazily load and cache the DeepFilterNet model and its state."""
        if self._model is not None:
            return
        try:
            _patch_torchaudio_backend_compat()
            from df.enhance import init_df
        except ImportError as e:
            raise ImportError(
                "DeepFilterNet is required for audio enhancement but is not installed. "
                "Install it with 'pip install deepfilternet' (requires a Rust toolchain; "
                "see README for setup instructions)."
            ) from e

        logger.debug("Loading DeepFilterNet model...")
        self._model, self._df_state, _ = init_df(post_filter=self.config.post_filter)
        logger.debug("DeepFilterNet model loaded")

    def enhance_waveform(self, audio: Any) -> Any:
        """Run DeepFilterNet enhancement on an in-memory waveform tensor (mono, 48kHz)."""
        self._load_model()
        from df.enhance import enhance

        return enhance(
            self._model,
            self._df_state,
            audio,
            atten_lim_db=self.config.atten_lim_db,
        )

    def enhance_audio_file(self, input_audio_file: str, output_audio_file: str) -> str:
        """Enhance a standalone audio file (e.g. .wav) and write the result to disk."""
        self._load_model()
        assert self._df_state is not None
        import torch
        from df.enhance import load_audio, save_audio

        audio, _ = load_audio(input_audio_file, sr=self._df_state.sr())
        enhanced = self.enhance_waveform(audio)
        # df.io.save_audio defaults to quantizing to int16 before calling
        # torchaudio.save(). With this project's torchaudio version, saving an
        # int16 tensor produces a corrupted (near full-scale noise) file on
        # reload; saving as float32 avoids that broken code path.
        save_audio(
            output_audio_file, enhanced, self._df_state.sr(), dtype=torch.float32
        )
        return output_audio_file

    def enhance_video(
        self, input_video_file: str, output_video_file: str
    ) -> Optional[str]:
        """
        Produce a copy of input_video_file with its audio track passed through
        DeepFilterNet. The video stream is copied untouched; only the audio
        track is replaced with the enhanced version.

        Returns the output path, or None if the input has no audio stream.
        """
        if not os.path.exists(input_video_file):
            logger.error(f"Input video not found: {input_video_file}")
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_audio_file = os.path.join(tmp_dir, "raw_audio.wav")
            enhanced_audio_file = os.path.join(tmp_dir, "enhanced_audio.wav")

            if not self._extract_audio(input_video_file, raw_audio_file):
                logger.error(f"No audio stream found in {input_video_file}")
                return None

            self.enhance_audio_file(raw_audio_file, enhanced_audio_file)
            self._mux_video_with_audio(
                video_file=input_video_file,
                audio_file=enhanced_audio_file,
                output_file=output_video_file,
            )

        return output_video_file

    def _extract_audio(self, video_file: str, audio_file: str) -> bool:
        """Extract mono 48kHz PCM audio from a video file. Returns False if no audio stream exists."""
        import ffmpeg

        try:
            (
                ffmpeg.input(video_file)
                .output(
                    audio_file,
                    vn=None,
                    acodec="pcm_s16le",
                    ac=1,
                    ar=DEEPFILTERNET_SAMPLE_RATE,
                )
                .run(overwrite_output=True, capture_stderr=True, quiet=True)
            )
        except ffmpeg.Error as e:
            logger.error(
                f"Failed to extract audio from {video_file}: {e.stderr.decode()}"
            )
            return False
        return os.path.exists(audio_file)

    def _mux_video_with_audio(
        self, video_file: str, audio_file: str, output_file: str
    ) -> None:
        """Combine the original video stream with an enhanced audio track."""
        import ffmpeg

        video_stream = ffmpeg.input(video_file).video
        audio_stream = ffmpeg.input(audio_file).audio
        output = ffmpeg.output(
            video_stream,
            audio_stream,
            output_file,
            vcodec="copy",
            acodec="aac",
            audio_bitrate="128k",
            shortest=None,
            map_metadata=-1,
        )
        try:
            ffmpeg.run(stream_spec=output, overwrite_output=True, capture_stderr=True)
        except ffmpeg.Error as e:
            logger.error(
                f"Failed to mux enhanced audio into {output_file}: {e.stderr.decode()}"
            )
            raise


def measure_audio_levels(audio_file: str) -> Dict[str, float]:
    """
    Compute basic loudness statistics (RMS and peak, in dBFS) for an audio
    file. Useful for quantifying noise reduction before/after enhancement.
    """
    import torchaudio

    waveform, _ = torchaudio.load(audio_file)
    waveform = waveform.mean(dim=0)  # collapse channels to mono

    eps = 1e-12
    rms = waveform.pow(2).mean().sqrt().item()
    peak = waveform.abs().max().item()

    return {
        "rms_dbfs": 20 * math.log10(max(rms, eps)),
        "peak_dbfs": 20 * math.log10(max(peak, eps)),
    }


if __name__ == "__main__":
    # Testing code for the module.

    def run_example_1():
        logger.info(
            "EXAMPLE 1: Enhance the audio track of a single video file, isolating "
            "human speech and suppressing background noise."
        )
        input_video = (
            "/Users/GZ5MCM/Documents/EufySecurityVideos/record/Batch057/T8600P1024260D5E_20260717215348.mp4"
        )
        output_video = "/Users/GZ5MCM/Documents/EufySecurityVideos/stories/T8600P1024260D5E_20260717215348_enhanced.mp4"

        enhancer = AudioEnhancer(AudioEnhancementConfig(atten_lim_db=20.0))
        result = enhancer.enhance_video(
            input_video_file=input_video, output_video_file=output_video
        )
        logger.info(f"Enhanced video written to: {result}")

    def run_example_2():
        logger.info(
            "EXAMPLE 2: Enhance a batch of videos across a range of atten_lim_db "
            "values, from minimal to maximal noise suppression, to compare results. "
            "The DeepFilterNet model is loaded once and each video's audio is "
            "extracted once, then reused across atten_lim_db values, since "
            "neither depends on that parameter."
        )
        input_dir = "/Users/GZ5MCM/Documents/EufySecurityVideos/record/Batch057/"
        output_dir = "/Users/GZ5MCM/Documents/EufySecurityVideos/stories"
        input_videos = [
            "T8600P1023450AFB_20260717215240.mp4",
            "T8600P1024260D5E_20260717215348.mp4",
            "T8600P1023450AFB_20260717215451.mp4",
        ]
        atten_lim_db_values = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

        enhancer = AudioEnhancer(AudioEnhancementConfig())
        enhancer._load_model()

        with tempfile.TemporaryDirectory() as tmp_dir:
            for input_video_name in input_videos:
                input_video = os.path.join(input_dir, input_video_name)
                video_basename, _ = os.path.splitext(input_video_name)

                raw_audio_file = os.path.join(tmp_dir, f"{video_basename}_raw.wav")
                if not enhancer._extract_audio(input_video, raw_audio_file):
                    logger.error(f"No audio stream found in {input_video}")
                    continue

                for atten_lim_db in atten_lim_db_values:
                    enhancer.config.atten_lim_db = atten_lim_db
                    enhanced_audio_file = os.path.join(
                        tmp_dir, f"{video_basename}_atten{int(atten_lim_db)}db.wav"
                    )
                    enhancer.enhance_audio_file(raw_audio_file, enhanced_audio_file)

                    output_video = os.path.join(
                        output_dir,
                        f"{video_basename}_atten{int(atten_lim_db)}db.mp4",
                    )
                    enhancer._mux_video_with_audio(
                        video_file=input_video,
                        audio_file=enhanced_audio_file,
                        output_file=output_video,
                    )
                    logger.info(f"Enhanced video written to: {output_video}")

    run_example_2()
