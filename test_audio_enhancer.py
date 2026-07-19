#!/usr/bin/env python3
"""
Isolation test for the audio_enhancer module.

Synthesizes a self-contained test video (no external media required) whose
audio track contains real synthesized speech (via macOS `say`) layered over
broadband noise for its full duration. Runs it through AudioEnhancer and
checks that DeepFilterNet:
  - removes the video's audio track and replaces it with an enhanced one
  - leaves the video stream itself untouched (same duration/frame count)
  - substantially reduces the noise floor during the noise-only segment
  - preserves most of the speech energy during the speech segment

A synthetic tone burst is deliberately not used as a stand-in for speech:
DeepFilterNet is trained to recognize the spectral/temporal structure of
real speech, and correctly suppresses steady harmonic tones as noise, which
would make that substitution self-defeating for this test.

Usage:
    pytest test_audio_enhancer.py -v
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import ffmpeg

from audio_enhancer import AudioEnhancer, AudioEnhancementConfig, measure_audio_levels
from logging_config import create_logger
from video_metadata import VideoMetadata

logger = create_logger(__name__)

SAMPLE_RATE = 48000
VOICE_START = 1.0  # seconds
VOICE_DURATION = 2.0  # seconds
TOTAL_DURATION = 4.0  # seconds
SPEECH_TEXT = "Testing one two three, this is a security camera recording."


def _synthesize_speech_wav(output_wav: str) -> None:
    """Use macOS `say` to synthesize a short real speech sample."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        aiff_file = os.path.join(tmp_dir, "speech.aiff")
        subprocess.run(["say", "-o", aiff_file, SPEECH_TEXT], check=True)
        (
            ffmpeg.input(aiff_file)
            .output(output_wav, acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run(overwrite_output=True, capture_stderr=True, quiet=True)
        )


def _generate_test_video(output_file: str) -> None:
    """Create a short synthetic mp4 with real speech layered over broadband noise."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        speech_wav = os.path.join(tmp_dir, "speech.wav")
        _synthesize_speech_wav(speech_wav)

        video = ffmpeg.input(
            f"testsrc=duration={TOTAL_DURATION}:size=320x240:rate=15", f="lavfi"
        )
        noise = ffmpeg.input(
            f"anoisesrc=color=pink:amplitude=0.05:sample_rate={SAMPLE_RATE}", f="lavfi"
        ).audio
        speech = (
            ffmpeg.input(speech_wav)
            .audio.filter("atrim", start=0, end=VOICE_DURATION)
            .filter("adelay", f"{int(VOICE_START * 1000)}")
            .filter("apad")
            .filter("atrim", start=0, end=TOTAL_DURATION)
            .filter("volume", 3.0)
        )
        mixed_audio = ffmpeg.filter([noise, speech], "amix", inputs=2, weights="1 1")

        output = ffmpeg.output(
            video,
            mixed_audio,
            output_file,
            vcodec="libx264",
            acodec="pcm_s16le",
            ac=1,
            ar=SAMPLE_RATE,
            t=TOTAL_DURATION,
        )
        ffmpeg.run(output, overwrite_output=True, capture_stderr=True, quiet=True)


def _extract_wav(video_file: str, wav_file: str) -> None:
    (
        ffmpeg.input(video_file)
        .output(wav_file, vn=None, acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
        .run(overwrite_output=True, capture_stderr=True, quiet=True)
    )


def _trim_wav(input_wav: str, output_wav: str, start: float, duration: float) -> None:
    (
        ffmpeg.input(input_wav, ss=start, t=duration)
        .output(output_wav, acodec="pcm_s16le")
        .run(overwrite_output=True, capture_stderr=True, quiet=True)
    )


class TestAudioEnhancer(unittest.TestCase):
    """Validates DeepFilterNet-based enhancement end-to-end on a synthetic clip."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="audio_enhancer_test_")
        cls.input_video = os.path.join(cls.tmp_dir, "input.mp4")
        cls.output_video = os.path.join(cls.tmp_dir, "input_enhanced.mp4")

        _generate_test_video(cls.input_video)

        enhancer = AudioEnhancer(AudioEnhancementConfig(atten_lim_db=None))
        result = enhancer.enhance_video(
            input_video_file=cls.input_video, output_video_file=cls.output_video
        )
        assert result == cls.output_video

        # Extract full audio tracks for level comparisons.
        cls.input_wav = os.path.join(cls.tmp_dir, "input.wav")
        cls.output_wav = os.path.join(cls.tmp_dir, "output.wav")
        _extract_wav(cls.input_video, cls.input_wav)
        _extract_wav(cls.output_video, cls.output_wav)

        # Isolate the silent (noise-only) and voice segments in each.
        silence_start, silence_duration = VOICE_START + VOICE_DURATION + 0.2, 0.6
        cls.input_silence_wav = os.path.join(cls.tmp_dir, "input_silence.wav")
        cls.output_silence_wav = os.path.join(cls.tmp_dir, "output_silence.wav")
        _trim_wav(cls.input_wav, cls.input_silence_wav, silence_start, silence_duration)
        _trim_wav(
            cls.output_wav, cls.output_silence_wav, silence_start, silence_duration
        )

        voice_start, voice_duration = VOICE_START + 0.3, VOICE_DURATION - 0.6
        cls.input_voice_wav = os.path.join(cls.tmp_dir, "input_voice.wav")
        cls.output_voice_wav = os.path.join(cls.tmp_dir, "output_voice.wav")
        _trim_wav(cls.input_wav, cls.input_voice_wav, voice_start, voice_duration)
        _trim_wav(cls.output_wav, cls.output_voice_wav, voice_start, voice_duration)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_output_video_exists_with_audio(self):
        self.assertTrue(os.path.exists(self.output_video))
        probe = ffmpeg.probe(self.output_video)
        audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
        self.assertEqual(len(audio_streams), 1)

    def test_video_stream_untouched(self):
        input_meta = VideoMetadata(full_path=self.input_video)
        output_meta = VideoMetadata(full_path=self.output_video)
        self.assertEqual(input_meta.width, output_meta.width)
        self.assertEqual(input_meta.height, output_meta.height)
        self.assertAlmostEqual(
            input_meta.duration.total_seconds(),
            output_meta.duration.total_seconds(),
            delta=0.2,
        )

    def test_noise_floor_is_reduced(self):
        before = measure_audio_levels(self.input_silence_wav)
        after = measure_audio_levels(self.output_silence_wav)
        logger.info(f"Noise-only segment RMS before: {before['rms_dbfs']:.1f} dBFS")
        logger.info(f"Noise-only segment RMS after:  {after['rms_dbfs']:.1f} dBFS")

        reduction_db = before["rms_dbfs"] - after["rms_dbfs"]
        logger.info(f"Noise reduction: {reduction_db:.1f} dB")
        self.assertGreater(
            reduction_db,
            10.0,
            "Expected DeepFilterNet to reduce the noise-only segment level by "
            "at least 10 dB",
        )

    def test_voice_segment_energy_is_preserved(self):
        before = measure_audio_levels(self.input_voice_wav)
        after = measure_audio_levels(self.output_voice_wav)
        logger.info(f"Voice segment RMS before: {before['rms_dbfs']:.1f} dBFS")
        logger.info(f"Voice segment RMS after:  {after['rms_dbfs']:.1f} dBFS")

        drop_db = before["rms_dbfs"] - after["rms_dbfs"]
        logger.info(f"Voice segment change: {drop_db:.1f} dB")
        self.assertLess(
            drop_db,
            10.0,
            "Expected the voice-like tone burst to be largely preserved, not "
            "attenuated along with the noise",
        )


if __name__ == "__main__":
    unittest.main()
