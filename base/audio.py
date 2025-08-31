"""
Audio processing utilities for speech-to-speech benchmarking.
"""

import base64
import io
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import soundfile as sf
import librosa
from scipy.io import wavfile
from loguru import logger

class AudioProcessor:
    """Handles audio processing for speech-to-speech models."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to required format."""
        try:
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            logger.debug(f"Loaded audio: {len(audio)} samples at {sr}Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise

    def save_audio(self, audio: np.ndarray, file_path: Union[str, Path], sample_rate: Optional[int] = None) -> None:
        """Save audio array to file."""
        try:
            sr = sample_rate or self.sample_rate
            sf.write(str(file_path), audio, sr)
            logger.debug(f"Saved audio to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save audio file {file_path}: {e}")
            raise

    def audio_to_base64(self, audio: np.ndarray) -> str:
        """Convert audio array to base64-encoded PCM."""
        try:
            # Convert to 16-bit PCM
            pcm_data = (audio * 32767).astype(np.int16)

            # Create WAV buffer
            buffer = io.BytesIO()
            wavfile.write(buffer, self.sample_rate, pcm_data)
            buffer.seek(0)

            # Encode to base64
            base64_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.debug(f"Encoded audio to base64: {len(base64_audio)} characters")
            return base64_audio
        except Exception as e:
            logger.error(f"Failed to encode audio to base64: {e}")
            raise

    def base64_to_audio(self, base64_audio: str) -> Tuple[np.ndarray, int]:
        """Convert base64-encoded PCM to audio array."""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_audio)

            # Read WAV from bytes
            buffer = io.BytesIO(audio_bytes)
            sr, pcm_data = wavfile.read(buffer)

            # Convert to float32
            audio = pcm_data.astype(np.float32) / 32768.0
            logger.debug(f"Decoded base64 to audio: {len(audio)} samples at {sr}Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {e}")
            raise

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            if orig_sr != target_sr:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
                logger.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
            return audio
        except Exception as e:
            logger.error(f"Failed to resample audio: {e}")
            raise

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        try:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            return audio
        except Exception as e:
            logger.error(f"Failed to normalize audio: {e}")
            raise

    def detect_audio_presence(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Detect if audio contains significant sound."""
        try:
            rms = np.sqrt(np.mean(audio ** 2))
            return rms > threshold
        except Exception as e:
            logger.error(f"Failed to detect audio presence: {e}")
            return False

    def calculate_amplitude(self, audio: np.ndarray) -> float:
        """Calculate maximum amplitude of audio."""
        try:
            return float(np.max(np.abs(audio)))
        except Exception as e:
            logger.error(f"Failed to calculate amplitude: {e}")
            return 0.0

    def generate_test_audio(self, duration_seconds: float = 1.0, frequency: float = 440.0) -> np.ndarray:
        """Generate test audio (sine wave)."""
        try:
            t = np.linspace(0, duration_seconds, int(self.sample_rate * duration_seconds), False)
            audio = np.sin(frequency * 2 * np.pi * t)
            return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate test audio: {e}")
            raise

    def get_audio_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        """Get duration of audio in seconds."""
        return len(audio) / sample_rate

    def trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        try:
            # Find non-silent segments
            mask = np.abs(audio) > threshold
            indices = np.where(mask)[0]

            if len(indices) == 0:
                return audio

            start_idx = indices[0]
            end_idx = indices[-1] + 1

            return audio[start_idx:end_idx]
        except Exception as e:
            logger.error(f"Failed to trim silence: {e}")
            return audio
