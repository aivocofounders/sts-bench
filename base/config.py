"""
Configuration management for benchmarks.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class APIConfig(BaseModel):
    """Configuration for API keys and endpoints."""

    aivoco_api_key: str = Field(default_factory=lambda: os.getenv('AIVOCO_API_KEY', ''))
    vispark_api_key: str = Field(default_factory=lambda: os.getenv('VISPARK_API_KEY', ''))

    aivoco_endpoint: str = "wss://sts.aivoco.on.cloud.vispark.in"
    vispark_base_url: str = "https://api.lab.vispark.in"

    class Config:
        validate_assignment = True

class AudioConfig(BaseModel):
    """Configuration for audio processing."""

    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    max_audio_size_mb: int = 25
    supported_formats: list = ["wav", "mp3", "flac", "m4a"]

class BenchmarkConfig(BaseModel):
    """Main configuration for benchmarks."""

    api: APIConfig = APIConfig()
    audio: AudioConfig = AudioConfig()

    # Benchmark-specific configurations
    boot_iterations: int = 100
    latency_sessions: int = 100
    latency_duration_minutes: int = 30
    multimodal_iterations: int = 100

    # Output configuration
    output_dir: str = "results"
    log_level: str = "INFO"
    enable_progress_bars: bool = True

    # Model configurations
    system_message: str = "You are a helpful AI assistant for testing purposes."
    voice_choice: str = "female"
    tts_voice: str = "girl"
    tts_size: str = "large"

    class Config:
        validate_assignment = True

    @classmethod
    def from_env(cls) -> 'BenchmarkConfig':
        """Create configuration from environment variables."""
        return cls()

    def validate_keys(self) -> bool:
        """Validate that all required API keys are present."""
        if not self.api.aivoco_api_key:
            raise ValueError("AIVOCO_API_KEY environment variable is required")
        if not self.api.vispark_api_key:
            raise ValueError("VISPARK_API_KEY environment variable is required")
        return True
