"""
Base model classes for interacting with different APIs.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
import threading
import requests
from loguru import logger
import socketio

from .config import BenchmarkConfig
from .audio import AudioProcessor

class BaseModel(ABC):
    """Abstract base class for model implementations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.audio_processor = AudioProcessor()
        self._is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the model."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the model."""
        pass

    @abstractmethod
    async def send_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send a message to the model."""
        pass

    @abstractmethod
    async def send_audio(self, audio_data: str, **kwargs) -> Dict[str, Any]:
        """Send audio data to the model."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if model is connected."""
        return self._is_connected

class VisparkClient(BaseModel):
    """Client for interacting with Vispark models."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": config.api.vispark_api_key,
            "Content-Type": "application/json"
        })

    async def connect(self) -> bool:
        """Test connection to Vispark API."""
        try:
            # Test connection with a simple request
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.session.get, f"{self.config.api.vispark_base_url}/health"
            )
            self._is_connected = response.status_code == 200
            logger.info(f"Vispark connection test: {'SUCCESS' if self._is_connected else 'FAILED'}")
            return self._is_connected
        except Exception as e:
            logger.error(f"Failed to connect to Vispark: {e}")
            return False

    async def disconnect(self) -> None:
        """Close Vispark session."""
        self.session.close()
        self._is_connected = False
        logger.info("Disconnected from Vispark")

    async def vision_analyze(self, content: List[Dict[str, Any]], size: str = "small",
                           system_message: Optional[str] = None) -> Dict[str, Any]:
        """Analyze content using Vispark Vision model."""
        try:
            payload = {
                "size": size,
                "content": content
            }
            if system_message:
                payload["system_message"] = system_message

            response = await asyncio.get_event_loop().run_in_executor(
                None, self.session.post,
                f"{self.config.api.vispark_base_url}/model/text/vision",
                json.dumps(payload)
            )

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Vision analysis successful: {result.get('status', 'unknown')}")
                return result
            else:
                logger.error(f"Vision analysis failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {"status": "error", "message": str(e)}

    async def text_to_speech(self, text: str, voice: str = "girl", size: str = "small") -> Dict[str, Any]:
        """Convert text to speech using Vispark TTS."""
        try:
            payload = {
                "size": size,
                "text": text,
                "voice": voice
            }

            response = await asyncio.get_event_loop().run_in_executor(
                None, self.session.post,
                f"{self.config.api.vispark_base_url}/model/audio/text_to_speech",
                json.dumps(payload)
            )

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"TTS successful: {result.get('status', 'unknown')}")
                return result
            else:
                logger.error(f"TTS failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {"status": "error", "message": str(e)}

    async def speech_to_text(self, audio_base64: str) -> Dict[str, Any]:
        """Convert speech to text using Vispark STT."""
        try:
            payload = {
                "audio": audio_base64
            }

            response = await asyncio.get_event_loop().run_in_executor(
                None, self.session.post,
                f"{self.config.api.vispark_base_url}/model/audio/speech_to_text",
                json.dumps(payload)
            )

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"STT successful: {result.get('status', 'unknown')}")
                return result
            else:
                logger.error(f"STT failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"STT error: {e}")
            return {"status": "error", "message": str(e)}

    async def send_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send message using Vision model."""
        content = [{"type": "text", "content": message}]
        return await self.vision_analyze(content, **kwargs)

    async def send_audio(self, audio_data: str, **kwargs) -> Dict[str, Any]:
        """Send audio using STT model."""
        return await self.speech_to_text(audio_data)

class AivocoClient(BaseModel):
    """Client for interacting with Aivoco speech-to-speech model."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.sio = socketio.AsyncClient()
        self.audio_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.text_response_queue = asyncio.Queue()
        self.is_call_active = False
        self.call_start_time = None
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup Socket.IO event handlers."""

        @self.sio.on('connect')
        async def on_connect():
            logger.info("Connected to Aivoco WebSocket")
            self._is_connected = True

        @self.sio.on('disconnect')
        async def on_disconnect():
            logger.info("Disconnected from Aivoco WebSocket")
            self._is_connected = False

        @self.sio.on('auth_success')
        async def on_auth_success(data):
            logger.info(f"Aivoco authentication successful: {data.get('credits', 0)} credits")

        @self.sio.on('auth_failed')
        async def on_auth_failed(data):
            logger.error(f"Aivoco authentication failed: {data.get('message', 'Unknown error')}")

        @self.sio.on('session_ready')
        async def on_session_ready(data):
            logger.info("Aivoco session ready")
            self.is_call_active = True
            self.call_start_time = time.time()

        @self.sio.on('session_ended')
        async def on_session_ended(data):
            logger.info("Aivoco session ended")
            self.is_call_active = False
            self.call_start_time = None

        @self.sio.on('audio_response')
        async def on_audio_response(data):
            await self.audio_queue.put(data)

        @self.sio.on('text_response')
        async def on_text_response(data):
            await self.text_response_queue.put(data)

        @self.sio.on('error')
        async def on_error(data):
            logger.error(f"Aivoco error: {data}")
            await self.response_queue.put({"status": "error", "data": data})

    async def connect(self) -> bool:
        """Connect to Aivoco WebSocket."""
        try:
            await self.sio.connect(self.config.api.aivoco_endpoint)
            # Wait a bit for connection to establish
            await asyncio.sleep(1)
            return self._is_connected
        except Exception as e:
            logger.error(f"Failed to connect to Aivoco: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Aivoco WebSocket."""
        try:
            if self.is_call_active:
                await self.stop_call()
            await self.sio.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from Aivoco: {e}")

    async def start_call(self, system_message: Optional[str] = None,
                        voice_choice: Optional[str] = None) -> Dict[str, Any]:
        """Start a new call session."""
        try:
            auth_data = {
                "auth_key": self.config.api.aivoco_api_key,
                "system_message": system_message or self.config.system_message,
                "voice_choice": voice_choice or self.config.voice_choice,
                "custom_functions": []
            }

            await self.sio.emit('start_call', auth_data)

            # Wait for session_ready or error
            try:
                response = await asyncio.wait_for(self.response_queue.get(), timeout=10.0)
                return response
            except asyncio.TimeoutError:
                return {"status": "error", "message": "Timeout waiting for session start"}

        except Exception as e:
            logger.error(f"Failed to start Aivoco call: {e}")
            return {"status": "error", "message": str(e)}

    async def stop_call(self) -> Dict[str, Any]:
        """Stop the current call session."""
        try:
            await self.sio.emit('stop_call')
            self.is_call_active = False
            return {"status": "success", "message": "Call stopped"}
        except Exception as e:
            logger.error(f"Failed to stop Aivoco call: {e}")
            return {"status": "error", "message": str(e)}

    async def send_audio_data(self, audio_base64: str, has_audio: bool = True,
                            max_amplitude: float = 1.0) -> Dict[str, Any]:
        """Send audio data to Aivoco."""
        try:
            if not self.is_call_active:
                return {"status": "error", "message": "No active call session"}

            audio_data = {
                "audio_data": audio_base64,
                "has_audio": has_audio,
                "max_amplitude": max_amplitude
            }

            await self.sio.emit('audio_data', audio_data)
            return {"status": "success", "message": "Audio data sent"}

        except Exception as e:
            logger.error(f"Failed to send audio data: {e}")
            return {"status": "error", "message": str(e)}

    async def receive_audio_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Receive audio response from Aivoco."""
        try:
            return await asyncio.wait_for(self.audio_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def receive_text_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Receive text response from Aivoco."""
        try:
            return await asyncio.wait_for(self.text_response_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def send_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send text message by converting to audio first."""
        try:
            # Convert text to audio using Vispark TTS
            vispark_client = VisparkClient(self.config)
            tts_result = await vispark_client.text_to_speech(
                message,
                voice=self.config.tts_voice,
                size=self.config.tts_size
            )

            if tts_result.get("status") != "success":
                return {"status": "error", "message": "TTS failed"}

            # Send audio to Aivoco
            audio_response = await self.send_audio_data(tts_result["data"]["content"])

            # Wait for response
            response = await self.receive_audio_response()

            if response:
                return {"status": "success", "data": response}
            else:
                return {"status": "error", "message": "No response received"}

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return {"status": "error", "message": str(e)}

    async def send_audio(self, audio_data: str, **kwargs) -> Dict[str, Any]:
        """Send audio data directly."""
        return await self.send_audio_data(audio_data)
