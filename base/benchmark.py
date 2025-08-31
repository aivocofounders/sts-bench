"""
Abstract benchmark class for speech-to-speech model evaluation.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from loguru import logger
from tqdm.asyncio import tqdm

from .config import BenchmarkConfig
from .metrics import BenchmarkMetrics
from .models import BaseModel

class AbstractBenchmark(ABC):
    """Abstract base class for all benchmarks."""

    def __init__(self, config: BenchmarkConfig, name: str):
        self.config = config
        self.name = name
        self.metrics = BenchmarkMetrics(name)
        self.models: Dict[str, BaseModel] = {}
        self.output_dir = Path(config.output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, level=self.config.log_level, rotation="10 MB")

    async def initialize_models(self) -> bool:
        """Initialize all required models."""
        try:
            # Initialize Vispark client
            from .models import VisparkClient
            vispark_client = VisparkClient(self.config)
            if await vispark_client.connect():
                self.models['vispark'] = vispark_client
                logger.info("Vispark client initialized successfully")
            else:
                logger.error("Failed to initialize Vispark client")
                return False

            # Initialize Aivoco client
            from .models import AivocoClient
            aivoco_client = AivocoClient(self.config)
            if await aivoco_client.connect():
                self.models['aivoco'] = aivoco_client
                logger.info("Aivoco client initialized successfully")
            else:
                logger.error("Failed to initialize Aivoco client")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False

    @abstractmethod
    async def run_benchmark(self) -> BenchmarkMetrics:
        """Run the benchmark and return metrics."""
        pass

    @abstractmethod
    async def setup(self) -> bool:
        """Setup benchmark-specific requirements."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup benchmark resources."""
        pass

    async def execute(self) -> BenchmarkMetrics:
        """Execute the complete benchmark workflow."""
        try:
            logger.info(f"Starting benchmark: {self.name}")

            # Validate configuration
            self.config.validate_keys()

            # Initialize models
            if not await self.initialize_models():
                raise RuntimeError("Failed to initialize models")

            # Setup benchmark
            if not await self.setup():
                raise RuntimeError("Failed to setup benchmark")

            # Run benchmark
            self.metrics = await self.run_benchmark()

            # Cleanup
            await self.cleanup()

            # Disconnect models
            await self._disconnect_models()

            # Finalize metrics
            self.metrics.finalize()

            # Save results
            await self._save_results()

            logger.info(f"Benchmark completed: {self.name}")
            return self.metrics

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            self.metrics.errors.append(str(e))
            self.metrics.finalize()
            raise

    async def _disconnect_models(self) -> None:
        """Disconnect all models."""
        for model_name, model in self.models.items():
            try:
                await model.disconnect()
                logger.info(f"Disconnected {model_name}")
            except Exception as e:
                logger.error(f"Error disconnecting {model_name}: {e}")

    async def _save_results(self) -> None:
        """Save benchmark results."""
        try:
            # Save detailed metrics
            metrics_file = self.output_dir / "metrics.json"
            self.metrics.save_to_json(metrics_file)

            # Save summary
            summary_file = self.output_dir / "summary.csv"
            self.metrics.save_summary_csv(summary_file)

            # Print summary
            self.metrics.print_summary()

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a model by name."""
        return self.models.get(name)

    def create_progress_bar(self, total: int, desc: str) -> tqdm:
        """Create a progress bar if enabled."""
        if self.config.enable_progress_bars:
            return tqdm(total=total, desc=desc, unit="items")
        else:
            # Return a dummy progress bar that does nothing
            class DummyProgressBar:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def update(self, n=1):
                    pass
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
            return DummyProgressBar()

    async def run_with_timeout(self, coro, timeout: float = 30.0):
        """Run a coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout} seconds")
            return None

    def log_benchmark_info(self, info: Dict[str, Any]) -> None:
        """Log benchmark information."""
        logger.info(f"Benchmark info: {json.dumps(info, indent=2)}")

    def validate_audio_response(self, response: Dict[str, Any]) -> bool:
        """Validate audio response format."""
        if not isinstance(response, dict):
            return False

        if response.get("status") != "success":
            return False

        if "data" not in response:
            return False

        return True

    def validate_text_response(self, response: Dict[str, Any]) -> bool:
        """Validate text response format."""
        if not isinstance(response, dict):
            return False

        if response.get("status") != "success":
            return False

        if "data" not in response or "content" not in response["data"]:
            return False

        return True
