"""
Boot Benchmark for Speech-to-Speech Models.

This benchmark measures the cold start performance by calling the model 100 times
in sequence and measuring the first response time for each call.
"""

import asyncio
import time
from typing import Dict, Any
from loguru import logger

from ..base.benchmark import AbstractBenchmark
from ..base.config import BenchmarkConfig
from ..base.audio import AudioProcessor

class BootBenchmark(AbstractBenchmark):
    """Boot performance benchmark for speech-to-speech models."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "boot")
        self.audio_processor = AudioProcessor()
        self.test_audio_base64 = None

    async def setup(self) -> bool:
        """Setup the boot benchmark."""
        try:
            logger.info("Setting up boot benchmark")

            # Generate test audio
            test_audio = self.audio_processor.generate_test_audio(duration_seconds=1.0)
            self.test_audio_base64 = self.audio_processor.audio_to_base64(test_audio)

            logger.info("Boot benchmark setup completed")
            return True

        except Exception as e:
            logger.error(f"Boot benchmark setup failed: {e}")
            return False

    async def run_benchmark(self) -> 'BenchmarkMetrics':
        """Run the boot benchmark."""
        logger.info(f"Starting boot benchmark with {self.config.boot_iterations} iterations")

        aivoco_client = self.get_model('aivoco')

        with self.create_progress_bar(self.config.boot_iterations, "Boot Test") as pbar:
            for i in range(self.config.boot_iterations):
                try:
                    iteration_start = time.time()

                    # Start a new call session
                    session_result = await aivoco_client.start_call()
                    if session_result.get("status") != "success":
                        self.metrics.record_request_end(iteration_start, success=False,
                                                      error=f"Session start failed: {session_result}")
                        pbar.update(1)
                        continue

                    # Record session start time
                    session_start_time = time.time()
                    first_response_time = None

                    # Send test audio and wait for first response
                    audio_result = await aivoco_client.send_audio_data(
                        self.test_audio_base64,
                        has_audio=True,
                        max_amplitude=1.0
                    )

                    if audio_result.get("status") == "success":
                        # Wait for first audio response
                        response = await self.run_with_timeout(
                            aivoco_client.receive_audio_response(),
                            timeout=10.0
                        )

                        if response:
                            first_response_time = time.time() - session_start_time
                            self.metrics.record_first_response(first_response_time)
                            self.metrics.record_request_end(iteration_start, success=True)
                            logger.debug(f"Iteration {i+1}: First response in {first_response_time:.3f}s")
                        else:
                            self.metrics.record_request_end(iteration_start, success=False,
                                                          error="No audio response received")
                    else:
                        self.metrics.record_request_end(iteration_start, success=False,
                                                      error=f"Audio send failed: {audio_result}")

                    # Stop the call
                    await aivoco_client.stop_call()

                    # Small delay between iterations to avoid overwhelming the service
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Boot benchmark iteration {i+1} failed: {e}")
                    self.metrics.record_request_end(time.time() if 'iteration_start' in locals() else time.time(),
                                                  success=False, error=str(e))

                pbar.update(1)

        logger.info("Boot benchmark completed")
        return self.metrics

    async def cleanup(self) -> None:
        """Cleanup boot benchmark resources."""
        logger.info("Cleaning up boot benchmark")
        # Cleanup is handled by parent class

async def main():
    """Main entry point for boot benchmark."""
    try:
        # Load configuration
        config = BenchmarkConfig.from_env()

        # Create and run benchmark
        benchmark = BootBenchmark(config)
        metrics = await benchmark.execute()

        logger.info(f"Boot benchmark completed. Average first response time: {metrics.average_first_response_time:.3f}s")

    except Exception as e:
        logger.error(f"Boot benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
