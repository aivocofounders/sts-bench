"""
Latency Benchmark for Speech-to-Speech Models.

This benchmark measures sustained performance by running 100 sessions of 30 minutes each,
simulating realistic conversation patterns with Vispark vision for scenario generation
and TTS/STT for natural interaction flow.
"""

import asyncio
import time
import random
from typing import Dict, Any, List
from loguru import logger

from ..base.benchmark import AbstractBenchmark
from ..base.config import BenchmarkConfig
from ..base.audio import AudioProcessor

class LatencyBenchmark(AbstractBenchmark):
    """Latency performance benchmark for speech-to-speech models."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "latency")
        self.audio_processor = AudioProcessor()
        self.conversation_scenarios = [
            "customer service call about a product issue",
            "medical consultation appointment booking",
            "technical support for software installation",
            "restaurant reservation and menu inquiry",
            "banking inquiry about account balance",
            "travel booking for vacation planning",
            "educational tutoring session",
            "job interview practice",
            "language learning conversation",
            "emergency services call simulation"
        ]

    async def setup(self) -> bool:
        """Setup the latency benchmark."""
        try:
            logger.info("Setting up latency benchmark")

            # Test all model connections
            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            # Test Vispark vision for scenario generation
            test_vision = await vispark_client.vision_analyze([
                {"type": "text", "content": "Generate a conversation scenario"}
            ])
            if test_vision.get("status") != "success":
                logger.error("Vispark vision test failed")
                return False

            # Test Vispark TTS
            test_tts = await vispark_client.text_to_speech("Test message")
            if test_tts.get("status") != "success":
                logger.error("Vispark TTS test failed")
                return False

            logger.info("Latency benchmark setup completed")
            return True

        except Exception as e:
            logger.error(f"Latency benchmark setup failed: {e}")
            return False

    async def run_single_session(self, session_id: int) -> Dict[str, Any]:
        """Run a single 30-minute session."""
        session_metrics = {
            "session_id": session_id,
            "start_time": time.time(),
            "message_count": 0,
            "response_latencies": [],
            "errors": [],
            "total_duration": 0
        }

        try:
            # Generate conversation scenario using Vispark Vision
            scenario = random.choice(self.conversation_scenarios)
            scenario_prompt = f"Generate a realistic conversation scenario for: {scenario}. Include system prompt and initial user message."

            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            scenario_result = await vispark_client.vision_analyze([
                {"type": "text", "content": scenario_prompt}
            ])

            if scenario_result.get("status") != "success":
                session_metrics["errors"].append("Failed to generate scenario")
                return session_metrics

            scenario_text = scenario_result["data"]["content"]
            system_prompt = f"You are role-playing in this scenario: {scenario_text}"
            initial_message = "Hello, I'd like to start our conversation."

            # Start Aivoco session
            session_result = await aivoco_client.start_call(
                system_message=system_prompt,
                voice_choice=self.config.voice_choice
            )

            if session_result.get("status") != "success":
                session_metrics["errors"].append("Failed to start session")
                return session_metrics

            # Convert initial message to audio and send
            tts_result = await vispark_client.text_to_speech(
                initial_message,
                voice=self.config.tts_voice,
                size=self.config.tts_size
            )

            if tts_result.get("status") != "success":
                session_metrics["errors"].append("TTS failed for initial message")
                await aivoco_client.stop_call()
                return session_metrics

            # Send initial audio
            message_start = time.time()
            audio_result = await aivoco_client.send_audio_data(
                tts_result["data"]["content"],
                has_audio=True,
                max_amplitude=1.0
            )

            if audio_result.get("status") == "success":
                # Wait for response
                response = await self.run_with_timeout(
                    aivoco_client.receive_audio_response(),
                    timeout=15.0
                )

                if response:
                    latency = time.time() - message_start
                    session_metrics["response_latencies"].append(latency)
                    session_metrics["message_count"] += 1

                    # Convert response back to text for next message
                    stt_result = await vispark_client.speech_to_text(
                        response["audio_data"]
                    )

                    if stt_result.get("status") == "success":
                        response_text = stt_result["data"]["content"]
                        # Generate follow-up message based on response
                        follow_up_prompt = f"Continue this conversation naturally. Previous response: {response_text}"
                        follow_up_result = await vispark_client.vision_analyze([
                            {"type": "text", "content": follow_up_prompt}
                        ])

                        if follow_up_result.get("status") == "success":
                            follow_up_text = follow_up_result["data"]["content"]
                            # Continue conversation...
                            session_metrics["message_count"] += 1

            # Continue conversation for remaining time
            session_end_time = session_metrics["start_time"] + (self.config.latency_duration_minutes * 60)

            while time.time() < session_end_time and len(session_metrics["errors"]) == 0:
                try:
                    # Generate next message using AI
                    next_message_prompt = f"Generate next natural message in this {scenario} conversation"
                    next_message_result = await vispark_client.vision_analyze([
                        {"type": "text", "content": next_message_prompt}
                    ])

                    if next_message_result.get("status") == "success":
                        next_message = next_message_result["data"]["content"]

                        # Convert to audio
                        tts_result = await vispark_client.text_to_speech(
                            next_message,
                            voice=self.config.tts_voice,
                            size=self.config.tts_size
                        )

                        if tts_result.get("status") == "success":
                            message_start = time.time()

                            # Send to speech-to-speech model
                            audio_result = await aivoco_client.send_audio_data(
                                tts_result["data"]["content"],
                                has_audio=True,
                                max_amplitude=1.0
                            )

                            if audio_result.get("status") == "success":
                                # Wait for response
                                response = await self.run_with_timeout(
                                    aivoco_client.receive_audio_response(),
                                    timeout=15.0
                                )

                                if response:
                                    latency = time.time() - message_start
                                    session_metrics["response_latencies"].append(latency)
                                    session_metrics["message_count"] += 1

                    # Small delay between messages
                    await asyncio.sleep(random.uniform(2, 5))

                except Exception as e:
                    session_metrics["errors"].append(f"Message exchange failed: {str(e)}")
                    break

            # Stop session
            await aivoco_client.stop_call()

        except Exception as e:
            session_metrics["errors"].append(f"Session failed: {str(e)}")

        session_metrics["total_duration"] = time.time() - session_metrics["start_time"]

        # Record latencies in main metrics
        for latency in session_metrics["response_latencies"]:
            self.metrics.record_latency(latency)

        return session_metrics

    async def run_benchmark(self) -> 'BenchmarkMetrics':
        """Run the latency benchmark."""
        logger.info(f"Starting latency benchmark with {self.config.latency_sessions} sessions")

        session_summaries = []

        with self.create_progress_bar(self.config.latency_sessions, "Latency Sessions") as pbar:
            for session_id in range(self.config.latency_sessions):
                try:
                    session_start = time.time()
                    session_result = await self.run_single_session(session_id)

                    session_summaries.append(session_result)

                    # Record session success/failure
                    success = len(session_result["errors"]) == 0
                    self.metrics.record_request_end(session_start, success=success,
                                                  error="; ".join(session_result["errors"]) if not success else None)

                    logger.info(f"Session {session_id + 1} completed: {session_result['message_count']} messages, "
                              f"{len(session_result['response_latencies'])} responses")

                except Exception as e:
                    logger.error(f"Session {session_id + 1} failed: {e}")
                    self.metrics.record_request_end(time.time(), success=False, error=str(e))

                pbar.update(1)

        # Store session summaries in custom metrics
        self.metrics.add_custom_metric("session_summaries", session_summaries)

        logger.info("Latency benchmark completed")
        return self.metrics

    async def cleanup(self) -> None:
        """Cleanup latency benchmark resources."""
        logger.info("Cleaning up latency benchmark")
        # Cleanup is handled by parent class

async def main():
    """Main entry point for latency benchmark."""
    try:
        # Load configuration
        config = BenchmarkConfig.from_env()

        # Create and run benchmark
        benchmark = LatencyBenchmark(config)
        metrics = await benchmark.execute()

        logger.info(f"Latency benchmark completed. Average latency: {metrics.average_latency:.3f}s")

    except Exception as e:
        logger.error(f"Latency benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
