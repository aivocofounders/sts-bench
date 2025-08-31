"""
Expressiveness Benchmark for Speech-to-Speech Models.

This benchmark evaluates the expressiveness and vocal variety of speech generation
using Vispark Vision to judge prosody, intonation, and expressive elements.
"""

import asyncio
import json
from typing import Dict, Any, List
from loguru import logger

from ..base.benchmark import AbstractBenchmark
from ..base.config import BenchmarkConfig
from ..base.audio import AudioProcessor

class ExpressivenessBenchmark(AbstractBenchmark):
    """Expressiveness benchmark with multimodal AI judging."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "expressiveness")
        self.audio_processor = AudioProcessor()

        self.expressive_scenarios = [
            "excited announcement of good news",
            "dramatic storytelling with suspense",
            "persuasive sales pitch",
            "comforting reassurance during distress",
            "enthusiastic teaching explanation",
            "poetic recitation with emotion",
            "humorous anecdote delivery",
            "urgent warning or alert"
        ]

    async def setup(self) -> bool:
        """Setup the expressiveness benchmark."""
        try:
            logger.info("Setting up expressiveness benchmark")
            vispark_client = self.get_model('vispark')
            test_result = await vispark_client.vision_analyze([
                {"type": "text", "content": "Test expressiveness analysis"}
            ])
            return test_result.get("status") == "success"
        except Exception as e:
            logger.error(f"Expressiveness benchmark setup failed: {e}")
            return False

    async def run_expressive_interaction(self, scenario: str, interaction_id: int) -> Dict[str, Any]:
        """Run a single expressive interaction."""
        try:
            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            prompt = f"Deliver this message with full expressiveness: {scenario}"

            # Get expressive response
            tts_result = await vispark_client.text_to_speech(prompt, voice=self.config.tts_voice, size=self.config.tts_size)
            if tts_result.get("status") != "success":
                return {"success": False, "error": "TTS failed"}

            session_result = await aivoco_client.start_call(
                system_message=f"You are an expressive speaker. Deliver your response with rich vocal variety, appropriate intonation, and emotional depth for: {scenario}"
            )

            if session_result.get("status") != "success":
                return {"success": False, "error": "Session failed"}

            audio_result = await aivoco_client.send_audio_data(tts_result["data"]["content"])
            if audio_result.get("status") != "success":
                await aivoco_client.stop_call()
                return {"success": False, "error": "Audio send failed"}

            response = await self.run_with_timeout(aivoco_client.receive_audio_response(), timeout=25.0)
            if not response:
                await aivoco_client.stop_call()
                return {"success": False, "error": "No response"}

            await aivoco_client.stop_call()

            # Analyze expressiveness
            expressiveness_prompt = """
            Analyze the expressiveness of this speech on a scale of 1-10:

            1. Prosody: Rhythm and stress patterns
            2. Intonation: Pitch variation and melody
            3. Pace: Speed and timing variations
            4. Emphasis: Stress on important words/phrases
            5. Emotional range: Vocal emotional expression

            Provide an overall expressiveness score (0-100%).
            """

            analysis_result = await vispark_client.vision_analyze([
                {"type": "text", "content": expressiveness_prompt},
                {"type": "audio", "content": response["audio_data"]}
            ])

            expressiveness_score = 50.0
            if analysis_result.get("status") == "success":
                expressiveness_score = self.parse_expressiveness_score(analysis_result["data"]["content"])

            self.metrics.record_audio_quality(expressiveness_score)

            return {
                "interaction_id": interaction_id,
                "scenario": scenario,
                "success": True,
                "expressiveness_score": expressiveness_score,
                "analysis": analysis_result["data"]["content"] if analysis_result.get("status") == "success" else ""
            }

        except Exception as e:
            logger.error(f"Expressiveness interaction failed: {e}")
            return {"success": False, "error": str(e)}

    def parse_expressiveness_score(self, analysis_text: str) -> float:
        """Parse expressiveness score from analysis."""
        import re
        percent_match = re.search(r'(\d+(?:\.\d+)?)%', analysis_text)
        if percent_match:
            return float(percent_match.group(1))
        return 50.0

    async def run_benchmark(self) -> 'BenchmarkMetrics':
        """Run the expressiveness benchmark."""
        logger.info(f"Starting expressiveness benchmark")

        results = []

        for scenario in self.expressive_scenarios:
            with self.create_progress_bar(self.config.multimodal_iterations, f"Expressiveness - {scenario[:30]}...") as pbar:
                for i in range(self.config.multimodal_iterations):
                    result = await self.run_expressive_interaction(scenario, len(results))
                    results.append(result)
                    pbar.update(1)

        # Calculate metrics
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            avg_expressiveness = sum(r["expressiveness_score"] for r in successful_results) / len(successful_results)
            self.metrics.add_custom_metric("avg_expressiveness_score", avg_expressiveness)

        logger.info("Expressiveness benchmark completed")
        return self.metrics

    async def cleanup(self) -> None:
        """Cleanup expressiveness benchmark resources."""
        logger.info("Cleaning up expressiveness benchmark")

async def main():
    """Main entry point for expressiveness benchmark."""
    try:
        config = BenchmarkConfig.from_env()
        benchmark = ExpressivenessBenchmark(config)
        metrics = await benchmark.execute()
        logger.info(f"Expressiveness benchmark completed. Average expressiveness: {metrics.average_audio_quality:.2f}%")
    except Exception as e:
        logger.error(f"Expressiveness benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
