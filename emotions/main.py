"""
Emotions Benchmark for Speech-to-Speech Models.

This benchmark evaluates the model's ability to convey and recognize emotions
through speech by having conversations with different emotional contexts,
then using Vispark Vision to judge the emotional authenticity of the responses.
"""

import asyncio
import base64
import json
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger

from ..base.benchmark import AbstractBenchmark
from ..base.config import BenchmarkConfig
from ..base.audio import AudioProcessor

class EmotionsBenchmark(AbstractBenchmark):
    """Emotions benchmark with multimodal AI judging."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "emotions")
        self.audio_processor = AudioProcessor()

        # Define emotional scenarios
        self.emotional_scenarios = [
            {
                "emotion": "joy",
                "scenario": "celebrating a promotion at work",
                "prompts": [
                    "Tell me about your best day ever!",
                    "Share a happy memory from your childhood",
                    "What's something that always makes you smile?"
                ]
            },
            {
                "emotion": "sadness",
                "scenario": "dealing with a personal loss",
                "prompts": [
                    "Tell me about a time when you felt really sad",
                    "How do you cope when you're feeling down?",
                    "What's the hardest thing you've ever gone through?"
                ]
            },
            {
                "emotion": "anger",
                "scenario": "facing an unfair situation",
                "prompts": [
                    "Tell me about a time when you were really angry",
                    "How do you handle it when someone treats you unfairly?",
                    "What's something that makes you furious?"
                ]
            },
            {
                "emotion": "fear",
                "scenario": "facing a frightening situation",
                "prompts": [
                    "Tell me about your biggest fear",
                    "How do you feel when you're really scared?",
                    "What's the most terrifying experience you've had?"
                ]
            },
            {
                "emotion": "surprise",
                "scenario": "experiencing something unexpected",
                "prompts": [
                    "Tell me about a time when you were completely surprised",
                    "What's the most unexpected thing that's happened to you?",
                    "How do you react when something shocks you?"
                ]
            },
            {
                "emotion": "disgust",
                "scenario": "encountering something unpleasant",
                "prompts": [
                    "Tell me about something that really disgusted you",
                    "What's something you find absolutely revolting?",
                    "How do you react when you encounter something gross?"
                ]
            },
            {
                "emotion": "trust",
                "scenario": "building trust in a relationship",
                "prompts": [
                    "Tell me about someone you really trust",
                    "How do you know when you can trust someone?",
                    "What's the most trustworthy person you've met?"
                ]
            },
            {
                "emotion": "anticipation",
                "scenario": "looking forward to something exciting",
                "prompts": [
                    "Tell me about something you're really excited about",
                    "What's coming up that you're looking forward to?",
                    "How do you feel when you're anticipating something good?"
                ]
            }
        ]

    async def setup(self) -> bool:
        """Setup the emotions benchmark."""
        try:
            logger.info("Setting up emotions benchmark")

            # Test all model connections
            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            # Test Vispark Vision for emotion analysis
            test_emotion_prompt = "Analyze the emotional content of this text: 'I'm so happy today!'"
            test_result = await vispark_client.vision_analyze([
                {"type": "text", "content": test_emotion_prompt}
            ])
            if test_result.get("status") != "success":
                logger.error("Vispark Vision emotion test failed")
                return False

            logger.info("Emotions benchmark setup completed")
            return True

        except Exception as e:
            logger.error(f"Emotions benchmark setup failed: {e}")
            return False

    async def run_emotional_interaction(self, emotion_data: Dict[str, Any], interaction_id: int) -> Dict[str, Any]:
        """Run a single emotional interaction."""
        try:
            emotion = emotion_data["emotion"]
            scenario = emotion_data["scenario"]
            prompts = emotion_data["prompts"]

            # Select a random prompt
            prompt = prompts[interaction_id % len(prompts)]

            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            logger.debug(f"Running {emotion} interaction {interaction_id}")

            # Create emotional context
            system_message = f"You are role-playing in a conversation where you need to express {emotion} authentically. The scenario is: {scenario}. Respond naturally and emotionally to the user's question."

            # Step 1: Convert prompt to speech
            tts_start = asyncio.get_event_loop().time()
            tts_result = await vispark_client.text_to_speech(
                prompt,
                voice=self.config.tts_voice,
                size=self.config.tts_size
            )

            if tts_result.get("status") != "success":
                return {
                    "interaction_id": interaction_id,
                    "emotion": emotion,
                    "success": False,
                    "error": "TTS failed"
                }

            tts_duration = asyncio.get_event_loop().time() - tts_start

            # Step 2: Start Aivoco session and send prompt
            session_start = asyncio.get_event_loop().time()

            session_result = await aivoco_client.start_call(
                system_message=system_message
            )

            if session_result.get("status") != "success":
                return {
                    "interaction_id": interaction_id,
                    "emotion": emotion,
                    "success": False,
                    "error": "Session start failed"
                }

            # Send the emotional prompt
            audio_result = await aivoco_client.send_audio_data(
                tts_result["data"]["content"],
                has_audio=True,
                max_amplitude=1.0
            )

            if audio_result.get("status") != "success":
                await aivoco_client.stop_call()
                return {
                    "interaction_id": interaction_id,
                    "emotion": emotion,
                    "success": False,
                    "error": "Audio send failed"
                }

            # Get emotional response
            response = await self.run_with_timeout(
                aivoco_client.receive_audio_response(),
                timeout=20.0
            )

            if not response:
                await aivoco_client.stop_call()
                return {
                    "interaction_id": interaction_id,
                    "emotion": emotion,
                    "success": False,
                    "error": "No response received"
                }

            session_duration = asyncio.get_event_loop().time() - session_start

            # Step 3: Convert response to base64 for emotion analysis
            audio_base64 = response["audio_data"]

            # Step 4: Use Vispark Vision to analyze emotional content
            analysis_start = asyncio.get_event_loop().time()

            emotion_analysis_prompt = f"""
            Analyze the emotional content of this audio response in the context of {emotion}.
            Rate the following aspects on a scale of 1-10:

            1. Emotional authenticity: How genuine does the emotion sound?
            2. Emotional intensity: How strong is the emotional expression?
            3. Contextual appropriateness: How well does it fit the {emotion} scenario?
            4. Vocal expression quality: How well is the emotion conveyed through voice?

            Provide a detailed analysis and overall emotional accuracy score (0-100%).
            """

            analysis_result = await vispark_client.vision_analyze([
                {"type": "text", "content": emotion_analysis_prompt},
                {"type": "audio", "content": audio_base64}
            ])

            analysis_duration = asyncio.get_event_loop().time() - analysis_start

            # Step 5: Also convert to text for additional analysis
            stt_result = await vispark_client.speech_to_text(audio_base64)
            response_text = stt_result["data"]["content"] if stt_result.get("status") == "success" else ""

            # Stop session
            await aivoco_client.stop_call()

            # Parse emotion analysis
            emotion_score = self.parse_emotion_analysis(
                analysis_result["data"]["content"] if analysis_result.get("status") == "success" else ""
            )

            # Record metrics
            total_duration = tts_duration + session_duration + analysis_duration
            self.metrics.record_request_end(asyncio.get_event_loop().time() - total_duration, success=True)

            return {
                "interaction_id": interaction_id,
                "emotion": emotion,
                "scenario": scenario,
                "prompt": prompt,
                "success": True,
                "response_text": response_text,
                "emotion_analysis": analysis_result["data"]["content"] if analysis_result.get("status") == "success" else "",
                "emotion_score": emotion_score,
                "durations": {
                    "tts": tts_duration,
                    "session": session_duration,
                    "analysis": analysis_duration,
                    "total": total_duration
                }
            }

        except Exception as e:
            logger.error(f"Failed emotional interaction {interaction_id}: {e}")
            return {
                "interaction_id": interaction_id,
                "emotion": emotion_data.get("emotion", "unknown"),
                "success": False,
                "error": str(e)
            }

    def parse_emotion_analysis(self, analysis_text: str) -> float:
        """Parse emotion analysis to extract numerical score."""
        try:
            # Look for percentage or score patterns
            import re

            # Look for percentage patterns
            percent_match = re.search(r'(\d+(?:\.\d+)?)%', analysis_text)
            if percent_match:
                return float(percent_match.group(1))

            # Look for score patterns like "85/100" or "8.5/10"
            score_match = re.search(r'(\d+(?:\.\d+)?)/(\d+)', analysis_text)
            if score_match:
                numerator = float(score_match.group(1))
                denominator = float(score_match.group(2))
                return (numerator / denominator) * 100

            # Look for standalone numbers that might be scores
            number_match = re.search(r'(?:score|rating|accuracy).*?(\d+(?:\.\d+)?)', analysis_text.lower())
            if number_match:
                score = float(number_match.group(1))
                if score <= 10:  # Likely out of 10
                    return score * 10
                elif score <= 100:  # Already percentage
                    return score

            # Default to 50 if no clear score found
            return 50.0

        except Exception as e:
            logger.warning(f"Failed to parse emotion analysis: {e}")
            return 50.0

    async def run_benchmark(self) -> 'BenchmarkMetrics':
        """Run the emotions benchmark."""
        logger.info(f"Starting emotions benchmark with {len(self.emotional_scenarios)} emotions and {self.config.multimodal_iterations} interactions each")

        results = []

        for emotion_data in self.emotional_scenarios:
            emotion = emotion_data["emotion"]
            logger.info(f"Testing emotion: {emotion}")

            with self.create_progress_bar(self.config.multimodal_iterations, f"{emotion.title()} Interactions") as pbar:
                for i in range(self.config.multimodal_iterations):
                    try:
                        result = await self.run_emotional_interaction(emotion_data, len(results))
                        results.append(result)

                        if result["success"]:
                            emotion_score = result.get("emotion_score", 0)
                            logger.debug(f"{emotion.title()} interaction {i+1}: {emotion_score:.1f}% emotional accuracy")
                            self.metrics.record_audio_quality(emotion_score)
                        else:
                            logger.warning(f"{emotion.title()} interaction {i+1} failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        logger.error(f"Unexpected error in {emotion} interaction {i+1}: {e}")
                        results.append({
                            "interaction_id": len(results),
                            "emotion": emotion,
                            "success": False,
                            "error": str(e)
                        })

                    pbar.update(1)

        # Calculate final metrics
        await self.calculate_emotion_metrics(results)

        # Save detailed results
        await self.save_emotion_results(results)

        logger.info("Emotions benchmark completed")
        return self.metrics

    async def calculate_emotion_metrics(self, results: List[Dict[str, Any]]) -> None:
        """Calculate comprehensive emotion metrics."""
        try:
            successful_results = [r for r in results if r["success"]]
            emotion_scores = [r["emotion_score"] for r in successful_results if "emotion_score" in r]

            if emotion_scores:
                avg_emotion_score = sum(emotion_scores) / len(emotion_scores)
                self.metrics.add_custom_metric("avg_emotion_score", avg_emotion_score)

            # Calculate per-emotion metrics
            emotion_stats = {}
            for result in successful_results:
                emotion = result["emotion"]
                score = result.get("emotion_score", 0)

                if emotion not in emotion_stats:
                    emotion_stats[emotion] = []
                emotion_stats[emotion].append(score)

            for emotion, scores in emotion_stats.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    self.metrics.add_custom_metric(f"{emotion}_score", avg_score)

            # Overall success rate
            success_rate = len(successful_results) / len(results) if results else 0
            self.metrics.add_custom_metric("emotion_success_rate", success_rate)

            logger.info(f"Emotion benchmark results: {len(successful_results)}/{len(results)} successful, "
                       f"Avg emotion score: {avg_emotion_score:.1f}%")

        except Exception as e:
            logger.error(f"Failed to calculate emotion metrics: {e}")

    async def save_emotion_results(self, results: List[Dict[str, Any]]) -> None:
        """Save detailed emotion analysis results."""
        try:
            results_file = self.output_dir / "emotion_results.json"

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Saved emotion results to {results_file}")

            # Create summary by emotion
            emotion_summary = {}
            for result in results:
                if result["success"]:
                    emotion = result["emotion"]
                    if emotion not in emotion_summary:
                        emotion_summary[emotion] = {
                            "count": 0,
                            "total_score": 0,
                            "avg_score": 0
                        }

                    emotion_summary[emotion]["count"] += 1
                    emotion_summary[emotion]["total_score"] += result.get("emotion_score", 0)

            for emotion, stats in emotion_summary.items():
                if stats["count"] > 0:
                    stats["avg_score"] = stats["total_score"] / stats["count"]

            summary_file = self.output_dir / "emotion_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(emotion_summary, f, indent=2)

            logger.info(f"Saved emotion summary to {summary_file}")

        except Exception as e:
            logger.error(f"Failed to save emotion results: {e}")

    async def cleanup(self) -> None:
        """Cleanup emotions benchmark resources."""
        logger.info("Cleaning up emotions benchmark")
        # Cleanup is handled by parent class

async def main():
    """Main entry point for emotions benchmark."""
    try:
        # Load configuration
        config = BenchmarkConfig.from_env()

        # Create and run benchmark
        benchmark = EmotionsBenchmark(config)
        metrics = await benchmark.execute()

        logger.info(f"Emotions benchmark completed. Average emotion score: {metrics.average_audio_quality:.2f}%")

    except Exception as e:
        logger.error(f"Emotions benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
