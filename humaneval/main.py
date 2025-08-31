"""
HumanEval Benchmark for Speech-to-Speech Models.

This benchmark evaluates functional correctness of code generation by converting
HumanEval problems to speech, processing through the speech-to-speech model,
and converting responses back to text for evaluation.
"""

import asyncio
import json
import os
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger

from ..base.benchmark import AbstractBenchmark
from ..base.config import BenchmarkConfig
from ..base.audio import AudioProcessor

# Import HumanEval components
try:
    from human_eval.data import read_problems
    from human_eval.evaluation import evaluate_functional_correctness
    from human_eval.execution import check_correctness
    HUMAN_EVAL_AVAILABLE = True
except ImportError:
    HUMAN_EVAL_AVAILABLE = False
    logger.warning("HumanEval package not available, functionality will be limited")

class HumanEvalBenchmark(AbstractBenchmark):
    """HumanEval benchmark with speech-to-speech integration."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "humaneval")
        self.audio_processor = AudioProcessor()
        self.problems = {}
        self.samples = []

    async def setup(self) -> bool:
        """Setup the HumanEval benchmark."""
        try:
            logger.info("Setting up HumanEval benchmark")

            if not HUMAN_EVAL_AVAILABLE:
                logger.error("HumanEval package not available")
                return False

            # Load HumanEval problems
            self.problems = read_problems()
            logger.info(f"Loaded {len(self.problems)} HumanEval problems")

            # Test model connections
            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            # Test TTS/STT pipeline
            test_text = "def test_function(): return 'Hello World'"
            tts_result = await vispark_client.text_to_speech(test_text)
            if tts_result.get("status") != "success":
                logger.error("TTS test failed")
                return False

            logger.info("HumanEval benchmark setup completed")
            return True

        except Exception as e:
            logger.error(f"HumanEval benchmark setup failed: {e}")
            return False

    async def process_problem(self, problem_id: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single HumanEval problem through speech-to-speech pipeline."""
        try:
            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            # Extract problem components
            prompt = problem.get('prompt', '')
            task_id = problem.get('task_id', problem_id)

            logger.debug(f"Processing problem {task_id}")

            # Step 1: Convert problem prompt to speech
            tts_start = asyncio.get_event_loop().time()
            tts_result = await vispark_client.text_to_speech(
                prompt,
                voice=self.config.tts_voice,
                size=self.config.tts_size
            )

            if tts_result.get("status") != "success":
                return {
                    "task_id": task_id,
                    "completion": "",
                    "error": "TTS failed",
                    "success": False
                }

            tts_duration = asyncio.get_event_loop().time() - tts_start

            # Step 2: Send to speech-to-speech model
            session_start = asyncio.get_event_loop().time()

            # Start session
            session_result = await aivoco_client.start_call(
                system_message="You are a coding assistant. Generate Python code solutions clearly and completely."
            )

            if session_result.get("status") != "success":
                return {
                    "task_id": task_id,
                    "completion": "",
                    "error": "Session start failed",
                    "success": False
                }

            # Send audio
            audio_result = await aivoco_client.send_audio_data(
                tts_result["data"]["content"],
                has_audio=True,
                max_amplitude=1.0
            )

            if audio_result.get("status") != "success":
                await aivoco_client.stop_call()
                return {
                    "task_id": task_id,
                    "completion": "",
                    "error": "Audio send failed",
                    "success": False
                }

            # Receive response
            response = await self.run_with_timeout(
                aivoco_client.receive_audio_response(),
                timeout=30.0
            )

            if not response:
                await aivoco_client.stop_call()
                return {
                    "task_id": task_id,
                    "completion": "",
                    "error": "No response received",
                    "success": False
                }

            session_duration = asyncio.get_event_loop().time() - session_start

            # Stop session
            await aivoco_client.stop_call()

            # Step 3: Convert response back to text
            stt_start = asyncio.get_event_loop().time()
            stt_result = await vispark_client.speech_to_text(
                response["audio_data"]
            )

            if stt_result.get("status") != "success":
                return {
                    "task_id": task_id,
                    "completion": "",
                    "error": "STT failed",
                    "success": False
                }

            stt_duration = asyncio.get_event_loop().time() - stt_start
            transcribed_text = stt_result["data"]["content"]

            # Extract code completion (remove prompt if present)
            completion = self.extract_completion(transcribed_text, prompt)

            # Record metrics
            total_duration = tts_duration + session_duration + stt_duration
            self.metrics.record_request_end(asyncio.get_event_loop().time() - total_duration,
                                          success=True)

            return {
                "task_id": task_id,
                "completion": completion,
                "transcribed_text": transcribed_text,
                "tts_duration": tts_duration,
                "session_duration": session_duration,
                "stt_duration": stt_duration,
                "total_duration": total_duration,
                "success": True
            }

        except Exception as e:
            logger.error(f"Failed to process problem {problem_id}: {e}")
            return {
                "task_id": task_id,
                "completion": "",
                "error": str(e),
                "success": False
            }

    def extract_completion(self, transcribed_text: str, prompt: str) -> str:
        """Extract code completion from transcribed text."""
        try:
            # Remove the original prompt if it's included in the response
            if prompt in transcribed_text:
                completion = transcribed_text.replace(prompt, "", 1).strip()
            else:
                completion = transcribed_text.strip()

            # Try to extract just the function definition and body
            lines = completion.split('\n')
            in_function = False
            function_lines = []

            for line in lines:
                line = line.strip()
                if line.startswith('def ') or line.startswith('class '):
                    in_function = True
                if in_function:
                    function_lines.append(line)
                    # Stop at empty line after function or next function/class
                    if not line and function_lines:
                        break
                    if (len(function_lines) > 1 and
                        (line.startswith('def ') or line.startswith('class '))):
                        function_lines.pop()  # Remove the new function line
                        break

            if function_lines:
                return '\n'.join(function_lines).strip()
            else:
                return completion

        except Exception as e:
            logger.warning(f"Failed to extract completion: {e}")
            return transcribed_text

    async def run_benchmark(self) -> 'BenchmarkMetrics':
        """Run the HumanEval benchmark."""
        logger.info(f"Starting HumanEval benchmark with {len(self.problems)} problems")

        # Limit to first 50 problems for testing (can be configured)
        problems_to_process = list(self.problems.items())[:50]

        with self.create_progress_bar(len(problems_to_process), "HumanEval Problems") as pbar:
            for problem_id, problem in problems_to_process:
                try:
                    result = await self.process_problem(problem_id, problem)
                    self.samples.append(result)

                    if result["success"]:
                        logger.debug(f"Problem {problem_id} processed successfully")
                    else:
                        logger.warning(f"Problem {problem_id} failed: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Unexpected error processing problem {problem_id}: {e}")
                    self.samples.append({
                        "task_id": problem_id,
                        "completion": "",
                        "error": str(e),
                        "success": False
                    })

                pbar.update(1)

        # Save samples for evaluation
        await self.save_samples()

        # Run HumanEval evaluation if possible
        if HUMAN_EVAL_AVAILABLE:
            await self.run_evaluation()

        logger.info("HumanEval benchmark completed")
        return self.metrics

    async def save_samples(self) -> None:
        """Save generated samples."""
        try:
            samples_file = self.output_dir / "samples.jsonl"

            with open(samples_file, 'w') as f:
                for sample in self.samples:
                    json.dump(sample, f)
                    f.write('\n')

            logger.info(f"Saved {len(self.samples)} samples to {samples_file}")

            # Save successful completions only for evaluation
            successful_samples = [
                {"task_id": s["task_id"], "completion": s["completion"]}
                for s in self.samples
                if s["success"] and s["completion"].strip()
            ]

            eval_samples_file = self.output_dir / "eval_samples.jsonl"
            with open(eval_samples_file, 'w') as f:
                for sample in successful_samples:
                    json.dump(sample, f)
                    f.write('\n')

            logger.info(f"Saved {len(successful_samples)} successful samples for evaluation")

        except Exception as e:
            logger.error(f"Failed to save samples: {e}")

    async def run_evaluation(self) -> None:
        """Run HumanEval functional correctness evaluation."""
        try:
            eval_samples_file = self.output_dir / "eval_samples.jsonl"

            if not eval_samples_file.exists():
                logger.warning("No samples file found for evaluation")
                return

            # Run evaluation
            logger.info("Running HumanEval functional correctness evaluation")
            evaluate_functional_correctness(str(eval_samples_file))

            # Read results
            results_file = eval_samples_file.with_suffix("") + "_results.jsonl"

            if results_file.exists():
                correct_count = 0
                total_count = 0

                with open(results_file, 'r') as f:
                    for line in f:
                        result = json.loads(line)
                        total_count += 1
                        if result.get("passed", False):
                            correct_count += 1

                accuracy = correct_count / total_count if total_count > 0 else 0
                self.metrics.add_custom_metric("humaneval_accuracy", accuracy)
                self.metrics.add_custom_metric("humaneval_correct", correct_count)
                self.metrics.add_custom_metric("humaneval_total", total_count)

                logger.info(f"HumanEval Results: {correct_count}/{total_count} correct ({accuracy:.2%})")

        except Exception as e:
            logger.error(f"HumanEval evaluation failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup HumanEval benchmark resources."""
        logger.info("Cleaning up HumanEval benchmark")
        # Cleanup is handled by parent class

async def main():
    """Main entry point for HumanEval benchmark."""
    try:
        # Load configuration
        config = BenchmarkConfig.from_env()

        # Create and run benchmark
        benchmark = HumanEvalBenchmark(config)
        metrics = await benchmark.execute()

        logger.info(f"HumanEval benchmark completed. Success rate: {metrics.success_rate:.2f}%")

    except Exception as e:
        logger.error(f"HumanEval benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
