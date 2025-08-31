
"""
BFCL (Berkeley Function Calling Leaderboard) Benchmark for Speech-to-Speech Models.

This benchmark evaluates function calling capabilities by converting BFCL test cases
to speech, processing through the speech-to-speech model, and evaluating the accuracy
of function calls generated from audio responses.
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

# Import BFCL components if available
try:
    BFCL_AVAILABLE = True
except ImportError:
    BFCL_AVAILABLE = False
    logger.warning("BFCL package not available, functionality will be limited")

class BFCLBenchmark(AbstractBenchmark):
    """BFCL benchmark with speech-to-speech integration."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "bfcl")
        self.audio_processor = AudioProcessor()
        self.test_cases = []
        self.results = []

    async def setup(self) -> bool:
        """Setup the BFCL benchmark."""
        try:
            logger.info("Setting up BFCL benchmark")

            # Load BFCL test cases (simplified version)
            await self.load_test_cases()

            if not self.test_cases:
                logger.error("No test cases loaded")
                return False

            # Test model connections
            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            # Test TTS/STT pipeline
            test_text = "Call function get_weather with location 'New York'"
            tts_result = await vispark_client.text_to_speech(test_text)
            if tts_result.get("status") != "success":
                logger.error("TTS test failed")
                return False

            logger.info(f"BFCL benchmark setup completed with {len(self.test_cases)} test cases")
            return True

        except Exception as e:
            logger.error(f"BFCL benchmark setup failed: {e}")
            return False

    async def load_test_cases(self) -> None:
        """Load BFCL test cases."""
        try:
            # Define some sample function calling test cases
            self.test_cases = [
                {
                    "id": "simple_weather",
                    "description": "Get weather information for a city",
                    "user_query": "What's the weather like in Tokyo?",
                    "expected_function": "get_weather",
                    "expected_parameters": {"location": "Tokyo"}
                },
                {
                    "id": "calendar_event",
                    "description": "Create a calendar event",
                    "user_query": "Schedule a meeting with John tomorrow at 3 PM",
                    "expected_function": "create_event",
                    "expected_parameters": {
                        "title": "Meeting with John",
                        "date": "tomorrow",
                        "time": "3 PM"
                    }
                },
                {
                    "id": "email_send",
                    "description": "Send an email",
                    "user_query": "Send an email to sarah@example.com with subject 'Project Update'",
                    "expected_function": "send_email",
                    "expected_parameters": {
                        "to": "sarah@example.com",
                        "subject": "Project Update"
                    }
                },
                {
                    "id": "file_operations",
                    "description": "File system operations",
                    "user_query": "Create a new folder called 'documents' in my home directory",
                    "expected_function": "create_folder",
                    "expected_parameters": {
                        "name": "documents",
                        "location": "home"
                    }
                },
                {
                    "id": "api_request",
                    "description": "Make an API request",
                    "user_query": "Get user information for user ID 12345",
                    "expected_function": "get_user",
                    "expected_parameters": {"user_id": "12345"}
                }
            ]

            # Try to load from BFCL data directory if available
            bfcl_data_dir = Path(__file__).parent / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
            if bfcl_data_dir.exists():
                await self.load_bfcl_data(bfcl_data_dir)

        except Exception as e:
            logger.warning(f"Failed to load test cases: {e}")

    async def load_bfcl_data(self, data_dir: Path) -> None:
        """Load actual BFCL test data."""
        try:
            # Load multiple BFCL test files for comprehensive evaluation
            bfcl_files = [
                "BFCL_v4_live_simple.json",
                "BFCL_v4_live_multiple.json",
                "BFCL_v4_live_parallel.json",
                "BFCL_v4_simple_java.json",
                "BFCL_v4_simple_javascript.json",
                "BFCL_v4_simple_python.json"
            ]

            total_loaded = 0
            for file_name in bfcl_files:
                file_path = data_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)

                    # Load up to 20 cases per file for comprehensive testing
                    cases_to_load = min(20, len(file_data))
                    for item in file_data[:cases_to_load]:
                        self.test_cases.append({
                            "id": item.get("id", f"bfcl_{len(self.test_cases)}"),
                            "description": f"BFCL {file_name.replace('.json', '').replace('BFCL_v4_', '')}",
                            "user_query": item.get("question", ""),
                            "expected_function": item.get("function", None),
                            "expected_parameters": item.get("parameters", None),
                            "category": self.categorize_bfcl_case(item)
                        })

                    logger.info(f"Loaded {cases_to_load} cases from {file_name}")
                    total_loaded += cases_to_load

            logger.info(f"Total BFCL test cases loaded: {total_loaded}")

        except Exception as e:
            logger.warning(f"Failed to load BFCL data: {e}")

    def categorize_bfcl_case(self, item: Dict[str, Any]) -> str:
        """Categorize BFCL test case by type."""
        try:
            question = item.get("question", "").lower()
            function = item.get("function", "")

            if "multiple" in item.get("id", "").lower():
                return "multiple_functions"
            elif "parallel" in item.get("id", "").lower():
                return "parallel_functions"
            elif function:
                return "single_function"
            elif any(word in question for word in ["java", "javascript", "python", "code"]):
                return "programming"
            else:
                return "general"

        except Exception:
            return "general"

    async def process_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single BFCL test case through speech-to-speech pipeline."""
        try:
            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            test_id = test_case.get('id', 'unknown')
            user_query = test_case.get('user_query', '')

            logger.debug(f"Processing test case {test_id}")

            # Step 1: Convert user query to speech
            tts_start = asyncio.get_event_loop().time()
            system_prompt = "You are an AI assistant with access to various functions. When users ask for something, respond by calling the appropriate function with correct parameters."

            tts_result = await vispark_client.text_to_speech(
                f"{system_prompt} User query: {user_query}",
                voice=self.config.tts_voice,
                size=self.config.tts_size
            )

            if tts_result.get("status") != "success":
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": "TTS failed",
                    "response": ""
                }

            tts_duration = asyncio.get_event_loop().time() - tts_start

            # Step 2: Send to speech-to-speech model
            session_start = asyncio.get_event_loop().time()

            # Start session
            session_result = await aivoco_client.start_call(
                system_message=system_prompt
            )

            if session_result.get("status") != "success":
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": "Session start failed",
                    "response": ""
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
                    "test_id": test_id,
                    "success": False,
                    "error": "Audio send failed",
                    "response": ""
                }

            # Receive response
            response = await self.run_with_timeout(
                aivoco_client.receive_audio_response(),
                timeout=30.0
            )

            if not response:
                await aivoco_client.stop_call()
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": "No response received",
                    "response": ""
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
                    "test_id": test_id,
                    "success": False,
                    "error": "STT failed",
                    "response": ""
                }

            stt_duration = asyncio.get_event_loop().time() - stt_start
            response_text = stt_result["data"]["content"]

            # Step 4: Evaluate response
            evaluation = self.evaluate_response(response_text, test_case)

            # Record metrics
            total_duration = tts_duration + session_duration + stt_duration
            self.metrics.record_request_end(asyncio.get_event_loop().time() - total_duration,
                                          success=evaluation["correct"])

            return {
                "test_id": test_id,
                "success": True,
                "response": response_text,
                "evaluation": evaluation,
                "tts_duration": tts_duration,
                "session_duration": session_duration,
                "stt_duration": stt_duration,
                "total_duration": total_duration
            }

        except Exception as e:
            logger.error(f"Failed to process test case {test_case.get('id', 'unknown')}: {e}")
            return {
                "test_id": test_case.get('id', 'unknown'),
                "success": False,
                "error": str(e),
                "response": ""
            }

    def evaluate_response(self, response_text: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the response using BFCL evaluation criteria."""
        try:
            evaluation = {
                "correct": False,
                "function_called": None,
                "parameters_extracted": {},
                "accuracy_score": 0,
                "reasoning": "",
                "category": test_case.get('category', 'general')
            }

            response_lower = response_text.lower()
            expected_function = test_case.get('expected_function')
            expected_params = test_case.get('expected_parameters', {})

            # Enhanced function detection
            if expected_function:
                # Check for exact function name
                if expected_function.lower() in response_lower:
                    evaluation["function_called"] = expected_function
                    evaluation["correct"] = True
                    evaluation["accuracy_score"] = 80
                    evaluation["reasoning"] = f"Function '{expected_function}' correctly identified"
                else:
                    # Check for partial matches or synonyms
                    function_keywords = expected_function.lower().split('_')
                    matches = sum(1 for keyword in function_keywords if keyword in response_lower)
                    if matches > 0:
                        evaluation["function_called"] = expected_function
                        evaluation["accuracy_score"] = min(60, matches * 20)
                        evaluation["reasoning"] = f"Partial function match ({matches}/{len(function_keywords)} keywords)"

            # Parameter extraction with better accuracy
            if expected_params:
                extracted_params = {}
                for param_name, param_value in expected_params.items():
                    if isinstance(param_value, str):
                        if param_value.lower() in response_lower:
                            extracted_params[param_name] = param_value
                            evaluation["accuracy_score"] += 10
                    elif isinstance(param_value, (int, float)):
                        # Try to find numeric values in response
                        import re
                        numbers = re.findall(r'\d+', response_text)
                        if str(param_value) in numbers:
                            extracted_params[param_name] = param_value
                            evaluation["accuracy_score"] += 10

                evaluation["parameters_extracted"] = extracted_params

                # Bonus for complete parameter extraction
                if len(extracted_params) == len(expected_params):
                    evaluation["accuracy_score"] += 20
                    evaluation["reasoning"] += " - All parameters correctly extracted"

            # Category-specific evaluation
            category = test_case.get('category', 'general')
            if category == "multiple_functions" and "multiple" in response_lower:
                evaluation["accuracy_score"] += 15
            elif category == "parallel_functions" and ("parallel" in response_lower or "simultaneous" in response_lower):
                evaluation["accuracy_score"] += 15
            elif category == "programming" and any(lang in response_lower for lang in ["java", "javascript", "python"]):
                evaluation["accuracy_score"] += 10

            # Cap accuracy score
            evaluation["accuracy_score"] = min(100, evaluation["accuracy_score"])

            # Final correctness determination
            evaluation["correct"] = evaluation["accuracy_score"] >= 70

            if not evaluation["reasoning"]:
                evaluation["reasoning"] = f"Accuracy score: {evaluation['accuracy_score']}%, Category: {category}"

            return evaluation

        except Exception as e:
            logger.warning(f"Failed to evaluate BFCL response: {e}")
            return {
                "correct": False,
                "accuracy_score": 0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "category": test_case.get('category', 'general')
            }

    async def run_benchmark(self) -> 'BenchmarkMetrics':
        """Run the BFCL benchmark."""
        logger.info(f"Starting BFCL benchmark with {len(self.test_cases)} test cases")

        with self.create_progress_bar(len(self.test_cases), "BFCL Test Cases") as pbar:
            for test_case in self.test_cases:
                try:
                    result = await self.process_test_case(test_case)
                    self.results.append(result)

                    if result["success"]:
                        accuracy_score = result.get("evaluation", {}).get("accuracy_score", 0)
                        logger.debug(f"Test case {result['test_id']} processed successfully: {accuracy_score:.1f}% accuracy")
                        self.metrics.record_audio_quality(accuracy_score)
                    else:
                        logger.warning(f"Test case {result['test_id']} failed: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Unexpected error processing test case {test_case.get('id', 'unknown')}: {e}")
                    self.results.append({
                        "test_id": test_case.get('id', 'unknown'),
                        "success": False,
                        "error": str(e),
                        "response": ""
                    })

                pbar.update(1)

        # Calculate final metrics
        await self.calculate_final_metrics()

        logger.info("BFCL benchmark completed")
        return self.metrics

    async def calculate_final_metrics(self) -> None:
        """Calculate final benchmark metrics."""
        try:
            total_cases = len(self.results)
            successful_cases = sum(1 for r in self.results if r["success"])
            correct_cases = sum(1 for r in self.results if r.get("evaluation", {}).get("correct", False))

            accuracy = correct_cases / total_cases if total_cases > 0 else 0
            success_rate = successful_cases / total_cases if total_cases > 0 else 0

            self.metrics.add_custom_metric("bfcl_accuracy", accuracy)
            self.metrics.add_custom_metric("bfcl_success_rate", success_rate)
            self.metrics.add_custom_metric("bfcl_correct", correct_cases)
            self.metrics.add_custom_metric("bfcl_total", total_cases)

            # Calculate average durations
            durations = [r.get("total_duration", 0) for r in self.results if r.get("success")]
            if durations:
                avg_duration = sum(durations) / len(durations)
                self.metrics.add_custom_metric("bfcl_avg_duration", avg_duration)

            logger.info(f"BFCL Results: {correct_cases}/{total_cases} correct ({accuracy:.2%}), "
                       f"{successful_cases}/{total_cases} successful ({success_rate:.2%})")

        except Exception as e:
            logger.error(f"Failed to calculate final metrics: {e}")

    async def cleanup(self) -> None:
        """Cleanup BFCL benchmark resources."""
        logger.info("Cleaning up BFCL benchmark")
        # Cleanup is handled by parent class

async def main():
    """Main entry point for BFCL benchmark."""
    try:
        # Load configuration
        config = BenchmarkConfig.from_env()

        # Create and run benchmark
        benchmark = BFCLBenchmark(config)
        metrics = await benchmark.execute()

        logger.info(f"BFCL benchmark completed. Success rate: {metrics.success_rate:.2f}%")

    except Exception as e:
        logger.error(f"BFCL benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
