"""
Metrics collection and analysis for benchmarks.
"""

import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from loguru import logger

@dataclass
class BenchmarkMetrics:
    """Metrics collection for benchmark results."""

    benchmark_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Timing metrics
    response_times: List[float] = field(default_factory=list)
    first_response_times: List[float] = field(default_factory=list)
    latency_measurements: List[float] = field(default_factory=list)

    # Success/failure metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    errors: List[str] = field(default_factory=list)

    # Audio quality metrics
    audio_qualities: List[float] = field(default_factory=list)
    transcription_accuracies: List[float] = field(default_factory=list)

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def record_request_start(self) -> float:
        """Record the start of a request."""
        return time.time()

    def record_request_end(self, start_time: float, success: bool = True, error: Optional[str] = None) -> None:
        """Record the end of a request."""
        end_time = time.time()
        response_time = end_time - start_time

        self.total_requests += 1
        self.response_times.append(response_time)

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                self.errors.append(error)

    def record_first_response(self, response_time: float) -> None:
        """Record first response time (for boot benchmarks)."""
        self.first_response_times.append(response_time)

    def record_latency(self, latency: float) -> None:
        """Record latency measurement."""
        self.latency_measurements.append(latency)

    def record_audio_quality(self, quality_score: float) -> None:
        """Record audio quality score."""
        self.audio_qualities.append(quality_score)

    def record_transcription_accuracy(self, accuracy: float) -> None:
        """Record transcription accuracy."""
        self.transcription_accuracies.append(accuracy)

    def add_custom_metric(self, key: str, value: Any) -> None:
        """Add a custom metric."""
        if key not in self.custom_metrics:
            self.custom_metrics[key] = []
        self.custom_metrics[key].append(value)

    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Total duration of benchmark in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def average_response_time(self) -> float:
        """Average response time in seconds."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def median_response_time(self) -> float:
        """Median response time in seconds."""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)

    @property
    def min_response_time(self) -> float:
        """Minimum response time in seconds."""
        if not self.response_times:
            return 0.0
        return min(self.response_times)

    @property
    def max_response_time(self) -> float:
        """Maximum response time in seconds."""
        if not self.response_times:
            return 0.0
        return max(self.response_times)

    @property
    def response_time_stddev(self) -> float:
        """Standard deviation of response times."""
        if len(self.response_times) < 2:
            return 0.0
        return statistics.stdev(self.response_times)

    @property
    def average_latency(self) -> float:
        """Average latency in seconds."""
        if not self.latency_measurements:
            return 0.0
        return statistics.mean(self.latency_measurements)

    @property
    def average_first_response_time(self) -> float:
        """Average first response time (for boot benchmarks)."""
        if not self.first_response_times:
            return 0.0
        return statistics.mean(self.first_response_times)

    @property
    def average_audio_quality(self) -> float:
        """Average audio quality score."""
        if not self.audio_qualities:
            return 0.0
        return statistics.mean(self.audio_qualities)

    @property
    def average_transcription_accuracy(self) -> float:
        """Average transcription accuracy."""
        if not self.transcription_accuracies:
            return 0.0
        return statistics.mean(self.transcription_accuracies)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": self.success_rate,
            "response_time_stats": {
                "average": self.average_response_time,
                "median": self.median_response_time,
                "min": self.min_response_time,
                "max": self.max_response_time,
                "stddev": self.response_time_stddev
            },
            "first_response_time_average": self.average_first_response_time,
            "latency_average": self.average_latency,
            "audio_quality_average": self.average_audio_quality,
            "transcription_accuracy_average": self.average_transcription_accuracy,
            "errors": self.errors,
            "custom_metrics": self.custom_metrics
        }

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Save metrics to JSON file."""
        try:
            data = self.to_dict()
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Metrics saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics to {file_path}: {e}")

    def save_summary_csv(self, file_path: Union[str, Path]) -> None:
        """Save metrics summary to CSV file."""
        try:
            data = {
                "metric": [
                    "benchmark_name", "duration_seconds", "total_requests",
                    "successful_requests", "failed_requests", "success_rate_percent",
                    "avg_response_time", "median_response_time", "min_response_time",
                    "max_response_time", "response_time_stddev", "avg_first_response_time",
                    "avg_latency", "avg_audio_quality", "avg_transcription_accuracy"
                ],
                "value": [
                    self.benchmark_name, self.duration, self.total_requests,
                    self.successful_requests, self.failed_requests, self.success_rate,
                    self.average_response_time, self.median_response_time, self.min_response_time,
                    self.max_response_time, self.response_time_stddev, self.average_first_response_time,
                    self.average_latency, self.average_audio_quality, self.average_transcription_accuracy
                ]
            }

            df = pd.DataFrame(data)
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            logger.info(f"Metrics summary saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics summary to {file_path}: {e}")

    def print_summary(self) -> None:
        """Print a summary of the metrics."""
        print(f"\n{'='*50}")
        print(f"BENCHMARK RESULTS: {self.benchmark_name.upper()}")
        print(f"{'='*50}")
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Total Requests: {self.total_requests}")
        print(f"Successful: {self.successful_requests}")
        print(f"Failed: {self.failed_requests}")
        print(f"Success Rate: {self.success_rate:.2f}%")

        if self.response_times:
            print(f"\nResponse Time Statistics:")
            print(f"  Average: {self.average_response_time:.3f}s")
            print(f"  Median: {self.median_response_time:.3f}s")
            print(f"  Min: {self.min_response_time:.3f}s")
            print(f"  Max: {self.max_response_time:.3f}s")
            print(f"  StdDev: {self.response_time_stddev:.3f}s")

        if self.first_response_times:
            print(f"Average First Response Time: {self.average_first_response_time:.3f}s")

        if self.latency_measurements:
            print(f"Average Latency: {self.average_latency:.3f}s")

        if self.audio_qualities:
            print(f"Average Audio Quality: {self.average_audio_quality:.3f}")

        if self.transcription_accuracies:
            print(f"Average Transcription Accuracy: {self.average_transcription_accuracy:.3f}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:5]):  # Show first 5 errors
                print(f"  {i+1}. {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")

        print(f"{'='*50}\n")
