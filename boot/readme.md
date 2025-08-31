# Boot Benchmark

## Overview
The Boot Benchmark measures the cold start performance of speech-to-speech models by evaluating how quickly they can respond to initial requests after establishing a connection.

## Flow

1. **Setup Phase**
   - Initialize Aivoco client connection
   - Generate test audio data (1 second sine wave)
   - Prepare authentication and session parameters

2. **Benchmark Execution**
   - For each of 100 iterations:
     - Start a new call session with Aivoco
     - Send test audio data immediately
     - Measure time to first audio response
     - Stop the call session
     - Record metrics

3. **Metrics Collection**
   - First response times for each iteration
   - Success/failure rates
   - Average, median, min, max response times
   - Error tracking

4. **Results Analysis**
   - Generate comprehensive metrics report
   - Save results to JSON and CSV formats
   - Calculate statistical summaries

## Key Metrics

- **Average First Response Time**: Mean time for initial responses across all iterations
- **Median First Response Time**: Median time for initial responses
- **Success Rate**: Percentage of successful call sessions
- **Response Time Distribution**: Statistical analysis of response times

## Configuration

The benchmark uses the following default settings:
- **Iterations**: 100 calls
- **Test Audio**: 1 second, 16kHz mono sine wave
- **Timeout**: 10 seconds per iteration
- **Delay**: 100ms between iterations

## Output Files

- `metrics.json`: Detailed metrics and timing data
- `summary.csv`: Summary statistics for easy analysis
- `boot_[timestamp].log`: Detailed execution logs

## Usage

```bash
cd boot
python main.py
```

Ensure your `.env` file contains:
- `AIVOCO_API_KEY`: Your Aivoco API key
- `VISPARK_API_KEY`: Your Vispark API key (for potential future extensions)

## Interpretation

Lower average first response times indicate better cold start performance. This benchmark helps identify:
- Connection establishment overhead
- Initial processing latency
- System responsiveness under sequential load
- Consistency of performance across multiple sessions
