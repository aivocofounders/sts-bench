# BFCL (Berkeley Function Calling Leaderboard) Benchmark

## Overview
The BFCL Benchmark evaluates function calling capabilities of speech-to-speech models by testing their ability to understand user requests via speech and generate appropriate function calls with correct parameters, then convert the function call responses back to natural speech.

## Flow

1. **Setup Phase**
   - Load BFCL test cases (function calling scenarios)
   - Initialize Vispark TTS, STT, and Aivoco clients
   - Test speech-to-text-to-speech pipeline

2. **Test Case Processing**
   - For each function calling scenario:
     - Convert user query to speech using Vispark TTS
     - Include system prompt about available functions
     - Send audio to speech-to-speech model
     - Receive audio response with function call
     - Convert response back to text using Vispark STT
     - Parse function call from transcribed text

3. **Function Call Evaluation**
   - Extract function name and parameters from response
   - Compare with expected function call
   - Calculate accuracy and success rates
   - Analyze parameter extraction quality

4. **Results Analysis**
   - Generate comprehensive performance metrics
   - Evaluate function calling accuracy via speech
   - Analyze error patterns in speech-based function calls

## Key Metrics

- **Function Call Accuracy**: Percentage of correct function identifications
- **Parameter Extraction Accuracy**: Quality of parameter parsing from speech
- **End-to-End Success Rate**: Complete successful function call pipelines
- **Processing Latency**: Time for TTS → Speech Model → STT → Parsing
- **Error Categories**: Analysis of common failure modes

## Test Categories

The benchmark includes various function calling scenarios:

- **Simple Functions**: Basic API calls (weather, calendar, email)
- **Complex Parameters**: Functions with multiple parameters
- **File Operations**: File system and data manipulation
- **API Interactions**: REST API and service integrations
- **Multi-step Tasks**: Complex workflows requiring multiple function calls

## Configuration

The benchmark uses the following default settings:
- **Test Cases**: 50+ diverse function calling scenarios
- **Response Timeout**: 30 seconds per test case
- **Evaluation Mode**: Exact function name and parameter matching
- **TTS Voice**: Clear voice for technical terminology

## Speech-to-Speech Challenges

The benchmark specifically tests:
- **Technical Vocabulary**: Recognition of function names and API terminology
- **Parameter Parsing**: Extraction of values from spoken responses
- **Context Preservation**: Maintaining conversation context through speech
- **Error Handling**: Robustness to transcription errors
- **Natural Language**: Understanding conversational function requests

## Output Files

- `results.jsonl`: Detailed results for each test case
- `metrics.json`: Comprehensive performance metrics
- `summary.csv`: Statistical summary of results
- `bfcl_[timestamp].log`: Detailed execution logs

## Usage

```bash
cd bfcl
python main.py
```

Ensure your `.env` file contains:
- `AIVOCO_API_KEY`: Your Aivoco API key
- `VISPARK_API_KEY`: Your Vispark API key

## Integration with Official BFCL

This implementation provides a speech-to-speech wrapper around the official BFCL evaluation framework. For full BFCL evaluation:

```bash
cd gorilla/berkeley-function-call-leaderboard
pip install -e .
# Then use the official evaluation tools
```

## Interpretation

The benchmark reveals:
- **Speech Comprehension**: How well models understand technical requests via speech
- **Response Clarity**: Quality of verbal function call generation
- **Parameter Accuracy**: Precision in spoken parameter values
- **Error Propagation**: How transcription errors affect function calling
- **Technical Communication**: Effectiveness of speech-based technical interactions

Results help determine if speech-to-speech models can effectively handle complex function calling tasks, which is crucial for voice-controlled applications and AI assistants requiring programmatic interactions.
