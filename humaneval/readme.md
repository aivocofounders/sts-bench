# HumanEval Benchmark

## Overview
The HumanEval Benchmark evaluates the functional correctness of code generation capabilities when processed through a speech-to-speech model. This benchmark converts HumanEval programming problems to speech, processes them through the speech-to-speech pipeline, and evaluates the accuracy of the generated code solutions.

## Flow

1. **Setup Phase**
   - Load HumanEval dataset (164 programming problems)
   - Initialize Vispark TTS, STT, and Aivoco clients
   - Test speech-to-text-to-speech pipeline

2. **Problem Processing**
   - For each HumanEval problem:
     - Extract problem description and requirements
     - Convert problem text to speech using Vispark TTS
     - Send audio to speech-to-speech model
     - Receive audio response
     - Convert response back to text using Vispark STT
     - Extract code completion from transcribed text

3. **Code Evaluation**
   - Process extracted code through HumanEval evaluation framework
   - Test functional correctness of generated solutions
   - Calculate pass rates and accuracy metrics

4. **Results Analysis**
   - Generate comprehensive performance metrics
   - Compare speech-processed results with direct text evaluation
   - Analyze error patterns and common failure modes

## Key Metrics

- **Functional Accuracy**: Percentage of problems solved correctly
- **Pass@k Rates**: Success rates for different numbers of attempts
- **Processing Latency**: Time for TTS → Speech Model → STT pipeline
- **Transcription Accuracy**: Quality of speech-to-text conversion
- **Code Extraction Success**: Ability to parse code from transcribed speech

## Configuration

The benchmark uses the following default settings:
- **Dataset**: Full HumanEval dataset (164 problems)
- **Processing Limit**: First 50 problems (configurable)
- **TTS Voice**: Girl voice for clear code dictation
- **Response Timeout**: 30 seconds per problem
- **Evaluation**: Standard HumanEval pass@k metrics

## Speech-to-Speech Integration

The benchmark specifically tests:
- **Code Comprehension**: Model's ability to understand programming problems via speech
- **Code Generation**: Quality of code produced through voice interaction
- **Technical Terminology**: Handling of programming-specific vocabulary
- **Code Structure**: Preservation of syntax and formatting in speech

## Output Files

- `samples.jsonl`: All generated samples with metadata
- `eval_samples.jsonl`: Successful completions for evaluation
- `eval_samples_results.jsonl`: HumanEval evaluation results
- `metrics.json`: Detailed performance metrics
- `summary.csv`: Statistical summary
- `humaneval_[timestamp].log`: Execution logs

## Usage

```bash
cd humaneval
python main.py
```

Ensure your `.env` file contains:
- `AIVOCO_API_KEY`: Your Aivoco API key
- `VISPARK_API_KEY`: Your Vispark API key

## Dependencies

This benchmark requires the HumanEval package:
```bash
cd human-eval
pip install -e .
```

## Interpretation

The benchmark reveals:
- **Speech Processing Impact**: How speech-to-speech conversion affects code generation accuracy
- **Technical Speech Handling**: Model performance with programming-specific content
- **Code Clarity**: How well the model communicates technical solutions verbally
- **Error Propagation**: How transcription errors affect final code quality

Results help identify whether speech-to-speech models can effectively handle complex technical content and programming tasks, which is crucial for applications requiring verbal code generation or explanation.
