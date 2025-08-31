# HLE (Helpful, Honest, and Harmless) Benchmark

## Overview
The HLE Benchmark evaluates the model's ability to provide Helpful, Honest, and Harmless responses through speech-to-speech interaction. This benchmark tests the alignment of speech-to-speech models with positive AI behavior guidelines by assessing responses to various challenging scenarios.

## Flow

1. **HLE Setup**
   - Define scenarios testing HLE principles through speech
   - Initialize Vispark TTS, STT, and Aivoco clients
   - Prepare challenging conversational scenarios

2. **Scenario Processing**
   - Convert HLE test prompts to speech input
   - Generate responses through speech-to-speech pipeline
   - Evaluate responses for HLE compliance

3. **HLE Evaluation**
   - **Helpfulness**: Providing useful and actionable responses
   - **Honesty**: Being truthful and acknowledging uncertainties
   - **Harmlessness**: Avoiding harmful suggestions or content

4. **Comprehensive Analysis**
   - Score responses across HLE dimensions
   - Identify patterns in response quality
   - Generate detailed compliance metrics

## Key Metrics

- **Overall HLE Score**: Average performance across all dimensions
- **Helpfulness Score**: Quality of helpful responses
- **Honesty Score**: Accuracy and transparency in responses
- **Harmlessness Score**: Safety and appropriateness of responses
- **HLE Compliance Rate**: Percentage of fully compliant responses

## Test Categories

### Helpfulness Scenarios
- **Complex Problem Solving**: Technical assistance and guidance
- **Creative Tasks**: Brainstorming and idea generation
- **Learning Support**: Educational explanations and tutoring

### Honesty Scenarios
- **Uncertain Information**: Handling incomplete knowledge
- **Misinformation Correction**: Addressing false claims
- **Evidence-Based Responses**: Supporting claims with reasoning

### Harmlessness Scenarios
- **Safety Concerns**: Rejecting dangerous requests
- **Ethical Boundaries**: Maintaining appropriate behavior
- **Responsible AI**: Promoting positive outcomes

## Configuration

- **Test Scenarios**: 5 comprehensive HLE test cases
- **Interactions per Scenario**: 10 attempts for reliability
- **Response Evaluation**: Automated HLE compliance scoring
- **Speech Processing**: Full TTS-STT pipeline testing

## Usage

```bash
cd hle
python main.py
```

Ensure your `.env` file contains:
- `AIVOCO_API_KEY`: Your Aivoco API key
- `VISPARK_API_KEY`: Your Vispark API key

## Interpretation

This benchmark reveals:
- **Ethical Alignment**: How well the model follows positive AI guidelines
- **Safety Compliance**: Ability to refuse harmful requests appropriately
- **Helpful Communication**: Effectiveness in providing useful responses
- **Truthful Interaction**: Honesty in speech-based communication
- **Responsible AI Behavior**: Overall alignment with ethical AI principles

Results help determine if speech-to-speech models can maintain HLE standards in voice interactions, which is crucial for safe and beneficial AI deployment in real-world applications.
