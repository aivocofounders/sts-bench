# Latency Benchmark

## Overview
The Latency Benchmark evaluates sustained performance and response consistency of speech-to-speech models by running 100 extended sessions (30 minutes each), simulating realistic conversation scenarios with natural interaction patterns.

## Flow

1. **Setup Phase**
   - Initialize Vispark Vision, TTS, STT, and Aivoco clients
   - Test all model connections and APIs
   - Prepare conversation scenario templates

2. **Scenario Generation**
   - For each session, use Vispark Vision to generate unique conversation scenarios:
     - Customer service calls
     - Medical consultations
     - Technical support
     - Restaurant reservations
     - Banking inquiries
     - Travel bookings
     - Educational tutoring
     - Job interviews
     - Language learning
     - Emergency services

3. **Session Execution**
   - Start 30-minute conversation session with Aivoco
   - Generate initial message using Vispark Vision
   - Convert text to speech using Vispark TTS
   - Send audio to speech-to-speech model
   - Receive audio response
   - Convert response to text using Vispark STT
   - Generate contextual follow-up using Vispark Vision
   - Repeat conversation flow with natural delays

4. **Metrics Collection**
   - Response latency for each message exchange
   - Message count per session
   - Session success/failure rates
   - Error tracking and categorization
   - Conversation flow analysis

5. **Results Analysis**
   - Average latency across all sessions
   - Latency distribution and percentiles
   - Session completion rates
   - Error analysis and patterns

## Key Metrics

- **Average Latency**: Mean response time across all message exchanges
- **Latency Percentiles**: P50, P95, P99 response times
- **Session Completion Rate**: Percentage of full 30-minute sessions completed
- **Message Throughput**: Average messages per session
- **Error Rate**: Failed interactions percentage

## Configuration

The benchmark uses the following default settings:
- **Sessions**: 100 extended sessions
- **Duration per Session**: 30 minutes
- **Message Delay**: 2-5 seconds between exchanges
- **Response Timeout**: 15 seconds per message
- **Scenarios**: 10 diverse conversation types

## Realistic Environment Features

- **Dynamic Scenarios**: AI-generated conversation contexts
- **Natural Flow**: Contextual follow-up messages
- **Variable Timing**: Realistic delays between exchanges
- **Error Handling**: Graceful failure recovery
- **Multi-turn Conversations**: Extended dialogue simulation

## Output Files

- `metrics.json`: Detailed latency measurements and session data
- `summary.csv`: Statistical summary of latency performance
- `latency_[timestamp].log`: Comprehensive execution logs
- Session summaries with individual performance metrics

## Usage

```bash
cd latency
python main.py
```

Ensure your `.env` file contains:
- `AIVOCO_API_KEY`: Your Aivoco API key
- `VISPARK_API_KEY`: Your Vispark API key

## Interpretation

This benchmark provides insights into:
- **Sustained Performance**: How well the model maintains performance over time
- **Real-world Latency**: Response times in conversational scenarios
- **Reliability**: Consistency across extended interactions
- **Error Recovery**: Ability to handle conversation flow disruptions
- **Scalability**: Performance under prolonged usage

Lower latency and higher completion rates indicate better real-world performance for production deployment.
