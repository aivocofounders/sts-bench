# Emotions Benchmark

## Overview
The Emotions Benchmark evaluates the speech-to-speech model's ability to convey and recognize emotions authentically. Using Vispark Vision as an AI judge, this benchmark tests how well the model expresses various emotions through voice and responds appropriately to emotional contexts.

## Flow

1. **Setup Phase**
   - Initialize Vispark Vision, TTS, STT, and Aivoco clients
   - Test multimodal emotion analysis capabilities
   - Prepare emotional scenario templates

2. **Emotional Scenarios**
   - Test 8 core emotions with contextual scenarios:
     - **Joy**: Celebrating achievements and happy memories
     - **Sadness**: Dealing with loss and difficult situations
     - **Anger**: Facing unfair treatment and frustration
     - **Fear**: Confronting frightening experiences
     - **Surprise**: Unexpected events and shocks
     - **Disgust**: Unpleasant encounters and revulsion
     - **Trust**: Building relationships and reliability
     - **Anticipation**: Excitement and looking forward

3. **Interaction Process**
   - For each emotion and interaction:
     - Select appropriate emotional prompt
     - Convert prompt to speech using Vispark TTS
     - Send to speech-to-speech model with emotional context
     - Receive emotional response
     - Use Vispark Vision to analyze emotional authenticity
     - Score emotional expression quality

4. **Emotion Analysis**
   - **Emotional Authenticity**: How genuine the emotion sounds
   - **Emotional Intensity**: Strength of emotional expression
   - **Contextual Appropriateness**: Fit with scenario requirements
   - **Vocal Expression Quality**: Effectiveness of voice modulation

5. **Results Aggregation**
   - Calculate average emotion scores per emotion type
   - Generate overall emotional intelligence metrics
   - Analyze emotion recognition and expression patterns

## Key Metrics

- **Overall Emotion Score**: Average emotional authenticity across all interactions
- **Per-Emotion Accuracy**: Individual emotion expression quality
- **Emotional Consistency**: How consistent emotion conveyance is
- **Context Awareness**: Appropriateness of emotional responses
- **Vocal Expressiveness**: Quality of emotional voice modulation

## Configuration

The benchmark uses the following default settings:
- **Emotions Tested**: 8 core emotional categories
- **Interactions per Emotion**: 100 (configurable)
- **Response Timeout**: 20 seconds per interaction
- **Analysis Depth**: Detailed multimodal emotion analysis
- **Scoring Scale**: 0-100% emotional accuracy

## Multimodal AI Judging

The benchmark leverages Vispark Vision's advanced capabilities:
- **Audio Emotion Analysis**: Direct emotional content analysis from audio
- **Contextual Understanding**: Scenario-aware emotion evaluation
- **Cross-Modal Validation**: Text transcript correlation with audio emotion
- **Detailed Feedback**: Specific recommendations for emotional improvement

## Emotional Intelligence Assessment

This benchmark evaluates:
- **Emotional Expression**: Ability to convey emotions through voice
- **Emotional Recognition**: Understanding emotional context from prompts
- **Emotional Appropriateness**: Contextually appropriate emotional responses
- **Emotional Nuance**: Subtle emotional expression variations
- **Emotional Consistency**: Maintaining emotional tone throughout interaction

## Output Files

- `emotion_results.json`: Detailed results for each emotional interaction
- `emotion_summary.json`: Summary statistics by emotion type
- `metrics.json`: Comprehensive performance metrics
- `summary.csv`: Statistical overview
- `emotions_[timestamp].log`: Detailed execution logs

## Usage

```bash
cd emotions
python main.py
```

Ensure your `.env` file contains:
- `AIVOCO_API_KEY`: Your Aivoco API key
- `VISPARK_API_KEY`: Your Vispark API key

## Interpretation

The benchmark reveals:
- **Emotional Authenticity**: How genuine emotions sound in speech
- **Emotional Range**: Breadth of emotional expression capability
- **Contextual Awareness**: Understanding when to use different emotions
- **Vocal Emotional Intelligence**: Quality of emotion conveyance through voice
- **Consistency**: Reliability of emotional expression across scenarios

Results help determine if speech-to-speech models can effectively handle emotionally nuanced conversations, which is crucial for applications requiring empathetic and emotionally intelligent interactions like therapy, counseling, customer service, and personal assistance.
