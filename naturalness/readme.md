# Naturalness Benchmark

## Overview
The Naturalness Benchmark evaluates the fluency and authenticity of speech generation across multiple languages using Vispark Vision as an AI judge to assess conversational flow, pronunciation accuracy, and linguistic naturalness. This comprehensive benchmark now includes both European and Indian regional languages to test global language capabilities.

## Flow

1. **Multilingual Setup**
   - Test speech generation in 15 languages: 5 European + 10 Indian regional languages
   - Initialize Vispark TTS, STT, and Aivoco clients
   - Prepare diverse conversational scenarios in native scripts

2. **Cross-Language Testing**
   - Generate natural conversation prompts in each language using native scripts
   - Process through speech-to-speech model
   - Analyze response naturalness using Vispark Vision multimodal analysis

3. **Naturalness Evaluation**
   - **Fluency**: Smoothness of speech flow and transitions
   - **Pronunciation**: Authenticity of accent and phonetics
   - **Grammar**: Naturalness of linguistic structure in native scripts
   - **Conversational Flow**: Back-and-forth interaction quality
   - **Script Accuracy**: Proper handling of different writing systems

4. **Language-Specific Analysis**
   - Compare naturalness scores across European and Indian language families
   - Identify language-specific strengths and weaknesses
   - Evaluate script-specific challenges (Latin, Devanagari, Bengali, etc.)
   - Assess regional accent authenticity

## Key Metrics

- **Overall Naturalness Score**: Average fluency across all 15 languages
- **European Languages Average**: Performance on Western languages
- **Indian Languages Average**: Performance on regional Indian languages
- **Language-Specific Scores**: Individual language performance
- **Script Family Analysis**: Performance by writing system
- **Pronunciation Accuracy**: Phonetic authenticity assessment
- **Grammar Naturalness**: Linguistic structure evaluation
- **Conversational Fluency**: Dialogue flow assessment

## Languages Tested

### European Languages (5)
- **English** (Latin script)
- **Spanish** (Latin script)
- **French** (Latin script)
- **German** (Latin script)
- **Italian** (Latin script)

### Indian Regional Languages (10)
- **Hindi** (Devanagari script) - North India
- **Bengali** (Bengali script) - West Bengal, Bangladesh
- **Tamil** (Tamil script) - Tamil Nadu, Sri Lanka
- **Telugu** (Telugu script) - Andhra Pradesh, Telangana
- **Marathi** (Devanagari script) - Maharashtra
- **Gujarati** (Gujarati script) - Gujarat
- **Kannada** (Kannada script) - Karnataka
- **Malayalam** (Malayalam script) - Kerala
- **Punjabi** (Gurmukhi script) - Punjab region
- **Urdu** (Perso-Arabic script) - North India, Pakistan

## Configuration

- **Languages Tested**: 15 total (5 European + 10 Indian regional)
- **Scenarios per Language**: 3 diverse conversation types
- **Interactions per Scenario**: 100 (configurable)
- **Response Timeout**: 20 seconds
- **Analysis Depth**: Comprehensive multilingual evaluation
- **Script Support**: Multiple writing systems (Latin, Devanagari, Bengali, Tamil, Telugu, etc.)

## Usage

```bash
cd naturalness
python main.py
```

Ensure your `.env` file contains:
- `AIVOCO_API_KEY`: Your Aivoco API key
- `VISPARK_API_KEY`: Your Vispark API key

## Interpretation

This benchmark reveals:
- **Global Language Capabilities**: Performance across diverse linguistic families
- **Script Handling**: Ability to process different writing systems
- **Regional Accent Authenticity**: Accuracy of pronunciation for regional variants
- **Cross-Language Consistency**: Performance uniformity across language families
- **Cultural and Linguistic Adaptation**: Appropriateness for different cultural contexts
- **Multilingual Robustness**: Handling of complex scripts and phonetic systems
- **Regional Language Support**: Effectiveness for Indian linguistic diversity

The inclusion of Indian regional languages provides critical insights into:
- **Script Complexity**: Handling of Brahmic scripts vs Latin scripts
- **Phonetic Diversity**: Regional pronunciation variations
- **Cultural Context**: Appropriateness for Indian communication styles
- **Language Technology Maturity**: Readiness for Indian language AI applications
