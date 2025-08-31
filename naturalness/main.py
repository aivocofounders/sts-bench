"""
Naturalness Benchmark for Speech-to-Speech Models.

This benchmark evaluates the naturalness of speech generation across multiple languages
using Vispark Vision as a judge to assess conversational flow and linguistic authenticity.
"""

import asyncio
import json
from typing import Dict, Any, List
from loguru import logger

from ..base.benchmark import AbstractBenchmark
from ..base.config import BenchmarkConfig
from ..base.audio import AudioProcessor

class NaturalnessBenchmark(AbstractBenchmark):
    """Naturalness benchmark with multilingual multimodal judging."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config, "naturalness")
        self.audio_processor = AudioProcessor()

        # Multilingual test scenarios
        self.language_scenarios = [
            # European Languages
            {
                "language": "english",
                "code": "en",
                "scenarios": [
                    "casual conversation about weekend plans",
                    "professional meeting discussion",
                    "storytelling about a personal experience"
                ]
            },
            {
                "language": "spanish",
                "code": "es",
                "scenarios": [
                    "conversación casual sobre planes de fin de semana",
                    "discusión de reunión profesional",
                    "narración de una experiencia personal"
                ]
            },
            {
                "language": "french",
                "code": "fr",
                "scenarios": [
                    "conversation décontractée sur les plans du week-end",
                    "discussion de réunion professionnelle",
                    "récit d'une expérience personnelle"
                ]
            },
            {
                "language": "german",
                "code": "de",
                "scenarios": [
                    "zwangloses Gespräch über Wochenendpläne",
                    "professionelle Besprechungsdiskussion",
                    "Erzählung einer persönlichen Erfahrung"
                ]
            },
            {
                "language": "italian",
                "code": "it",
                "scenarios": [
                    "conversazione informale sui piani del fine settimana",
                    "discussione riunione professionale",
                    "racconto di un'esperienza personale"
                ]
            },
            # Indian Regional Languages
            {
                "language": "hindi",
                "code": "hi",
                "scenarios": [
                    "weekend के प्लान्स के बारे में casual बातचीत",
                    "professional मीटिंग की discussion",
                    "personal experience के बारे में storytelling"
                ]
            },
            {
                "language": "bengali",
                "code": "bn",
                "scenarios": [
                    "weekend এর পরিকল্পনা নিয়ে casual কথাবার্তা",
                    "professional মিটিং এর আলোচনা",
                    "ব্যক্তিগত অভিজ্ঞতা নিয়ে গল্প বলা"
                ]
            },
            {
                "language": "tamil",
                "code": "ta",
                "scenarios": [
                    "வார இறுதி திட்டங்களைப் பற்றிய casual உரையாடல்",
                    "professional கூட்டம் பற்றிய விவாதம்",
                    "தனிப்பட்ட அனுபவம் பற்றிய கதை சொல்லல்"
                ]
            },
            {
                "language": "telugu",
                "code": "te",
                "scenarios": [
                    "వీకెండ్ ప్లాన్స్ గురించి casual సంభాషణ",
                    "professional మీటింగ్ చర్చ",
                    "వ్యక్తిగత అనుభవం గురించి కథ చెప్పడం"
                ]
            },
            {
                "language": "marathi",
                "code": "mr",
                "scenarios": [
                    "weekendच्या प्लॅन्सबद्दल casual गप्पा",
                    "professional मीटिंगची चर्चा",
                    "वैयक्तिक अनुभवाबद्दल कहाणी सांगणे"
                ]
            },
            {
                "language": "gujarati",
                "code": "gu",
                "scenarios": [
                    "weekendના પ્લાન્સ વિશે casual વાતચીત",
                    "professional મીટિંગની ચર્ચા",
                    "વ્યક્તિગત અનુભવ વિશે વાર્તા કહેવી"
                ]
            },
            {
                "language": "kannada",
                "code": "kn",
                "scenarios": [
                    "ವೀಕೆಂಡ್ ಯೋಜನೆಗಳ ಬಗ್ಗೆ casual ಮಾತುಕತೆ",
                    "professional ಸಭೆಯ ಚರ್ಚೆ",
                    "ವೈಯಕ್ತಿಕ ಅನುಭವದ ಬಗ್ಗೆ ಕಥೆ ಹೇಳುವುದು"
                ]
            },
            {
                "language": "malayalam",
                "code": "ml",
                "scenarios": [
                    "വീക്കെൻഡ് പ്ലാനുകൾക്കുറിച്ച് casual സംഭാഷണം",
                    "professional മീറ്റിംഗിന്റെ ചർച്ച",
                    "സ്വകാര്യ അനുഭവം പറയുന്ന കഥ"
                ]
            },
            {
                "language": "punjabi",
                "code": "pa",
                "scenarios": [
                    "weekend ਦੇ ਪਲਾਨਾਂ ਬਾਰੇ casual ਗੱਲਬਾਤ",
                    "professional ਮੀਟਿੰਗ ਦੀ ਚਰਚਾ",
                    "ਨਿੱਜੀ ਤਜ਼ਰਬੇ ਬਾਰੇ ਕਹਾਣੀ ਸੁਣਾਉਣਾ"
                ]
            },
            {
                "language": "urdu",
                "code": "ur",
                "scenarios": [
                    "weekend کے منصوبوں کے بارے میں casual بات چیت",
                    "professional ملاقات کی بحث",
                    "ذاتی تجربے کے بارے میں کہانی سنانا"
                ]
            }
        ]

    async def setup(self) -> bool:
        """Setup the naturalness benchmark."""
        try:
            logger.info("Setting up naturalness benchmark")
            # Test connections - same as emotions benchmark
            vispark_client = self.get_model('vispark')
            test_result = await vispark_client.vision_analyze([
                {"type": "text", "content": "Test analysis"}
            ])
            return test_result.get("status") == "success"
        except Exception as e:
            logger.error(f"Naturalness benchmark setup failed: {e}")
            return False

    async def run_language_interaction(self, language_data: Dict[str, Any], scenario: str, interaction_id: int) -> Dict[str, Any]:
        """Run a single multilingual interaction."""
        try:
            language = language_data["language"]
            lang_code = language_data["code"]

            vispark_client = self.get_model('vispark')
            aivoco_client = self.get_model('aivoco')

            # Generate natural conversation prompt
            prompt = f"Engage in a natural {language} conversation about: {scenario}"

            # TTS with language specification
            tts_result = await vispark_client.text_to_speech(
                prompt,
                voice=self.config.tts_voice,
                size=self.config.tts_size
            )

            if tts_result.get("status") != "success":
                return {"success": False, "error": "TTS failed"}

            # Start session and get response
            session_result = await aivoco_client.start_call(
                system_message=f"You are having a natural conversation in {language}. Respond authentically and fluently."
            )

            if session_result.get("status") != "success":
                return {"success": False, "error": "Session failed"}

            audio_result = await aivoco_client.send_audio_data(tts_result["data"]["content"])
            if audio_result.get("status") != "success":
                await aivoco_client.stop_call()
                return {"success": False, "error": "Audio send failed"}

            response = await self.run_with_timeout(aivoco_client.receive_audio_response(), timeout=20.0)
            if not response:
                await aivoco_client.stop_call()
                return {"success": False, "error": "No response"}

            await aivoco_client.stop_call()

            # Analyze naturalness using Vispark Vision
            naturalness_prompt = f"""
            Analyze the naturalness of this {language} speech response on a scale of 1-10:

            1. Fluency: How smooth and natural does the speech flow?
            2. Pronunciation: How authentic does the accent and pronunciation sound?
            3. Grammar: How correct and natural is the linguistic structure?
            4. Conversational flow: How natural does the back-and-forth feel?

            Provide an overall naturalness score (0-100%) and detailed feedback.
            """

            analysis_result = await vispark_client.vision_analyze([
                {"type": "text", "content": naturalness_prompt},
                {"type": "audio", "content": response["audio_data"]}
            ])

            naturalness_score = 50.0  # Default
            if analysis_result.get("status") == "success":
                naturalness_score = self.parse_naturalness_score(analysis_result["data"]["content"])

            self.metrics.record_audio_quality(naturalness_score)

            return {
                "interaction_id": interaction_id,
                "language": language,
                "scenario": scenario,
                "success": True,
                "naturalness_score": naturalness_score,
                "analysis": analysis_result["data"]["content"] if analysis_result.get("status") == "success" else ""
            }

        except Exception as e:
            logger.error(f"Naturalness interaction failed: {e}")
            return {"success": False, "error": str(e)}

    def parse_naturalness_score(self, analysis_text: str) -> float:
        """Parse naturalness score from analysis text."""
        import re
        # Similar to emotion parsing
        percent_match = re.search(r'(\d+(?:\.\d+)?)%', analysis_text)
        if percent_match:
            return float(percent_match.group(1))
        return 50.0

    async def run_benchmark(self) -> 'BenchmarkMetrics':
        """Run the naturalness benchmark."""
        logger.info(f"Starting naturalness benchmark")

        results = []

        for language_data in self.language_scenarios:
            language = language_data["language"]
            scenarios = language_data["scenarios"]

            logger.info(f"Testing language: {language}")

            for scenario in scenarios:
                with self.create_progress_bar(self.config.multimodal_iterations, f"{language.title()} - {scenario[:30]}...") as pbar:
                    for i in range(self.config.multimodal_iterations):
                        result = await self.run_language_interaction(language_data, scenario, len(results))
                        results.append(result)
                        pbar.update(1)

        # Calculate metrics
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            avg_naturalness = sum(r["naturalness_score"] for r in successful_results) / len(successful_results)
            self.metrics.add_custom_metric("avg_naturalness_score", avg_naturalness)

        logger.info("Naturalness benchmark completed")
        return self.metrics

    async def cleanup(self) -> None:
        """Cleanup naturalness benchmark resources."""
        logger.info("Cleaning up naturalness benchmark")

async def main():
    """Main entry point for naturalness benchmark."""
    try:
        config = BenchmarkConfig.from_env()
        benchmark = NaturalnessBenchmark(config)
        metrics = await benchmark.execute()
        logger.info(f"Naturalness benchmark completed. Average naturalness: {metrics.average_audio_quality:.2f}%")
    except Exception as e:
        logger.error(f"Naturalness benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
