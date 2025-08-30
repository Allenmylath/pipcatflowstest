#!/usr/bin/env python3
"""
Skills Assessment Quiz Bot

A conversational AI bot that conducts scored assessments using the Daily platform
and Pipecat framework. This bot loads quiz configuration from JSON and uses the
ScoredQuizBuilder class to create dynamic conversation flows.

Requirements:
- Daily room URL and API key
- OpenAI API key
- Deepgram API key

Usage:
    python bot.py
"""

import asyncio
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowManager
from scored_quiz_builder import ScoredQuizBuilder

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class QuizBot:
    """
    Main bot class that orchestrates the quiz conversation flow.
    """

    def __init__(self, config_path: str = "quiz_config.json"):
        """
        Initialize the quiz bot with configuration.

        Args:
            config_path: Path to the JSON quiz configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Quiz configuration not found: {self.config_path}")

        # Initialize quiz builder
        self.quiz_builder = ScoredQuizBuilder(self.config_path)

        # Register handlers in this module's namespace
        self.quiz_builder.register_handlers_in_module(sys.modules[__name__])

        # Bot configuration
        self.room_url = None
        self.flow_manager = None

        logger.info(f"QuizBot initialized with config: {self.config_path}")
        logger.info(f"Quiz contains {len(self.quiz_builder.get_questions())} questions")
        logger.info(f"Score range: {self.quiz_builder.get_score_range()}")

    def validate_environment(self) -> bool:
        """
        Validate that all required environment variables are set.

        Returns:
            True if all required variables are present, False otherwise
        """
        required_vars = {
            "DAILY_SAMPLE_ROOM_URL": "Daily room URL for voice/video communication",
            "OPENAI_API_KEY": "OpenAI API key for LLM services",
            "DEEPGRAM_API_KEY": "Deepgram API key for STT/TTS services",
        }

        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"  {var}: {description}")

        if missing_vars:
            logger.error("Missing required environment variables:")
            for var in missing_vars:
                logger.error(var)
            return False

        return True

    def log_quiz_overview(self):
        """Log an overview of the loaded quiz configuration."""
        logger.info("=== Quiz Overview ===")

        questions = self.quiz_builder.get_questions()
        for i, q in enumerate(questions):
            logger.info(f"Q{i+1}: {q['question']}")
            option_scores = [f"{opt['label']}({opt['score']})" for opt in q["options"]]
            logger.info(f"     Options: {', '.join(option_scores)}")

        scoring_ranges = self.quiz_builder.get_scoring_ranges()
        if scoring_ranges:
            logger.info("\n=== Scoring Ranges ===")
            for range_info in scoring_ranges:
                logger.info(
                    f"{range_info['min']}-{range_info['max']}: {range_info['result'][:50]}..."
                )

    async def setup_services(self) -> tuple:
        """
        Set up and configure all the AI services needed for the bot.

        Returns:
            Tuple of (transport, stt, tts, llm, context_aggregator)
        """
        # Get Daily room URL
        self.room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")
        if not self.room_url:
            raise ValueError("DAILY_SAMPLE_ROOM_URL environment variable is required")

        # Initialize Daily transport
        transport = DailyTransport(
            self.room_url,
            None,  # No token needed for sample rooms
            "Skills Assessment Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        # Initialize speech-to-text service
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # Initialize text-to-speech service with professional voice
        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice="aura-asteria-en",  # Professional, clear voice for assessments
        )

        # Initialize language model service
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Set up context management
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        return transport, stt, tts, llm, context_aggregator

    async def create_pipeline(
        self, transport, stt, tts, llm, context_aggregator
    ) -> PipelineTask:
        """
        Create the conversation pipeline with all components.

        Args:
            transport: Daily transport for audio/video
            stt: Speech-to-text service
            tts: Text-to-speech service
            llm: Language model service
            context_aggregator: Context management aggregator

        Returns:
            Configured PipelineTask
        """
        # Create the processing pipeline
        pipeline = Pipeline(
            [
                transport.input(),  # Receive audio from Daily
                stt,  # Convert speech to text
                context_aggregator.user(),  # Add user message to context
                llm,  # Process with language model
                tts,  # Convert response to speech
                transport.output(),  # Send audio to Daily
                context_aggregator.assistant(),  # Add assistant response to context
            ]
        )

        # Create pipeline task with interruption support
        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        return task

    async def initialize_flow_manager(
        self, task, llm, context_aggregator, tts
    ) -> FlowManager:
        """
        Initialize the FlowManager with the quiz configuration.

        Args:
            task: The pipeline task
            llm: Language model service
            context_aggregator: Context aggregator
            tts: Text-to-speech service

        Returns:
            Initialized FlowManager
        """
        # Build flow configuration from quiz JSON
        flow_config = self.quiz_builder.build_flow_config()

        logger.info(f"Created flow with nodes: {list(flow_config['nodes'].keys())}")

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            tts=tts,
            flow_config=flow_config,
        )

        self.flow_manager = flow_manager
        return flow_manager

    async def run(self):
        """
        Main method to run the quiz bot.
        """
        logger.info("=== Starting Skills Assessment Quiz Bot ===")

        # Validate environment
        if not self.validate_environment():
            return

        # Log quiz details
        self.log_quiz_overview()

        try:
            async with aiohttp.ClientSession() as session:
                # Set up services
                (
                    transport,
                    stt,
                    tts,
                    llm,
                    context_aggregator,
                ) = await self.setup_services()

                # Create pipeline
                task = await self.create_pipeline(
                    transport, stt, tts, llm, context_aggregator
                )

                # Initialize flow manager
                flow_manager = await self.initialize_flow_manager(
                    task, llm, context_aggregator, tts
                )

                # Set up event handlers
                @transport.event_handler("on_first_participant_joined")
                async def on_first_participant_joined(transport, participant):
                    await transport.capture_participant_transcription(participant["id"])
                    logger.info("=== Skills Assessment Bot Ready ===")
                    logger.info("Participant joined - starting conversation flow")

                    # Initialize the conversation flow
                    await flow_manager.initialize()

                @transport.event_handler("on_participant_left")
                async def on_participant_left(transport, participant, reason):
                    logger.info(
                        f"Participant left: {participant.get('info', {}).get('userName', 'Unknown')} - {reason}"
                    )

                    # Check if quiz was completed (results are automatically exported in show_results)
                    if self.flow_manager and "final_results" in self.flow_manager.state:
                        export_info = self.flow_manager.state.get("export_info", {})
                        if export_info.get("export_successful"):
                            logger.info(
                                f"✅ Quiz results were automatically exported to: {export_info.get('export_file')}"
                            )
                        else:
                            logger.warning(
                                f"❌ Quiz export failed: {export_info.get('export_error', 'Unknown error')}"
                            )
                    else:
                        logger.info("Quiz was not completed - no results to export")

                # Run the pipeline
                runner = PipelineRunner()
                await runner.run(task)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot encountered an error: {e}")
            raise
        finally:
            logger.info("Quiz bot shutting down")


async def main():
    """
    Entry point for the quiz bot application.
    """
    # You can specify a different config file here if needed
    config_file = "quiz_config.json"

    # Check if custom config file is provided as command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    try:
        # Create and run the bot
        bot = QuizBot(config_file)
        await bot.run()

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        logger.info(
            "Please ensure quiz_config.json exists or provide a valid config file path"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
