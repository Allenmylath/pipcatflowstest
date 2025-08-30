#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Complete Scored Quiz Bot Example

This example demonstrates how to create a bot that conducts scored quizzes using JSON configuration:
- Asks multiple choice questions with scored options
- Tracks total score in FlowManager state
- Provides results based on score ranges
- Supports any number of options per question (A, B, C, D, etc.)

Requirements:
- Daily room URL and API key
- OpenAI API key
- Deepgram API key
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict

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

from pipecat_flows import FlowArgs, FlowManager, FlowResult

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Example configuration - Skills Assessment Quiz
SKILLS_ASSESSMENT_CONFIG = {
    "main_prompt": """You are Alex, a professional skills assessment coordinator. 
    You conduct evaluations to help people understand their strengths and areas for development. 
    Be encouraging, professional, and clear in your communication. 
    
    IMPORTANT - TTS Formatting Rules:
    - End every sentence with proper punctuation (. ! ?)
    - Use commas for natural pauses and before names
    - Add exclamation points for enthusiasm and encouragement
    - Use hyphens for additional pauses when needed
    - Keep sentences conversational and not too long
    - Use direct address with commas: "Great job, Sarah!"
    - Add natural fillers like "Let me see..." or "One moment..."
    - Use quotes around specific instructions: Say "A" or "option A"
    This is a voice conversation that will be converted to speech.""",
    "greeting": """Hello there! I'm Alex from the Skills Assessment team. 

Today I'll be conducting a brief evaluation to help identify your professional strengths - and areas for development. Exciting, right?

This assessment has 6 questions, each with multiple choice options. There are no right or wrong answers - just choose the option that best describes your typical approach or preference. 

Each answer has different point values, and your total score will help us provide personalized feedback tailored just for you. 

Are you ready to begin? Let's dive in!""",
    "questions": [
        {
            "id": "problem_solving",
            "question": "When faced with a complex problem at work, your first instinct is to:",
            "options": [
                {
                    "label": "A",
                    "text": "Break it down into smaller, manageable parts",
                    "score": 15,
                },
                {
                    "label": "B",
                    "text": "Research similar problems and solutions",
                    "score": 12,
                },
                {
                    "label": "C",
                    "text": "Brainstorm with colleagues for different perspectives",
                    "score": 10,
                },
                {
                    "label": "D",
                    "text": "Dive in and learn by trial and error",
                    "score": 8,
                },
                {
                    "label": "E",
                    "text": "Seek guidance from a supervisor or expert",
                    "score": 5,
                },
            ],
        },
        {
            "id": "time_management",
            "question": "How do you typically handle competing priorities and deadlines?",
            "options": [
                {
                    "label": "A",
                    "text": "Create detailed schedules and stick to them religiously",
                    "score": 15,
                },
                {
                    "label": "B",
                    "text": "Prioritize based on importance and urgency",
                    "score": 13,
                },
                {
                    "label": "C",
                    "text": "Focus on what interests me most first",
                    "score": 7,
                },
                {
                    "label": "D",
                    "text": "Work on whatever seems most urgent at the moment",
                    "score": 9,
                },
            ],
        },
        {
            "id": "leadership_style",
            "question": "In a team setting, you naturally tend to:",
            "options": [
                {
                    "label": "A",
                    "text": "Take charge and coordinate everyone's efforts",
                    "score": 15,
                },
                {
                    "label": "B",
                    "text": "Support others and help resolve conflicts",
                    "score": 12,
                },
                {
                    "label": "C",
                    "text": "Contribute your expertise when asked",
                    "score": 10,
                },
                {
                    "label": "D",
                    "text": "Follow directions and complete assigned tasks",
                    "score": 8,
                },
            ],
        },
        {
            "id": "learning_approach",
            "question": "When learning a new skill or technology, you prefer to:",
            "options": [
                {
                    "label": "A",
                    "text": "Take a structured course or formal training",
                    "score": 12,
                },
                {
                    "label": "B",
                    "text": "Learn through hands-on practice and experimentation",
                    "score": 15,
                },
                {
                    "label": "C",
                    "text": "Watch others and learn from observation",
                    "score": 9,
                },
                {
                    "label": "D",
                    "text": "Read documentation and study independently",
                    "score": 11,
                },
                {
                    "label": "E",
                    "text": "Find a mentor or expert to guide you",
                    "score": 8,
                },
            ],
        },
        {
            "id": "communication_preference",
            "question": "Your preferred method for sharing important information with your team is:",
            "options": [
                {
                    "label": "A",
                    "text": "Detailed written reports with supporting data",
                    "score": 12,
                },
                {
                    "label": "B",
                    "text": "Interactive presentations with visual aids",
                    "score": 15,
                },
                {
                    "label": "C",
                    "text": "One-on-one discussions with key stakeholders",
                    "score": 11,
                },
                {
                    "label": "D",
                    "text": "Quick informal updates in person or via chat",
                    "score": 8,
                },
            ],
        },
        {
            "id": "innovation_mindset",
            "question": "When it comes to trying new approaches or technologies:",
            "options": [
                {
                    "label": "A",
                    "text": "I actively seek out and champion innovative solutions",
                    "score": 15,
                },
                {
                    "label": "B",
                    "text": "I'm open to change when I see clear benefits",
                    "score": 12,
                },
                {
                    "label": "C",
                    "text": "I prefer to let others test things first, then adopt what works",
                    "score": 9,
                },
                {
                    "label": "D",
                    "text": "I'm most comfortable sticking with proven methods",
                    "score": 6,
                },
            ],
        },
    ],
    "scoring": {
        "ranges": [
            {
                "min": 30,
                "max": 50,
                "result": "Developing Professional: You're building foundational skills and prefer structured support. Focus on developing systematic approaches to problem-solving and taking on more leadership opportunities. Consider seeking mentorship and formal training programs.",
            },
            {
                "min": 51,
                "max": 70,
                "result": "Competent Contributor: You have solid professional skills and work well in team environments. You balance independence with collaboration effectively. Consider expanding your influence by mentoring others and taking on more complex challenges.",
            },
            {
                "min": 71,
                "max": 85,
                "result": "Strong Performer: You demonstrate advanced professional capabilities with excellent problem-solving and communication skills. You're ready for leadership roles and should consider strategic projects that leverage your strengths while developing emerging leaders.",
            },
            {
                "min": 86,
                "max": 90,
                "result": "High Achiever: You exhibit exceptional professional skills across multiple domains. You're a natural leader who drives innovation and excellence. Consider executive development programs and opportunities to shape organizational strategy and culture.",
            },
        ]
    },
    "completion_message": "Fantastic! You've completed the skills assessment. Your results will help identify your professional strengths - and suggest exciting development opportunities tailored specifically to your unique profile!",
}


class ScoredQuizBuilder:
    """Builds FlowManager configurations for scored quiz/survey systems."""

    def __init__(self, json_config: Dict):
        self.config = json_config
        self._validate_config()
        self._questions = self.config["questions"]
        self._main_prompt = self.config.get(
            "main_prompt", "You are a helpful quiz assistant."
        )
        self._greeting = self.config.get("greeting", "Welcome to the quiz!")
        self._scoring = self.config.get("scoring", {})
        self._completion_message = self.config.get(
            "completion_message", "Thank you for completing the quiz!"
        )

        logger.debug(
            f"Initialized Scored Quiz Builder with {len(self._questions)} questions"
        )

    def _validate_config(self):
        """Validate the JSON configuration."""
        if "questions" not in self.config:
            raise ValueError("Missing required field: questions")
        if not isinstance(self.config["questions"], list):
            raise ValueError("questions must be a list")
        if len(self.config["questions"]) == 0:
            raise ValueError("Must have at least one question")

    def get_node_count(self) -> int:
        """Calculate total number of nodes that will be created."""
        # greeting + one per question + results
        return 2 + len(self._questions)

    def get_max_possible_score(self) -> int:
        """Calculate the maximum possible score."""
        max_score = 0
        for question in self._questions:
            question_max = max(option["score"] for option in question["options"])
            max_score += question_max
        return max_score

    def get_min_possible_score(self) -> int:
        """Calculate the minimum possible score."""
        min_score = 0
        for question in self._questions:
            question_min = min(option["score"] for option in question["options"])
            min_score += question_min
        return min_score

    def build_flow_config(self):
        """Build a FlowConfig from the JSON configuration."""
        nodes = {}

        # Create greeting node
        nodes["greeting"] = {
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        f"{self._main_prompt} "
                        "Always use the available functions to progress through the quiz. "
                        "Format your speech naturally with proper punctuation for TTS. "
                        "Read questions and options clearly, then wait for user responses. "
                        "Be encouraging and use exclamation points for enthusiasm!"
                    ),
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": (
                        f"{self._greeting} "
                        "After your greeting, use the start_quiz function to begin the assessment. "
                        "Make sure to speak naturally with proper punctuation - commas for pauses, "
                        "periods to end sentences, and exclamation points for enthusiasm!"
                    ),
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "start_quiz",
                        "handler": "__function__:start_quiz",
                        "description": "Begin the scored quiz",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "question_0" if self._questions else "results",
                    },
                }
            ],
        }

        # Create a node for each question
        for i, question in enumerate(self._questions):
            next_node = (
                f"question_{i + 1}" if i + 1 < len(self._questions) else "results"
            )

            # Create options display for the prompt
            options_text = []
            for option in question["options"]:
                options_text.append(f"{option['label']}: {option['text']}")
            options_display = "\n".join(options_text)

            nodes[f"question_{i}"] = {
                "task_messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Great! Let's move to question {i + 1} of {len(self._questions)}.\n\n"
                            f"Here's your question: '{question['question']}'\n\n"
                            f"Your options are:\n{options_display}\n\n"
                            "Please read the question clearly, then present each option with natural pauses. "
                            "Use proper TTS formatting: commas for pauses, periods to end sentences. "
                            "After presenting all options, say something like: \"Which option sounds most like you? "
                            "You can say the letter - like 'A' or 'B' - or describe your choice.\" "
                            "Wait for their response, then use the record_answer function. "
                            "Be encouraging throughout!"
                        ),
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": f"record_answer_{i}",
                            "handler": f"__function__:record_answer_{i}",
                            "description": f"Record answer for question {i + 1}",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "selected_option": {
                                        "type": "string",
                                        "description": f"User's choice from: {', '.join([opt['label'] for opt in question['options']])}",
                                    }
                                },
                                "required": ["selected_option"],
                            },
                            "transition_to": next_node,
                        },
                    }
                ],
            }

        # Create results node
        nodes["results"] = {
            "task_messages": [
                {
                    "role": "system",
                    "content": (
                        "Excellent! You've completed all the questions. "
                        "Now it's time to calculate and present your final results! "
                        "Use the show_results function to display the total score and interpretation. "
                        "Be very encouraging and enthusiastic - use exclamation points! "
                        "Format your response with proper punctuation: commas for pauses, "
                        "periods for sentence endings. Make it sound celebratory and personal. "
                        "Address them directly and explain what their score means for their professional development."
                    ),
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "show_results",
                        "handler": "__function__:show_results",
                        "description": "Show final quiz results and score interpretation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "post_actions": [{"type": "end_conversation"}],
        }

        return {"initial_node": "greeting", "nodes": nodes}

    def register_handlers_in_module(self, module):
        """Register handler functions in the specified module."""

        # Start quiz handler
        async def start_quiz(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
            logger.info("=== Starting scored quiz ===")
            logger.info(f"Questions: {len(self._questions)}")
            logger.info(
                f"Score range: {self.get_min_possible_score()} - {self.get_max_possible_score()}"
            )

            # Initialize score tracking
            flow_manager.state["total_score"] = 0
            flow_manager.state["answers"] = []
            flow_manager.state["max_possible_score"] = self.get_max_possible_score()

            return {"status": "success", "message": "Quiz started", "current_score": 0}

        setattr(module, "start_quiz", start_quiz)

        # Create answer recording handlers for each question
        for i, question in enumerate(self._questions):
            question_id = question["id"]
            question_text = question["question"]
            options = question["options"]

            async def create_recorder(
                q_id=question_id, q_text=question_text, q_options=options, q_index=i
            ):
                async def record_answer(
                    args: FlowArgs, flow_manager: FlowManager
                ) -> FlowResult:
                    selected_option = args.get("selected_option", "").strip().upper()

                    # Find the selected option by label or text matching
                    selected_opt = None
                    for option in q_options:
                        # Check if user said the letter
                        if selected_option == option["label"].upper():
                            selected_opt = option
                            break
                        # Check if user said part of the option text
                        elif len(selected_option) > 2 and (
                            selected_option.lower() in option["text"].lower()
                            or option["text"].lower() in selected_option.lower()
                        ):
                            selected_opt = option
                            break

                    # If no match found, try to find by first letter or default to first option
                    if not selected_opt:
                        # Try matching just the first character
                        for option in q_options:
                            if selected_option.startswith(option["label"].upper()):
                                selected_opt = option
                                break

                        # Still no match? Default to first option
                        if not selected_opt:
                            selected_opt = q_options[0]
                            logger.warning(
                                f"Could not match '{selected_option}', defaulting to option {selected_opt['label']}"
                            )

                    # Update total score
                    current_score = flow_manager.state.get("total_score", 0)
                    new_score = current_score + selected_opt["score"]
                    flow_manager.state["total_score"] = new_score

                    # Store individual answer
                    flow_manager.state["answers"].append(
                        {
                            "question_id": q_id,
                            "question_text": q_text,
                            "question_index": q_index,
                            "selected_option": selected_opt["label"],
                            "option_text": selected_opt["text"],
                            "points_earned": selected_opt["score"],
                        }
                    )

                    logger.info(f"=== Question {q_index + 1} Results ===")
                    logger.info(f"Question: {q_text}")
                    logger.info(
                        f"Selected: {selected_opt['label']} - {selected_opt['text']}"
                    )
                    logger.info(f"Points earned: {selected_opt['score']}")
                    logger.info(
                        f"Running total: {new_score}/{flow_manager.state.get('max_possible_score', 0)}"
                    )

                    return {
                        "status": "success",
                        "question_id": q_id,
                        "selected_option": selected_opt["label"],
                        "option_text": selected_opt["text"],
                        "points_earned": selected_opt["score"],
                        "total_score": new_score,
                        "progress": f"{q_index + 1}/{len(self._questions)}",
                    }

                return record_answer

            setattr(module, f"record_answer_{i}", asyncio.run(create_recorder()))

        # Show results handler
        async def show_results(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
            total_score = flow_manager.state.get("total_score", 0)
            answers = flow_manager.state.get("answers", [])
            max_score = flow_manager.state.get("max_possible_score", 0)
            percentage = (
                round((total_score / max_score) * 100, 1) if max_score > 0 else 0
            )

            logger.info("=== QUIZ COMPLETED ===")
            logger.info(f"Final Score: {total_score}/{max_score} ({percentage}%)")

            # Determine result category based on scoring ranges
            result_message = self._completion_message
            if "ranges" in self._scoring:
                for score_range in self._scoring["ranges"]:
                    if score_range["min"] <= total_score <= score_range["max"]:
                        result_message = score_range["result"]
                        break

            logger.info(f"Result Category: {result_message[:100]}...")
            logger.info("\n=== Individual Answers ===")
            for answer in answers:
                logger.info(
                    f"Q{answer['question_index'] + 1}: {answer['selected_option']} ({answer['points_earned']} pts)"
                )
                logger.info(f"   {answer['option_text']}")

            # Store final results in state for potential export
            flow_manager.state["final_results"] = {
                "total_score": total_score,
                "max_score": max_score,
                "percentage": percentage,
                "result_message": result_message,
                "completed_at": asyncio.get_event_loop().time(),
            }

            return {
                "status": "completed",
                "total_score": total_score,
                "max_possible_score": max_score,
                "percentage": percentage,
                "result_message": result_message,
                "answers": answers,
                "quiz_completed": True,
            }

        setattr(module, "show_results", show_results)


# Create the quiz builder and register handlers
quiz_builder = ScoredQuizBuilder(SKILLS_ASSESSMENT_CONFIG)
quiz_builder.register_handlers_in_module(sys.modules[__name__])


async def main():
    """Main function to set up and run the scored quiz bot."""

    logger.info("=== Skills Assessment Quiz Bot ===")
    logger.info(f"Total questions: {len(SKILLS_ASSESSMENT_CONFIG['questions'])}")
    logger.info(f"Total nodes to create: {quiz_builder.get_node_count()}")
    logger.info(
        f"Score range: {quiz_builder.get_min_possible_score()} - {quiz_builder.get_max_possible_score()}"
    )

    logger.info("\n=== Questions Overview ===")
    for i, q in enumerate(SKILLS_ASSESSMENT_CONFIG["questions"]):
        logger.info(f"Q{i+1}: {q['question']}")
        option_scores = [f"{opt['label']}({opt['score']})" for opt in q["options"]]
        logger.info(f"     Options: {', '.join(option_scores)}")

    async with aiohttp.ClientSession() as session:
        # Configure Daily
        room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")
        if not room_url:
            logger.error("Please set DAILY_SAMPLE_ROOM_URL environment variable")
            return

        # Initialize services
        transport = DailyTransport(
            room_url,
            None,
            "Skills Assessment Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice="aura-asteria-en",  # Professional, clear voice
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        # Create pipeline
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        # Build flow configuration
        flow_config = quiz_builder.build_flow_config()

        logger.info(f"\nCreated flow with nodes: {list(flow_config['nodes'].keys())}")

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            tts=tts,
            flow_config=flow_config,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            logger.info("=== Skills Assessment Bot Ready ===")
            logger.info(
                "Bot will conduct scored assessment and track results in FlowManager state"
            )

            # Initialize the flow
            await flow_manager.initialize()

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
