"""
Scored Quiz Builder Class

This module provides the ScoredQuizBuilder class that converts JSON quiz configurations
into FlowManager configurations for Pipecat-based conversational AI applications.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from pipecat_flows import FlowArgs, FlowManager, FlowResult


class ScoredQuizBuilder:
    """
    Builds FlowManager configurations for scored quiz/survey systems from JSON configuration.

    This class takes a JSON configuration defining questions, scoring, and prompts,
    and generates the corresponding flow nodes and handler functions for a conversational AI quiz bot.
    """

    def __init__(self, json_config: Union[Dict, str, Path]):
        """
        Initialize the ScoredQuizBuilder with a quiz configuration.

        Args:
            json_config: Either a dictionary with the quiz config,
                        a JSON string, or a Path to a JSON file
        """
        if isinstance(json_config, (str, Path)):
            self.config = self._load_json_config(json_config)
        else:
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

    def _load_json_config(self, config_path: Union[str, Path]) -> Dict:
        """Load JSON configuration from file or string."""
        if isinstance(config_path, Path) or (
            isinstance(config_path, str) and config_path.endswith(".json")
        ):
            # It's a file path
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Quiz configuration file not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # It's a JSON string
            try:
                return json.loads(config_path)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string provided: {e}")

    def _validate_config(self):
        """Validate the JSON configuration structure and required fields."""
        required_fields = ["questions"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(self.config["questions"], list):
            raise ValueError("questions must be a list")

        if len(self.config["questions"]) == 0:
            raise ValueError("Must have at least one question")

        # Validate each question structure
        for i, question in enumerate(self.config["questions"]):
            if "id" not in question:
                raise ValueError(f"Question {i} missing required field: id")
            if "question" not in question:
                raise ValueError(f"Question {i} missing required field: question")
            if "options" not in question or not isinstance(question["options"], list):
                raise ValueError(f"Question {i} must have options as a list")
            if len(question["options"]) == 0:
                raise ValueError(f"Question {i} must have at least one option")

            # Validate options
            for j, option in enumerate(question["options"]):
                required_option_fields = ["label", "text", "score"]
                for field in required_option_fields:
                    if field not in option:
                        raise ValueError(
                            f"Question {i}, option {j} missing required field: {field}"
                        )

    def get_node_count(self) -> int:
        """Calculate total number of nodes that will be created."""
        return 2 + len(self._questions)  # greeting + questions + results

    def get_max_possible_score(self) -> int:
        """Calculate the maximum possible score across all questions."""
        max_score = 0
        for question in self._questions:
            question_max = max(option["score"] for option in question["options"])
            max_score += question_max
        return max_score

    def get_min_possible_score(self) -> int:
        """Calculate the minimum possible score across all questions."""
        min_score = 0
        for question in self._questions:
            question_min = min(option["score"] for option in question["options"])
            min_score += question_min
        return min_score

    def get_score_range(self) -> tuple[int, int]:
        """Get the (min, max) score range."""
        return self.get_min_possible_score(), self.get_max_possible_score()

    def get_questions(self) -> List[Dict]:
        """Get the list of questions."""
        return self._questions.copy()

    def get_scoring_ranges(self) -> List[Dict]:
        """Get the scoring ranges if defined."""
        return self._scoring.get("ranges", [])

    def build_flow_config(self) -> Dict[str, Any]:
        """
        Build a FlowConfig dictionary from the JSON configuration.

        Returns:
            A dictionary containing the flow configuration with nodes and initial state
        """
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
        """
        Register all necessary handler functions in the specified module.

        Args:
            module: The module object where handlers will be registered (typically sys.modules[__name__])
        """

        # Start quiz handler
        async def start_quiz(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
            logger.info("=== Starting scored quiz ===")
            logger.info(f"Questions: {len(self._questions)}")
            logger.info(
                f"Score range: {self.get_min_possible_score()} - {self.get_max_possible_score()}"
            )

            # Initialize score tracking in FlowManager state
            flow_manager.state["total_score"] = 0
            flow_manager.state["answers"] = []
            flow_manager.state["max_possible_score"] = self.get_max_possible_score()
            flow_manager.state["quiz_config"] = {
                "total_questions": len(self._questions),
                "completion_message": self._completion_message,
                "scoring_ranges": self.get_scoring_ranges(),
            }

            return {"status": "success", "message": "Quiz started", "current_score": 0}

        setattr(module, "start_quiz", start_quiz)

        # Create answer recording handlers for each question
        for i, question in enumerate(self._questions):
            question_id = question["id"]
            question_text = question["question"]
            options = question["options"]

            # Use closure to capture variables for each question
            def create_recorder(
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
                        for option in q_options:
                            if selected_option.startswith(option["label"].upper()):
                                selected_opt = option
                                break

                        # Still no match? Default to first option with warning
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

            # Register the handler function
            setattr(module, f"record_answer_{i}", create_recorder())

        # Show results handler
        async def show_results(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
            total_score = flow_manager.state.get("total_score", 0)
            answers = flow_manager.state.get("answers", [])
            max_score = flow_manager.state.get("max_possible_score", 0)
            quiz_config = flow_manager.state.get("quiz_config", {})

            percentage = (
                round((total_score / max_score) * 100, 1) if max_score > 0 else 0
            )

            logger.info("=== QUIZ COMPLETED ===")
            logger.info(f"Final Score: {total_score}/{max_score} ({percentage}%)")

            # Determine result category based on scoring ranges
            result_message = quiz_config.get(
                "completion_message", self._completion_message
            )
            scoring_ranges = quiz_config.get(
                "scoring_ranges", self.get_scoring_ranges()
            )

            for score_range in scoring_ranges:
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

            # Store final results in state
            flow_manager.state["final_results"] = {
                "total_score": total_score,
                "max_score": max_score,
                "percentage": percentage,
                "result_message": result_message,
                "completed_at": asyncio.get_event_loop().time(),
                "answers_summary": answers,
            }

            # MANDATORY: Export results to JSON automatically
            try:
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(f"quiz_results_{timestamp}.json")

                exported_results = self.export_results_to_json(
                    flow_manager, output_file
                )

                logger.info(
                    f"✅ MANDATORY EXPORT COMPLETED: Results saved to {output_file}"
                )

                # Also store export info in the flow state
                flow_manager.state["export_info"] = {
                    "exported_at": asyncio.get_event_loop().time(),
                    "export_file": str(output_file),
                    "export_successful": True,
                }

            except Exception as e:
                logger.error(f"❌ MANDATORY EXPORT FAILED: {e}")
                flow_manager.state["export_info"] = {
                    "exported_at": asyncio.get_event_loop().time(),
                    "export_file": None,
                    "export_successful": False,
                    "export_error": str(e),
                }
                # Re-raise the exception to ensure the failure is noticed
                raise RuntimeError(f"Mandatory JSON export failed: {e}") from e

            return {
                "status": "completed",
                "total_score": total_score,
                "max_possible_score": max_score,
                "percentage": percentage,
                "result_message": result_message,
                "answers": answers,
                "quiz_completed": True,
                "exported_to": str(output_file) if "output_file" in locals() else None,
            }

        setattr(module, "show_results", show_results)

    def export_results_to_json(
        self, flow_manager: FlowManager, output_path: Optional[Path] = None
    ) -> Dict:
        """
        Export quiz results to JSON format.

        Args:
            flow_manager: The FlowManager instance containing the quiz state
            output_path: Optional path to save the results as JSON file

        Returns:
            Dictionary containing the complete quiz results
        """
        results = {
            "quiz_metadata": {
                "total_questions": len(self._questions),
                "score_range": self.get_score_range(),
                "completion_time": flow_manager.state.get("final_results", {}).get(
                    "completed_at"
                ),
            },
            "final_results": flow_manager.state.get("final_results", {}),
            "detailed_answers": flow_manager.state.get("answers", []),
            "quiz_config_snapshot": {
                "questions": [
                    {"id": q["id"], "question": q["question"]} for q in self._questions
                ],
                "scoring_ranges": self.get_scoring_ranges(),
            },
        }

        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Quiz results exported to: {output_path}")

        return results

    def __repr__(self) -> str:
        return (
            f"ScoredQuizBuilder(questions={len(self._questions)}, "
            f"score_range={self.get_score_range()})"
        )
