"""
Enhanced Scored Quiz Builder Class with RTVI Integration

This module provides the ScoredQuizBuilder class that converts JSON quiz configurations
into FlowManager configurations for Pipecat-based conversational AI applications.
Now includes RTVI server message integration for structured client communication.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from pipecat_flows import FlowArgs, FlowManager, FlowResult
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame


class ScoredQuizBuilder:
    """
    Builds FlowManager configurations for scored quiz/survey systems from JSON configuration.
    Enhanced with RTVI server message integration for structured client communication.

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

    def _create_question_server_message(self, question_index: int) -> Dict[str, Any]:
        """
        Create the server message data for a specific question.

        Args:
            question_index: Index of the question (0-based)

        Returns:
            Dictionary containing the question data for RTVI server message
        """
        if question_index >= len(self._questions):
            return None

        question = self._questions[question_index]
        total_questions = len(self._questions)

        # Format options for client
        formatted_options = []
        for option in question["options"]:
            formatted_options.append(
                {
                    "label": option["label"],
                    "text": option["text"],
                    "value": option["label"],  # Client can submit this value
                }
            )

        # Calculate progress
        progress_percentage = round(((question_index + 1) / total_questions) * 100, 1)

        return {
            "type": "quiz_question",
            "question_data": {
                "question_id": question["id"],
                "question_index": question_index,
                "question_number": question_index + 1,
                "total_questions": total_questions,
                "question_text": question["question"],
                "options": formatted_options,
                "progress": {
                    "current": question_index + 1,
                    "total": total_questions,
                    "percentage": progress_percentage,
                },
            },
            "metadata": {
                "expected_response_type": "single_choice",
                "timeout_seconds": 60,
                "can_skip": False,
                "timestamp": asyncio.get_event_loop().time(),
            },
        }

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

    def _create_results_server_message(
        self, flow_manager: FlowManager
    ) -> Dict[str, Any]:
        """
        Create the server message data for quiz results.

        Args:
            flow_manager: The FlowManager instance containing the quiz state

        Returns:
            Dictionary containing the results data for RTVI server message
        """
        total_score = flow_manager.state.get("total_score", 0)
        answers = flow_manager.state.get("answers", [])
        max_score = flow_manager.state.get("max_possible_score", 0)
        quiz_config = flow_manager.state.get("quiz_config", {})

        percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

        # Determine result category based on scoring ranges
        result_message = quiz_config.get("completion_message", self._completion_message)
        result_category = "Custom"
        scoring_ranges = quiz_config.get("scoring_ranges", self.get_scoring_ranges())

        for score_range in scoring_ranges:
            if score_range["min"] <= total_score <= score_range["max"]:
                result_message = score_range["result"]
                # Extract category name from result message (first part before colon)
                if ":" in result_message:
                    result_category = result_message.split(":")[0].strip()
                break

        # Format detailed answers for client
        formatted_answers = []
        for answer in answers:
            formatted_answers.append(
                {
                    "question_number": answer["question_index"] + 1,
                    "question_id": answer["question_id"],
                    "question_text": answer["question_text"],
                    "selected_option": {
                        "label": answer["selected_option"],
                        "text": answer["option_text"],
                        "points_earned": answer["points_earned"],
                    },
                }
            )

        # Create performance breakdown
        performance_metrics = {
            "score_distribution": {
                "earned_points": total_score,
                "possible_points": max_score,
                "percentage": percentage,
            },
            "category_placement": {
                "category": result_category,
                "description": result_message,
            },
            "question_performance": {
                "total_questions": len(self._questions),
                "questions_answered": len(answers),
                "average_score_per_question": round(total_score / len(answers), 1)
                if answers
                else 0,
            },
        }

        # Determine performance level for client UI
        performance_level = "developing"
        if percentage >= 85:
            performance_level = "excellent"
        elif percentage >= 70:
            performance_level = "strong"
        elif percentage >= 50:
            performance_level = "competent"

        return {
            "type": "quiz_results",
            "results_data": {
                "summary": {
                    "total_score": total_score,
                    "max_possible_score": max_score,
                    "percentage": percentage,
                    "performance_level": performance_level,
                    "category": result_category,
                    "completion_timestamp": asyncio.get_event_loop().time(),
                },
                "interpretation": {
                    "result_message": result_message,
                    "recommendations": self._generate_recommendations(
                        percentage, result_category
                    ),
                    "strengths": self._identify_strengths(answers),
                    "development_areas": self._identify_development_areas(answers),
                },
                "detailed_breakdown": {
                    "performance_metrics": performance_metrics,
                    "question_by_question": formatted_answers,
                    "score_analysis": {
                        "highest_scoring_areas": self._get_highest_scoring_areas(
                            answers
                        ),
                        "lowest_scoring_areas": self._get_lowest_scoring_areas(answers),
                    },
                },
            },
            "metadata": {
                "quiz_completed": True,
                "results_generated": True,
                "export_available": True,
                "timestamp": asyncio.get_event_loop().time(),
            },
        }

    def _generate_recommendations(self, percentage: float, category: str) -> List[str]:
        """Generate personalized recommendations based on performance."""
        recommendations = []

        if percentage < 50:
            recommendations.extend(
                [
                    "Focus on building foundational skills through structured learning",
                    "Seek mentorship opportunities to accelerate development",
                    "Consider formal training programs in your areas of interest",
                ]
            )
        elif percentage < 70:
            recommendations.extend(
                [
                    "Build on your solid foundation by taking on stretch assignments",
                    "Develop leadership skills through team collaboration",
                    "Expand your influence by sharing knowledge with others",
                ]
            )
        elif percentage < 85:
            recommendations.extend(
                [
                    "Consider taking on leadership roles in complex projects",
                    "Mentor junior team members to develop your coaching skills",
                    "Look for strategic initiatives that leverage your strengths",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Pursue executive development opportunities",
                    "Shape organizational strategy and culture",
                    "Drive innovation and transformation initiatives",
                ]
            )

        return recommendations

    def _identify_strengths(self, answers: List[Dict]) -> List[str]:
        """Identify strength areas based on high-scoring responses."""
        strengths = []
        high_score_threshold = 12  # Adjust based on your scoring system

        strength_areas = {
            "problem_solving": "Analytical Problem Solving",
            "time_management": "Time Management & Prioritization",
            "leadership_style": "Leadership & Team Collaboration",
            "learning_approach": "Learning & Skill Development",
            "communication_preference": "Communication & Presentation",
            "innovation_mindset": "Innovation & Change Management",
        }

        for answer in answers:
            if answer["points_earned"] >= high_score_threshold:
                area = strength_areas.get(answer["question_id"], "Professional Skills")
                if area not in strengths:
                    strengths.append(area)

        return strengths

    def _identify_development_areas(self, answers: List[Dict]) -> List[str]:
        """Identify development areas based on lower-scoring responses."""
        development_areas = []
        low_score_threshold = 8  # Adjust based on your scoring system

        area_mapping = {
            "problem_solving": "Strategic Problem Solving",
            "time_management": "Advanced Planning & Prioritization",
            "leadership_style": "Leadership Presence & Influence",
            "learning_approach": "Self-Directed Learning",
            "communication_preference": "Executive Communication",
            "innovation_mindset": "Change Leadership & Innovation",
        }

        for answer in answers:
            if answer["points_earned"] <= low_score_threshold:
                area = area_mapping.get(
                    answer["question_id"], "Professional Development"
                )
                if area not in development_areas:
                    development_areas.append(area)

        return development_areas

    def _get_highest_scoring_areas(self, answers: List[Dict]) -> List[Dict]:
        """Get the highest scoring question areas."""
        sorted_answers = sorted(answers, key=lambda x: x["points_earned"], reverse=True)
        return sorted_answers[:3]  # Top 3 performing areas

    def _get_lowest_scoring_areas(self, answers: List[Dict]) -> List[Dict]:
        """Get the lowest scoring question areas."""
        sorted_answers = sorted(answers, key=lambda x: x["points_earned"])
        return sorted_answers[:2]  # Bottom 2 performing areas

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

            # Send first question to client via RTVI server message
            if self._questions:
                question_data = self._create_question_server_message(0)
                if question_data:
                    logger.info("Sending first question data to client via RTVI")
                    await flow_manager.llm.push_frame(
                        RTVIServerMessageFrame(data=question_data)
                    )

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

                    # Send next question to client via RTVI server message (if not the last question)
                    next_question_index = q_index + 1
                    if next_question_index < len(self._questions):
                        question_data = self._create_question_server_message(
                            next_question_index
                        )
                        if question_data:
                            logger.info(
                                f"Sending question {next_question_index + 1} data to client via RTVI"
                            )
                            await flow_manager.llm.push_frame(
                                RTVIServerMessageFrame(data=question_data)
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

            # Send results to client via RTVI server message
            results_data = self._create_results_server_message(flow_manager)

            logger.info("Sending comprehensive quiz results to client via RTVI")
            await flow_manager.llm.push_frame(RTVIServerMessageFrame(data=results_data))

            #
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
