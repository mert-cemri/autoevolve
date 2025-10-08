"""
Base class for custom search strategies
"""

import importlib.util
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


class Program:
    """Represents a program candidate"""

    def __init__(self, code: str, parent_id: Optional[str] = None, generation: int = 0):
        self.id = f"{generation}_{int(time.time() * 1000000) % 1000000}"
        self.code = code
        self.parent_id = parent_id
        self.generation = generation
        self.metrics: Optional[Dict[str, float]] = None
        self.score: float = -float('inf')
        self.evaluated = False

    def __repr__(self):
        return f"Program(id={self.id}, gen={self.generation}, score={self.score:.4f})"


class BaseSearch(ABC):
    """Base class for all search strategies"""

    def __init__(
        self,
        initial_program_path: str,
        evaluator_path: str,
        output_dir: str,
        strategy_name: str = "unknown",
        num_eval_problems: int = 10,
        model: str = "gpt-5",
        agent_model: str = "gpt-5-mini",
        temperature: float = 0.8,
        max_tokens: int = 16000,
    ):
        self.initial_program_path = initial_program_path
        self.evaluator_path = evaluator_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategy_name = strategy_name
        self.num_eval_problems = num_eval_problems

        # Set environment variables
        os.environ["SEARCH_STRATEGY"] = strategy_name
        os.environ["MATH_EVAL_PROBLEMS"] = str(num_eval_problems)
        os.environ["OPENEVOLVE_MODEL"] = agent_model  # Model for multi-agent system

        # LLM configuration
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Load initial program
        self.initial_program = self._load_program(initial_program_path)

        # Load evaluator
        self.evaluator = self._load_evaluator(evaluator_path)

        # Logging
        self.setup_logging()
        self.history: List[Dict] = []

    def setup_logging(self):
        """Setup logging for the search"""
        log_file = self.output_dir / "search.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _load_program(self, path: str) -> str:
        """Load program code from file"""
        with open(path, 'r') as f:
            return f.read()

    def _load_evaluator(self, path: str):
        """Load evaluator module"""
        spec = importlib.util.spec_from_file_location("evaluator", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load evaluator from {path}")

        evaluator = importlib.util.module_from_spec(spec)
        sys.modules["evaluator"] = evaluator
        spec.loader.exec_module(evaluator)

        return evaluator

    def mutate_program(self, program: Program, prompt_context: str = "") -> str:
        """
        Use LLM to generate an improved version of the program

        Args:
            program: Current program to improve
            prompt_context: Additional context for the mutation

        Returns:
            New program code
        """
        logger.info(f"  Calling LLM ({self.model}) to generate mutation...")

        system_prompt = """You are an expert at evolving multi-agent systems for mathematical problem solving.

Your goal is to improve a multi-agent math solver by modifying the code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END markers.

Key areas to evolve:
1. **Agent Prompts** (SOLVER_PROMPT, VERIFIER_PROMPT, REVISER_PROMPT, REFINER_PROMPT):
   - Make prompts more specific for mathematical reasoning
   - Add techniques like chain-of-thought, self-consistency, step verification
   - Improve error detection and correction strategies

2. **Communication Protocol** (MAX_REVISION_ROUNDS, USE_REFINER, REFINER_THRESHOLD):
   - Balance accuracy vs efficiency (fewer LLM calls = better efficiency score)
   - Optimize when to use revision vs refinement

3. **Decision Logic**:
   - Improve the flow of information between agents
   - Enhance error detection and recovery

EVALUATION METRICS (your changes will be judged on):
- Accuracy: % of problems solved correctly (70% of score)
- Efficiency: Fewer LLM calls per problem (25% of score, target: 3 calls, penalty at 10+)
- Speed: Time per problem (5% of score)

CRITICAL RULES:
1. Keep ALL code outside EVOLVE blocks EXACTLY the same
2. Maintain the same function signatures (run_evaluation_sample, call_llm, etc.)
3. Ensure the code is syntactically correct Python
4. Return ONLY the complete program code, no explanations"""

        user_prompt = f"""Current program performance:
- Score: {program.score:.4f}
- Metrics: {program.metrics}

{prompt_context}

Current program:
```python
{program.code}
```

Generate an IMPROVED version of this program that will achieve better accuracy and efficiency.
Focus your changes on the code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END markers.
Return the complete improved Python code."""

        # Build API parameters
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_completion_tokens": self.max_tokens
        }

        # GPT-5 only supports temperature=1 (default), so don't pass it
        if not self.model.startswith("gpt-5"):
            api_params["temperature"] = self.temperature

        response = self.client.chat.completions.create(**api_params)

        new_code = response.choices[0].message.content
        logger.info(f"  LLM response received ({len(new_code)} chars)")

        # Extract code from markdown if present
        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0].strip()
        elif "```" in new_code:
            new_code = new_code.split("```")[1].split("```")[0].strip()

        # Validate the generated code
        if not self._validate_program(new_code, program.code):
            logger.warning("  Generated code failed validation, using parent code")
            return program.code

        logger.info("  ✓ Generated code validated successfully")
        return new_code

    def _validate_program(self, new_code: str, fallback_code: str) -> bool:
        """
        Validate that generated code is syntactically correct and has required functions

        Args:
            new_code: Generated program code to validate
            fallback_code: Original code to fall back to if validation fails

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check syntax by parsing
            import ast
            ast.parse(new_code)

            # Check for required function
            if "def run_evaluation_sample" not in new_code:
                logger.warning("  Validation failed: missing run_evaluation_sample function")
                return False

            # Check for EVOLVE blocks
            if "# EVOLVE-BLOCK-START" not in new_code or "# EVOLVE-BLOCK-END" not in new_code:
                logger.warning("  Validation failed: missing EVOLVE-BLOCK markers")
                return False

            return True

        except SyntaxError as e:
            logger.warning(f"  Validation failed: syntax error - {e}")
            return False
        except Exception as e:
            logger.warning(f"  Validation failed: {e}")
            return False

    def evaluate_program(self, program: Program) -> Dict[str, float]:
        """
        Evaluate a program using the evaluator

        Args:
            program: Program to evaluate

        Returns:
            Dictionary of metrics
        """
        logger.info(f"  Evaluating program {program.id}...")

        # Write program to temporary file
        temp_file = self.output_dir / f"temp_{program.id}.py"
        with open(temp_file, 'w') as f:
            f.write(program.code)

        try:
            # Run full evaluation (stage 3 only - no cascade)
            metrics = self.evaluator.evaluate_stage3(str(temp_file))

            # Calculate combined score
            accuracy = metrics.get('accuracy', 0.0)
            avg_calls = metrics.get('avg_llm_calls', 10.0)
            efficiency_score = metrics.get('efficiency_score', 0.0)

            # Combined score: 70% accuracy, 25% efficiency, 5% time
            score = (
                0.70 * accuracy +
                0.25 * efficiency_score +
                0.05 * metrics.get('time_score', 0.0)
            )

            program.metrics = metrics
            program.score = score
            program.evaluated = True

            logger.info(
                f"  ✓ Evaluation complete: accuracy={accuracy:.2%}, "
                f"avg_calls={avg_calls:.1f}, combined_score={score:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed for {program.id}: {e}")
            program.metrics = {"error": str(e)}
            program.score = 0.0
            program.evaluated = True
            return program.metrics

        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()

    def save_program(self, program: Program, filename: str):
        """Save program code to file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(program.code)

        # Save metadata
        metadata = {
            "id": program.id,
            "generation": program.generation,
            "parent_id": program.parent_id,
            "score": program.score,
            "metrics": program.metrics
        }
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_history(self):
        """Save search history"""
        history_file = self.output_dir / "history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    @abstractmethod
    def search(self, **kwargs) -> Program:
        """
        Run the search algorithm

        Returns:
            Best program found
        """
        pass
