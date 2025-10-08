# EVOLVE-BLOCK-START
"""
Multi-Agent System for Mathematical Problem Solving

This system uses up to 4 agents with configurable roles, prompts, and communication protocols
to collaboratively solve mathematical problems from the Math500 dataset.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Agent System Prompts - These will be evolved by OpenEvolve
SOLVER_PROMPT = """You are a mathematical problem solver. Solve the problem step-by-step with clear reasoning.
Think carefully about the mathematical concepts involved. Show all intermediate steps.
Provide your final answer in a boxed format: \\boxed{answer}"""

VERIFIER_PROMPT = """You are a solution verifier. Analyze the given solution critically:
1. Check if each step follows logically from the previous
2. Verify mathematical operations are correct
3. Ensure the final answer addresses the question

Respond with:
VERIFICATION: [your detailed analysis]
DECISION: [APPROVED if correct, NEEDS_REVISION if issues found]
ISSUES: [list specific problems if NEEDS_REVISION]"""

REVISER_PROMPT = """You are a solution reviser. Given a problem, an attempted solution, and verifier feedback:
1. Carefully address each issue raised by the verifier
2. Fix mathematical errors
3. Clarify unclear reasoning steps
4. Provide a corrected solution with final answer in \\boxed{answer} format"""

REFINER_PROMPT = """You are a mathematical refiner. Review the solution and:
1. Simplify expressions where possible
2. Check for computational errors
3. Ensure the answer is in simplest form
Provide the refined final answer in \\boxed{answer} format"""


# Communication Protocol Configuration
MAX_REVISION_ROUNDS = 2
USE_REFINER = True
REFINER_THRESHOLD = 0.8  # Only use refiner if verifier confidence is high

# Global counters and logging
_llm_call_count = 0
_conversation_log = []  # Log of all agent interactions

# Logging configuration
_log_dir = None
_run_id = None
_iteration_counter = 0


def call_llm(system_prompt: str, user_message: str, model: Optional[str] = None, agent_role: str = "unknown") -> str:
    """
    Call LLM with given prompts. Uses environment variable for API configuration.

    Args:
        system_prompt: System message defining agent role
        user_message: User query/task
        model: Optional model override
        agent_role: Name of the agent (for logging)

    Returns:
        LLM response string
    """
    global _llm_call_count, _conversation_log

    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage

        # Get model from environment or use default
        if model is None:
            model = os.environ.get("OPENEVOLVE_MODEL", "gpt-5-mini")

        llm = ChatOpenAI(
            model_name=model,
            temperature=0.0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = llm.invoke(messages)

        # Increment call counter
        _llm_call_count += 1

        # Log the conversation
        _conversation_log.append({
            "agent": agent_role,
            "call_number": _llm_call_count,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "response": response.content,
            "model": model
        })

        return response.content

    except Exception as e:
        # Fallback for when LangChain isn't available or API fails
        error_msg = f"Error calling LLM: {str(e)}"
        _conversation_log.append({
            "agent": agent_role,
            "call_number": _llm_call_count,
            "error": str(e)
        })
        return error_msg


def extract_decision(verification_response: str) -> tuple[str, str]:
    """Extract decision and issues from verifier response"""
    decision = "NEEDS_REVISION"  # Default to revision
    issues = ""

    lines = verification_response.split('\n')
    for line in lines:
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip()
        elif line.startswith("ISSUES:"):
            issues = line.replace("ISSUES:", "").strip()

    # Check for approval in various formats
    if "APPROVED" in verification_response.upper():
        decision = "APPROVED"

    return decision, issues


def solver_agent(problem: str) -> str:
    """First agent: Solves the mathematical problem"""
    return call_llm(SOLVER_PROMPT, problem, agent_role="solver")


def verifier_agent(problem: str, solution: str) -> str:
    """Second agent: Verifies the solution"""
    message = f"Problem:\n{problem}\n\nProposed Solution:\n{solution}"
    return call_llm(VERIFIER_PROMPT, message, agent_role="verifier")


def reviser_agent(problem: str, solution: str, feedback: str) -> str:
    """Third agent: Revises solution based on feedback"""
    message = f"Problem:\n{problem}\n\nPrevious Solution:\n{solution}\n\nVerifier Feedback:\n{feedback}"
    return call_llm(REVISER_PROMPT, message, agent_role="reviser")


def refiner_agent(problem: str, solution: str) -> str:
    """Fourth agent: Refines and simplifies the final answer"""
    message = f"Problem:\n{problem}\n\nSolution to Refine:\n{solution}"
    return call_llm(REFINER_PROMPT, message, agent_role="refiner")


def multi_agent_solve(problem: str) -> tuple[str, int]:
    """
    Main multi-agent pipeline for solving math problems.

    Protocol:
    1. Solver attempts the problem
    2. Verifier checks the solution
    3. If issues found, Reviser fixes them (up to MAX_REVISION_ROUNDS)
    4. If USE_REFINER enabled, Refiner polishes the final answer

    Args:
        problem: Mathematical problem statement

    Returns:
        Tuple of (final solution string, number of LLM calls made)
    """
    global _llm_call_count, _conversation_log
    _llm_call_count = 0  # Reset counter for this problem
    _conversation_log = []  # Reset conversation log for this problem

    # Stage 1: Initial solve
    current_solution = solver_agent(problem)

    # Stage 2: Verification and revision loop
    for _ in range(MAX_REVISION_ROUNDS):
        verification = verifier_agent(problem, current_solution)
        decision, issues = extract_decision(verification)

        if decision == "APPROVED":
            break

        # Revise if issues found
        if issues or "NEEDS_REVISION" in decision:
            current_solution = reviser_agent(problem, current_solution, verification)

    # Stage 3: Optional refinement
    if USE_REFINER:
        # Check if verification suggests high confidence
        final_verification = verifier_agent(problem, current_solution)
        if "APPROVED" in final_verification.upper():
            current_solution = refiner_agent(problem, current_solution)

    return current_solution, _llm_call_count


def extract_boxed_answer(solution: str) -> str:
    """
    Extract the final answer from solution text.
    Looks for \\boxed{...} format or falls back to last line.
    """
    import re

    # Try to find boxed answer
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, solution)
    if matches:
        return matches[-1]  # Return last boxed answer

    # Fallback: look for "answer is" pattern
    if "answer is" in solution.lower():
        parts = solution.lower().split("answer is")
        if len(parts) > 1:
            answer_part = parts[-1].strip()
            # Extract first reasonable chunk
            answer_part = answer_part.split('\n')[0].split('.')[0]
            return answer_part.strip()

    # Last resort: return solution as-is
    return solution


# EVOLVE-BLOCK-END


def solve_problem(problem: str) -> Dict[str, str]:
    """
    External interface for solving a single problem.

    Args:
        problem: Mathematical problem text

    Returns:
        Dictionary with 'solution', 'answer', 'llm_calls', and 'conversation' keys
    """
    global _conversation_log

    solution, llm_calls = multi_agent_solve(problem)
    answer = extract_boxed_answer(solution)

    return {
        'solution': solution,
        'answer': answer,
        'pred': answer,  # For evaluator compatibility
        'llm_calls': llm_calls,  # Track efficiency
        'conversation': _conversation_log.copy()  # Include full conversation log
    }


def initialize_logging(base_dir: str = "logs"):
    """Initialize logging directory structure."""
    global _log_dir, _run_id

    # Create base logs directory
    logs_path = Path(base_dir)
    logs_path.mkdir(exist_ok=True)

    # Create run-specific directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _run_id = f"run_{timestamp}"
    _log_dir = logs_path / _run_id
    _log_dir.mkdir(exist_ok=True)

    return _log_dir


def save_iteration_log(results: List[Dict], iteration: int):
    """Save conversation logs for an iteration."""
    global _log_dir, _iteration_counter

    if _log_dir is None:
        _log_dir = initialize_logging()

    # Use provided iteration or increment counter
    if iteration is None:
        _iteration_counter += 1
        iteration = _iteration_counter

    # Create iteration log
    iteration_data = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "num_problems": len(results),
        "problems": []
    }

    for idx, result in enumerate(results):
        problem_data = {
            "problem_index": idx,
            "problem": result.get('problem', ''),
            "gold_answer": result.get('gold_answer', ''),
            "predicted_answer": result.get('pred', ''),
            "llm_calls": result.get('llm_calls', 0),
            "conversation": result.get('conversation', []),
            "error": result.get('error')
        }
        iteration_data["problems"].append(problem_data)

    # Save to file
    filename = _log_dir / f"iteration_{iteration:04d}.json"
    with open(filename, 'w') as f:
        json.dump(iteration_data, f, indent=2)

    return filename


def run_evaluation_sample(problems: List[Dict[str, str]], sample_size: int = 5) -> List[Dict]:
    """
    Run multi-agent system on a sample of problems.
    Used by evaluator to test system performance.

    Args:
        problems: List of problem dictionaries with 'question' and 'final_answer'
        sample_size: Number of problems to evaluate

    Returns:
        List of result dictionaries
    """
    results = []

    for problem_data in problems[:sample_size]:
        try:
            problem_text = problem_data.get('question', '')
            gold_answer = problem_data.get('final_answer', '')

            # Solve the problem
            result = solve_problem(problem_text)

            # Add ground truth for evaluation
            result['gold_answer'] = gold_answer
            result['problem'] = problem_text

            results.append(result)

        except Exception as e:
            # Log error but continue
            results.append({
                'problem': problem_data.get('question', ''),
                'gold_answer': problem_data.get('final_answer', ''),
                'solution': f"Error: {str(e)}",
                'answer': "",
                'pred': "",
                'llm_calls': 0,
                'conversation': [],
                'error': str(e)
            })

    # Save logs for this evaluation (get iteration from environment or use None)
    iteration = int(os.environ.get('OPENEVOLVE_ITERATION', '0'))
    save_iteration_log(results, iteration)

    return results


if __name__ == "__main__":
    # Test the system with a sample problem
    test_problem = "What is the value of $3! + 4! + 5!$?"

    print("Testing Multi-Agent Math Solver")
    print("=" * 50)
    print(f"Problem: {test_problem}\n")

    result = solve_problem(test_problem)

    print("Solution:")
    print(result['solution'])
    print("\n" + "=" * 50)
    print(f"Extracted Answer: {result['answer']}")
