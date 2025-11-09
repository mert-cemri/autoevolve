"""
Multi-Agent System for Mathematical Problem Solving

This version allows OpenEvolve to evolve the entire multi-agent architecture:
- Agent roles and prompts
- Communication protocols
- Execution order
- Decision logic

STRUCTURE:
- FIXED: Minimal framework (call_llm, evaluation interface)
- EVOLVABLE: Everything else (agents, protocols, orchestration logic)
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ===================================================================================
# FIXED FRAMEWORK: Do not evolve this section
# Provides basic LLM calling and evaluation interface
# ===================================================================================

# Global counters for tracking
_llm_call_count = 0
_conversation_log = []
_log_dir = None
_run_id = None
_iteration_counter = 0


def call_llm(system_prompt: str, user_message: str, model: Optional[str] = None, agent_role: str = "unknown") -> str:
    """
    Basic LLM calling function. Do not modify this.

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

        _llm_call_count += 1
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
        error_msg = f"Error calling LLM: {str(e)}"
        _conversation_log.append({
            "agent": agent_role,
            "call_number": _llm_call_count,
            "error": str(e)
        })
        return error_msg


def initialize_logging(base_dir: str = "logs"):
    """Initialize logging directory. Do not modify."""
    global _log_dir, _run_id

    logs_path = Path(base_dir)
    logs_path.mkdir(exist_ok=True)

    approach = os.environ.get("SEARCH_STRATEGY", "openevolve")
    approach_path = logs_path / approach
    approach_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _run_id = f"run_{timestamp}"
    _log_dir = approach_path / _run_id
    _log_dir.mkdir(exist_ok=True)

    return _log_dir


def save_iteration_log(results: List[Dict], iteration: Optional[int] = None):
    """Save conversation logs. Do not modify."""
    global _log_dir, _iteration_counter

    if _log_dir is None:
        _log_dir = initialize_logging()

    if iteration is None:
        _iteration_counter += 1
        iteration = _iteration_counter

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

    filename = _log_dir / f"iteration_{iteration:04d}.json"
    with open(filename, 'w') as f:
        json.dump(iteration_data, f, indent=2)

    return filename


def run_evaluation_sample(problems: List[Dict[str, str]], sample_size: int = 5) -> List[Dict]:
    """
    Evaluation interface. Do not modify.
    Calls solve_problem() for each problem (which IS evolvable).
    """
    results = []

    for problem_data in problems[:sample_size]:
        try:
            problem_text = problem_data.get('question', '')
            gold_answer = problem_data.get('final_answer', '')

            # Call the evolvable solve_problem function
            result = solve_problem(problem_text)

            result['gold_answer'] = gold_answer
            result['problem'] = problem_text
            results.append(result)

        except Exception as e:
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

    # Save logs
    try:
        iteration = int(os.environ.get('OPENEVOLVE_ITERATION', '-1'))
        if iteration < 0:
            iteration = None
    except (ValueError, TypeError):
        iteration = None

    save_iteration_log(results, iteration)

    return results


# ===================================================================================
# EVOLVE-BLOCK-START
# ===================================================================================
# EVOLVABLE ZONE: Evolve the entire multi-agent architecture below
# You can modify:
#   - Agent definitions (roles, prompts)
#   - Communication protocols (order, decision logic)
#   - Number of agents
#   - Orchestration logic
# ===================================================================================


# Agent Prompts (evolvable)
SOLVER_PROMPT = """You are a mathematical problem solver. Solve the problem step-by-step with clear reasoning.
Think carefully about the mathematical concepts involved. Show all intermediate steps.
For complex problems, break them down into smaller sub-problems.
Provide your final answer in a boxed format: \\boxed{answer}"""

VERIFIER_PROMPT = """You are a solution verifier. Analyze the given solution critically:
1. Check if each step follows logically from the previous
2. Verify mathematical operations are correct
3. Ensure the final answer addresses the question
4. Check for computational errors

Respond with:
VERIFICATION: [your detailed analysis]
DECISION: [APPROVED if correct, NEEDS_REVISION if issues found]
ISSUES: [list specific problems if NEEDS_REVISION, or "None" if approved]"""

REVISER_PROMPT = """You are a solution reviser. Given a problem, an attempted solution, and verifier feedback:
1. Carefully address each issue raised by the verifier
2. Fix mathematical errors
3. Clarify unclear reasoning steps
4. Break down complex steps if needed
5. Provide a corrected solution with final answer in \\boxed{answer} format"""

REFINER_PROMPT = """You are a mathematical refiner. Review the solution and:
1. Simplify expressions where possible
2. Check for computational errors
3. Ensure the answer is in simplest form
4. Verify the boxed answer is correct
Provide the refined final answer in \\boxed{answer} format"""


# Multi-Agent System Configuration (evolvable)
class AgentConfig:
    """Configuration for multi-agent system architecture."""

    # Agent execution parameters
    MAX_REVISION_ROUNDS = 2  # Number of solver-verifier-reviser cycles (0-3)
    USE_REFINER = True       # Enable final refinement pass
    USE_VERIFIER = True      # Enable verification (can disable for speed)

    # Decision thresholds
    REFINER_THRESHOLD = 0.8  # Confidence threshold to use refiner
    EARLY_STOP_THRESHOLD = 0.95  # Stop early if confidence is very high

    # Agent communication order (evolvable)
    # Format: List of (agent_function, should_use_previous_output)
    AGENT_PIPELINE = [
        ("solver", False),      # Solver sees only the problem
        ("verifier", True),     # Verifier sees solver's output
        ("reviser", True),      # Reviser sees both (conditional)
        ("refiner", True),      # Refiner polishes final (conditional)
    ]


def extract_decision(verification_response: str) -> tuple[str, str]:
    """
    Extract decision and issues from verifier response.
    Evolvable: Can modify parsing logic.
    """
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


def extract_boxed_answer(solution: str) -> str:
    """
    Extract final answer from solution text.
    Evolvable: Can modify answer extraction logic.
    """
    import re

    # Try to find boxed answer
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, solution)
    if matches:
        return matches[-1]

    # Fallback: look for "answer is" pattern
    if "answer is" in solution.lower():
        parts = solution.lower().split("answer is")
        if len(parts) > 1:
            answer_part = parts[-1].strip()
            answer_part = answer_part.split('\n')[0].split('.')[0]
            return answer_part.strip()

    # Last resort: return solution as-is
    return solution


def solver_agent(problem: str) -> str:
    """Solver agent: Initial problem solving."""
    return call_llm(SOLVER_PROMPT, problem, agent_role="solver")


def verifier_agent(problem: str, solution: str) -> str:
    """Verifier agent: Solution verification."""
    message = f"Problem:\n{problem}\n\nProposed Solution:\n{solution}"
    return call_llm(VERIFIER_PROMPT, message, agent_role="verifier")


def reviser_agent(problem: str, solution: str, feedback: str) -> str:
    """Reviser agent: Solution revision based on feedback."""
    message = f"Problem:\n{problem}\n\nPrevious Solution:\n{solution}\n\nVerifier Feedback:\n{feedback}"
    return call_llm(REVISER_PROMPT, message, agent_role="reviser")


def refiner_agent(problem: str, solution: str) -> str:
    """Refiner agent: Final answer polishing."""
    message = f"Problem:\n{problem}\n\nSolution to Refine:\n{solution}"
    return call_llm(REFINER_PROMPT, message, agent_role="refiner")


def solve_problem(problem: str) -> Dict[str, str]:
    """
    Main multi-agent orchestration logic.
    FULLY EVOLVABLE: Modify the agent communication protocol here.

    Current architecture:
    1. Solver attempts problem
    2. Verifier checks solution (loop with reviser if needed)
    3. Refiner polishes final answer (optional)

    Args:
        problem: Mathematical problem statement

    Returns:
        Dictionary with solution, answer, llm_calls, conversation
    """
    global _llm_call_count, _conversation_log
    _llm_call_count = 0
    _conversation_log = []

    # Stage 1: Initial solve
    current_solution = solver_agent(problem)

    # Stage 2: Verification and revision loop
    if AgentConfig.USE_VERIFIER:
        for revision_round in range(AgentConfig.MAX_REVISION_ROUNDS):
            verification = verifier_agent(problem, current_solution)
            decision, issues = extract_decision(verification)

            # Check if approved
            if decision == "APPROVED":
                break

            # Check for early stopping (high confidence)
            if revision_round > 0 and "high confidence" in verification.lower():
                break

            # Revise if issues found
            if issues and issues.lower() != "none":
                current_solution = reviser_agent(problem, current_solution, verification)

    # Stage 3: Optional refinement
    if AgentConfig.USE_REFINER:
        # Only use refiner if verification passed
        if AgentConfig.USE_VERIFIER:
            final_verification = verifier_agent(problem, current_solution)
            if "APPROVED" in final_verification.upper():
                current_solution = refiner_agent(problem, current_solution)
        else:
            # No verifier, use refiner directly
            current_solution = refiner_agent(problem, current_solution)

    # Extract final answer
    answer = extract_boxed_answer(current_solution)

    return {
        'solution': current_solution,
        'answer': answer,
        'pred': answer,
        'llm_calls': _llm_call_count,
        'conversation': _conversation_log.copy()
    }


# ===================================================================================
# EVOLVE-BLOCK-END
# ===================================================================================


if __name__ == "__main__":
    # Test the system
    test_problem = "What is the value of $3! + 4! + 5!$?"

    print("Testing Multi-Agent Math Solver")
    print("=" * 50)
    print(f"Problem: {test_problem}\n")

    result = solve_problem(test_problem)

    print("Solution:")
    print(result['solution'])
    print("\n" + "=" * 50)
    print(f"Extracted Answer: {result['answer']}")
    print(f"LLM Calls: {result['llm_calls']}")
