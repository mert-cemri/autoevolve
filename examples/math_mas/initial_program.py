# Cell 1: Install dependencies (if not already installed)
!pip install langchain langchain-openai langgraph datasets
!pip install timeout-decorator
# Cell 2: Imports
import random
import re
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import os
os.environ["OPENAI_API_KEY"] = "API_KEY"

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# Cell 4: Load OlympiadBench dataset and sample 50 problems (seed=42)

dataset = load_dataset("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", split="train")

random.seed(42)
sampled = random.sample(list(dataset), 50)

print(f"Loaded {len(sampled)} problems for evaluation.")
# Cell 5: Define agents

def solver(problem: str) -> str:
    """Problem Solver Agent"""
    messages = [
        SystemMessage(content="You are a mathematical problem solver. "
                              "Solve the problem step by step and clearly state the final boxed answer."),
        HumanMessage(content=problem),
    ]
    return llm.invoke(messages).content

def verifier(problem: str, solution: str) -> str:
    """Verifier Agent - Always provides reasoning"""
    messages = [
        SystemMessage(content="You are a verifier. Analyze the solution and ALWAYS explain your reasoning.\n\n"
                              "Format your response as:\n\n"
                              "VERIFICATION ANALYSIS:\n"
                              "[Detailed explanation of your verification process, checking each step]\n\n"
                              "DECISION: [APPROVED or NEEDS_REVISION]\n\n"
                              "If APPROVED: Explain why the solution is correct and the answer is valid.\n"
                              "If NEEDS_REVISION: List specific errors and required changes."),
        HumanMessage(content=f"Problem:\n{problem}\n\nSolution Attempt:\n{solution}")
    ]
    return llm.invoke(messages).content

def reviser(problem: str, solution: str, feedback: str) -> str:
    """Reviser Agent"""
    messages = [
        SystemMessage(content="You are a reviser. Based on the detailed verifier feedback, "
                              "carefully fix each identified error. "
                              "Address each point mentioned in the feedback. "
                              "Provide a corrected solution with a final boxed answer."),
        HumanMessage(content=f"Problem:\n{problem}\n\n"
                            f"Previous Solution:\n{solution}\n\n"
                            f"Verifier Feedback:\n{feedback}")
    ]
    return llm.invoke(messages).content

def judge(problem: str, solution: str, alt_solution: str = None) -> str:
    """Final Judge Agent"""
    messages = [
        SystemMessage(content="You are the final judge. Compare the given solutions and provide the cleanest, most correct final boxed answer."),
        HumanMessage(content=f"Problem:\n{problem}\n\nSolution 1:\n{solution}\n\nSolution 2:\n{alt_solution}")
    ]
    return llm.invoke(messages).content
from timeout_decorator import timeout, TimeoutError
@timeout(60)  # 60-second timeout per problem
def multi_agent_pipeline(problem: str) -> str:
    attempt = solver(problem)
    for _ in range(2):
        feedback = verifier(problem, attempt)
        
        # Check if approved (works with either format)
        if "APPROVED" in feedback.upper():
            return attempt
        elif "NEEDS_REVISION" in feedback.upper() or "REVISE" in feedback.upper():
            attempt = reviser(problem, attempt, feedback)
        else:
            # Fallback: if unclear, treat as needs revision
            attempt = reviser(problem, attempt, feedback)
    
    return judge(problem, attempt)
    def extract_boxed_answer(output: str):
    """Extract predicted answer from agent output."""
    return output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()
# Cell 8: Run evaluation on the sampled 50 problems

correct = 0
results = []
counter = 1

for ex in sampled:
    print(counter)
    problem, gold = ex["question"], ex["final_answer"]
    system_answer = multi_agent_pipeline(problem)
    pred = extract_boxed_answer(system_answer)

    results.append((problem, gold, pred, system_answer))
    if pred in gold:   # handles multiple valid answers
        correct += 1
    counter = 1 + counter

print(f"Accuracy: {correct} / {len(sampled)} = {100*correct/len(sampled):.2f}%")
# Cell 9: Save execution traces for MAST
import json

execution_traces_test = []
counter1 = 10
counter2 = 20

for ex in sampled:
    print(counter2)
    problem, gold = ex["question"], ex["final_answer"]

    # store detailed trace for each problem
    trace = {"problem": problem, "gold": gold, "agents": []}

    # solver
    attempt = solver(problem)
    if hasattr(attempt, "content"):
        attempt = attempt.content
    trace["agents"].append({"role": "solver", "output": attempt})

    # verifier/reviser loop
    for _ in range(2):
        print(counter1)
        feedback = verifier(problem, attempt)
        if hasattr(feedback, "content"):
            feedback = feedback.content
        trace["agents"].append({"role": "verifier", "output": feedback})

        if feedback.strip().upper().startswith("REVISE"):
            attempt = reviser(problem, attempt, feedback)
            if hasattr(attempt, "content"):
                attempt = attempt.content
            trace["agents"].append({"role": "reviser", "output": attempt})
        else:
            break
        counter1 = counter1+1
    counter2 = counter2+1

    # judge
    final = judge(problem, attempt)
    if hasattr(final, "content"):
        final = final.content
    trace["agents"].append({"role": "judge", "output": final})

    # store final results
    execution_traces_test.append(trace)

# save for MAST
with open("execution_traces_test.json", "w") as f:
    json.dump(execution_traces_test, f, indent=2)

print("Saved execution traces to execution_traces_test.json")
