import numpy as np
from eval_utils import load_jsonl, get_success
from initial_solution import multi_agent_pipeline, extract_boxed_answer

dataset = load_dataset("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", split="train")

def extract_boxed_answer(output: str):
    """Extract predicted answer from agent output."""
    return output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()
  
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
