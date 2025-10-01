import numpy as np
from eval_utils import load_jsonl, get_success

dataset = load_dataset("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", split="train")
