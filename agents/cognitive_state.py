# agents/cognitive_state.py

import random

dimensions = [
    "Problem Understanding",
    "Instruction Understanding",
    "Calculation",
    "Knowledge Mastery",
    "Thirst for Learning"
]

def generate_cognitive_state(persona: str) -> dict:
    """
    Randomized but persona-influenced state (for now, use randomized levels).
    Could later be based on persona traits or improved via tracking turn history.
    """
    levels = ["Low", "Medium", "High"]
    return {dim: random.choice(levels) for dim in dimensions}
