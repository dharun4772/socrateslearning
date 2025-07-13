# agents/dean.py

import json
from utils.ollama_client import ollama_chat

def dean_agent(question: str, student_reply: str, teacher_reply: str, model: str = "llama3") -> dict:
    prompt = f"""
You are the Dean evaluating whether a teacher's response is Socratic in nature.

Definitions:
- Socratic = asking probing questions to guide the student to insight.
- NOT Socratic = directly giving the answer or being too suggestive.

Evaluate this interaction:

Question: "{question}"
Student's Attempt: "{student_reply}"
Teacher's Response: "{teacher_reply}"

Return your decision as JSON in this format:
{{
  "verdict": "Acceptable" or "Too direct",
  "revision": "Revised response if it's too direct, otherwise repeat the original"
}}
"""
    response = ollama_chat(prompt, model=model)

    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Fallback if model gives invalid JSON
        return {
            "verdict": "Acceptable",
            "revision": teacher_reply
        }
