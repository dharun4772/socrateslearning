# agents/student.py

from utils.ollama_client import ollama_chat
from config.personas import persona_traits

def student_agent(question: str, persona: str, model: str = "llama3") -> str:
    traits = persona_traits[persona]

    prompt = f"""
You are a data science student with the following traits:
- Persona: {persona}
- Problem Understanding: {traits['Problem Understanding']}
- Instruction Understanding: {traits['Instruction Understanding']}
- Calculation: {traits['Calculation']}
- Knowledge Mastery: {traits['Knowledge Mastery']}
- Thirst for Learning: {traits['Thirst for Learning']}

Your task is to respond to this data science interview question:

"{question}"

Write a response in 2–4 sentences, reflecting your level of knowledge and confidence.
"""
    return ollama_chat(prompt, model=model)
