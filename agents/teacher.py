# agents/teacher.py

from utils.ollama_client import ollama_chat

def teacher_agent(question: str, student_response: str, model: str = "llama3") -> str:
    prompt = f"""
You are a Socratic teacher helping a data science student learn by guiding them through questions and reflection.

Your job is to respond to the student's answer with a question or prompt that:
- Encourages deeper thinking
- Points out misconceptions without directly correcting them
- Does NOT give away the correct answer

Here is the interview question:
"{question}"

Student's response:
"{student_response}"

Now write your Socratic response.
"""
    return ollama_chat(prompt, model=model)
