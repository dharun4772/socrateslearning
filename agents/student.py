# agents/student.py

from config.llm_config import chat_with_llm, LLMProvider
from config.personas import persona_traits
from typing import List, Dict, Optional

def student_agent(
    question: str, 
    persona: str, 
    provider: LLMProvider = "gemini", 
    model: str = "gemini-2.5-flash",
    teacher_guidance: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Student agent that responds to questions based on their persona and learning state.
    
    Args:
        question: The original interview question
        persona: Student personality type
        provider: LLM provider ("gemini" or "ollama")
        model: Model name
        teacher_guidance: Latest teacher response for iterative learning
        conversation_history: Full conversation context
    """
    traits = persona_traits[persona]
    
    # Build context from conversation history
    context = ""
    if conversation_history:
        context = "\n\nPrevious conversation:\n"
        for entry in conversation_history[-4:]:  # Last 4 exchanges for context
            role = entry["role"].capitalize()
            content = entry["content"][:200] + "..." if len(entry["content"]) > 200 else entry["content"]
            context += f"{role}: {content}\n"
    
    # Different prompts for initial vs iterative responses
    if teacher_guidance and conversation_history:
        # Continuing the Socratic dialogue
        prompt = f"""
You are a data science student with the following traits:
- Persona: {persona}
- Problem Understanding: {traits['Problem Understanding']}
- Instruction Understanding: {traits['Instruction Understanding']}
- Calculation: {traits['Calculation']}
- Knowledge Mastery: {traits['Knowledge Mastery']}
- Thirst for Learning: {traits['Thirst for Learning']}

Original question: "{question}"

{context}

Your teacher just asked you: "{teacher_guidance}"

Respond as this student persona would, building on what you've learned so far. Show your thinking process and any new insights you've gained. Keep your response to 2-4 sentences.

Remember:
- Stay true to your persona's learning style and knowledge level
- Show progression in understanding from previous responses
- Ask follow-up questions if you're curious about something
- Acknowledge when something clicks or when you're still confused
"""
    else:
        # Initial response to the question
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

Write a response in 2–4 sentences, reflecting your level of knowledge and confidence based on your persona traits. Don't be afraid to show uncertainty or ask clarifying questions if that fits your persona.
"""
    
    return chat_with_llm(prompt, provider=provider, model=model)