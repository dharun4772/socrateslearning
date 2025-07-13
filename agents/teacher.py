# agents/teacher.py

from config.llm_config import chat_with_llm, LLMProvider
from typing import List, Dict, Optional

def teacher_agent(
    question: str, 
    student_reply: str, 
    provider: LLMProvider = "gemini", 
    model: str = "gemini-2.5-flash",
    conversation_history: Optional[List[Dict]] = None,
    iteration: int = 1
) -> str:
    """
    Socratic teacher agent that guides students through questioning.
    
    Args:
        question: Original interview question
        student_reply: Student's current response
        provider: LLM provider ("gemini" or "ollama") 
        model: Model name
        conversation_history: Full conversation context
        iteration: Current iteration number
    """
    
    # Build conversation context
    context = ""
    if conversation_history and len(conversation_history) > 2:
        context = "\n\nConversation so far:\n"
        for entry in conversation_history:
            role = entry["role"].capitalize()
            content = entry["content"][:150] + "..." if len(entry["content"]) > 150 else entry["content"]
            context += f"{role} (Round {entry['iteration']}): {content}\n"
    
    # Adjust teaching strategy based on iteration
    if iteration == 1:
        teaching_focus = "Start by identifying what the student knows and guide them toward key concepts they're missing."
    elif iteration <= 3:
        teaching_focus = "Build on their partial understanding. Ask questions that help them connect concepts or see gaps in their reasoning."
    else:
        teaching_focus = "Help them synthesize their learning. Guide them toward a complete understanding or help them recognize what they still need to learn."
    
    prompt = f"""
You are a Socratic teacher helping a data science student learn through guided questioning and reflection.

Your teaching philosophy:
- NEVER give direct answers or solutions
- Ask probing questions that lead students to insights
- Point out inconsistencies in their thinking without correcting them directly
- Encourage deeper exploration of concepts
- Help students connect ideas and see patterns
- Guide them to discover their own misconceptions

Original interview question: "{question}"

Student's latest response: "{student_reply}"

{context}

Current teaching focus (Iteration {iteration}): {teaching_focus}

Your task: Write a Socratic response that guides the student toward deeper understanding. Your response should:

1. Acknowledge what the student got right (if anything)
2. Ask 1-2 thoughtful questions that probe their understanding
3. Maybe hint at a direction to explore without giving the answer
4. Stay encouraging but intellectually challenging

Keep your response to 2-3 sentences. Remember: Questions, not answers!
"""
    
    return chat_with_llm(prompt, provider=provider, model=model)