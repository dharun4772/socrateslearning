# agents/teacher.py

from config.llm_config import chat_with_llm, LLMProvider
from typing import List, Dict, Optional

def teacher_agent(
    question: str, 
    student_reply: str, 
    provider: LLMProvider = "gemini", 
    model: str = "gemini-2.0-flash",
    conversation_history: Optional[List[Dict]] = None,
    iteration: int = 1
) -> str:
    """
    Socratic teacher agent that guides students through questioning.
    Recognizes when students have answered correctly and provides appropriate closure.
    """
    
    # Build conversation context
    context = ""
    if conversation_history and len(conversation_history) > 2:
        context = "\n\nConversation so far:\n"
        for entry in conversation_history:
            role = entry["role"].capitalize()
            content = entry["content"][:150] + "..." if len(entry["content"]) > 150 else entry["content"]
            context += f"{role} (Round {entry['iteration']}): {content}\n"
    
    prompt = f"""
You are a Socratic teacher helping a data science student learn through guided questioning.

CRITICAL INSTRUCTION: First, evaluate if the student has already provided a CORRECT and COMPLETE answer to the original question. If they have, acknowledge their success and provide closure rather than asking more questions.

Original interview question: "{question}"
Student's latest response: "{student_reply}"
{context}

Your evaluation process:
1. FIRST: Has the student correctly answered the core question? 
   - Do they demonstrate understanding of key concepts?
   - Have they addressed all main components of the question?
   - Is their explanation accurate and reasonably complete?

2. IF YES (correct answer): 
   - Acknowledge their correct understanding
   - Briefly reinforce the key insight they demonstrated
   - Provide encouraging closure (e.g., "Excellent! You've grasped the core concept...")
   - DO NOT ask follow-up questions

3. IF NO (incomplete/incorrect):
   - Ask 1-2 Socratic questions to guide them toward the missing pieces
   - Focus on gaps in their understanding
   - Never give direct answers

Teaching guidelines when continuing dialogue:
- Point out inconsistencies without directly correcting
- Ask questions that lead to insights
- Help them connect concepts
- Stay encouraging but challenging

Current iteration: {iteration}

Response format:
- If answer is correct: 2-3 sentences of acknowledgment and closure
- If answer needs work: 2-3 sentences with Socratic questions

Remember: The goal is learning, not endless questioning. Recognize success when you see it!
"""
    
    return chat_with_llm(prompt, provider=provider, model=model)