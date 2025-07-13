# agents/dean.py

import json
from config.llm_config import chat_with_llm, LLMProvider
from typing import List, Dict

def dean_agent(
    question: str, 
    conversation_history: List[Dict],
    current_iteration: int,
    max_iterations: int,
    provider: LLMProvider = "gemini", 
    model: str = "gemini-2.5-flash"
) -> dict:
    """
    Dean agent that evaluates the overall learning progress and decides whether to continue.
    
    Args:
        question: Original interview question
        conversation_history: Full conversation between student and teacher
        current_iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        provider: LLM provider ("gemini" or "ollama")
        model: Model name
    """
    
    # Extract student responses to analyze learning progression
    student_responses = [entry for entry in conversation_history if entry["role"] == "student"]
    teacher_responses = [entry for entry in conversation_history if entry["role"] == "teacher"]
    
    # Build conversation summary
    conversation_summary = ""
    for i, (student, teacher) in enumerate(zip(student_responses, teacher_responses), 1):
        conversation_summary += f"Round {i}:\n"
        conversation_summary += f"Student: {student['content'][:200]}...\n" if len(student['content']) > 200 else f"Student: {student['content']}\n"
        conversation_summary += f"Teacher: {teacher['content'][:200]}...\n\n" if len(teacher['content']) > 200 else f"Teacher: {teacher['content']}\n\n"
    
    prompt = f"""
You are the Dean evaluating a Socratic dialogue session between a teacher and student.

Your job is to assess:
1. Student's learning progression and current understanding level
2. Whether the dialogue should continue or if learning objectives are met
3. Quality of the Socratic teaching method being used

Original question: "{question}"

Conversation summary (Iteration {current_iteration}/{max_iterations}):
{conversation_summary}

Based on this dialogue, evaluate:

UNDERSTANDING LEVELS:
- "poor": Student shows little to no understanding, many misconceptions
- "developing": Student grasps some concepts but has significant gaps
- "good": Student demonstrates solid understanding with minor gaps
- "excellent": Student shows comprehensive understanding and can explain concepts clearly

VERDICT OPTIONS:
- "continue": Student is learning but needs more guidance
- "satisfactory": Student has reached good understanding, dialogue can end
- "max_reached": Maximum iterations reached, end regardless of understanding

Return your assessment as JSON:
{{
  "verdict": "continue/satisfactory/max_reached",
  "understanding_level": "poor/developing/good/excellent",
  "reasoning": "Brief explanation of your decision",
  "key_insights_gained": ["list", "of", "insights", "student", "demonstrated"],
  "remaining_gaps": ["list", "of", "concepts", "still", "unclear"]
}}

Consider:
- Is the student making meaningful progress?
- Are they connecting concepts better than in earlier rounds?
- Have they addressed the core components of the question?
- Is continued dialogue likely to yield significant additional learning?
"""
    
    response = chat_with_llm(prompt, provider=provider, model=model)
    
    try:
        result = json.loads(response)
        
        # Validate the response structure
        if "verdict" not in result or "understanding_level" not in result:
            raise ValueError("Missing required fields")
        
        # Override verdict if max iterations reached
        if current_iteration >= max_iterations and result["verdict"] == "continue":
            result["verdict"] = "max_reached"
            result["reasoning"] = f"Maximum iterations ({max_iterations}) reached. " + result.get("reasoning", "")
        
        return result
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Dean agent JSON parsing error: {e}")
        print(f"Raw response: {response}")
        
        # Fallback decision logic
        if current_iteration >= max_iterations:
            return {
                "verdict": "max_reached",
                "understanding_level": "developing",
                "reasoning": "Maximum iterations reached, ending dialogue",
                "key_insights_gained": ["Some progress observed"],
                "remaining_gaps": ["Assessment incomplete due to parsing error"]
            }
        else:
            return {
                "verdict": "continue",
                "understanding_level": "developing", 
                "reasoning": "Continuing dialogue due to parsing error",
                "key_insights_gained": ["Assessment incomplete"],
                "remaining_gaps": ["Unable to assess due to error"]
            }

# Legacy function for backward compatibility
def dean_agent_legacy(question: str, student_reply: str, teacher_reply: str, model: str = "llama3") -> dict:
    """Legacy dean agent function for backward compatibility"""
    # Convert to new format
    conversation_history = [
        {"role": "student", "content": student_reply, "iteration": 1},
        {"role": "teacher", "content": teacher_reply, "iteration": 1}
    ]
    
    result = dean_agent(question, conversation_history, 1, 5, "ollama", model)
    
    # Convert back to legacy format
    return {
        "verdict": "Acceptable" if result["verdict"] in ["continue", "satisfactory"] else "Too direct",
        "revision": teacher_reply  # Legacy format doesn't modify teacher response
    }