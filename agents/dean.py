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
    model: str = "gemini-2.0-flash"
) -> dict:
    """
    Dean agent that evaluates learning progress with strict stopping criteria.
    Ends dialogue when student demonstrates correct understanding.
    """
    
    # Extract student responses to analyze learning progression
    student_responses = [entry for entry in conversation_history if entry["role"] == "student"]
    teacher_responses = [entry for entry in conversation_history if entry["role"] == "teacher"]
    
    # Get the most recent student response for detailed analysis
    latest_student_response = student_responses[-1]["content"] if student_responses else ""
    
    # Build conversation summary (last 2-3 exchanges for context)
    conversation_summary = ""
    recent_pairs = list(zip(student_responses, teacher_responses))[-2:]  # Last 2 rounds
    
    for i, (student, teacher) in enumerate(recent_pairs, len(recent_pairs)-1):
        conversation_summary += f"Round {i+1}:\n"
        conversation_summary += f"Student: {student['content']}\n"
        conversation_summary += f"Teacher: {teacher['content']}\n\n"
    
    prompt = f"""
You are the Dean evaluating a Socratic dialogue session. Your PRIMARY job is to determine if the student has CORRECTLY ANSWERED the original question.

CRITICAL: If the student demonstrates correct understanding, END the dialogue immediately. Do not continue just because they ask follow-up questions.

Original question: "{question}"

Most recent student response: "{latest_student_response}"

Recent conversation:
{conversation_summary}

Evaluation criteria (in order of priority):

1. ANSWER CORRECTNESS: Does the student's response correctly address the core question?
   - Are the main concepts explained accurately?
   - Have they covered the essential components?
   - Is their understanding fundamentally sound?

2. COMPLETENESS: Is the answer reasonably complete for the question level?
   - No need for PhD-level depth on basic questions
   - Match completeness expectations to question difficulty

3. UNDERSTANDING DEMONSTRATION: Can they explain the concept clearly?
   - Do they show genuine comprehension vs. memorization?
   - Can they connect related ideas?

VERDICT RULES:
- "satisfactory": Student has correctly answered the question with good understanding
- "continue": Student needs guidance on core concepts OR has significant misconceptions  
- "max_reached": Hit iteration limit regardless of understanding

UNDERSTANDING LEVELS:
- "excellent": Perfect understanding, clear explanation, connects concepts
- "good": Correct answer with solid understanding, minor gaps OK
- "developing": Partial understanding, some correct elements, needs guidance
- "poor": Little understanding, major misconceptions, far from correct answer

Return assessment as JSON:
{{
  "verdict": "continue/satisfactory/max_reached",
  "understanding_level": "poor/developing/good/excellent", 
  "reasoning": "Brief explanation focusing on answer correctness",
  "answer_correctness": "correct/partially_correct/incorrect",
  "key_insights_gained": ["specific correct concepts demonstrated"],
  "remaining_gaps": ["only list if verdict is continue"]
}}

IMPORTANT: 
- If understanding_level is "good" or "excellent" AND answer_correctness is "correct", verdict should be "satisfactory"
- Don't extend dialogue just because student asks deeper questions
- Focus on whether they've answered the ORIGINAL question, not potential follow-ups
- Iteration {current_iteration}/{max_iterations}
"""
    
    response = chat_with_llm(prompt, provider=provider, model=model)
    
    try:
        result = json.loads(response)
        
        # Validate the response structure
        required_fields = ["verdict", "understanding_level", "answer_correctness"]
        if not all(field in result for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")
        
        # Enforce stopping rules based on correctness
        if (result.get("answer_correctness") == "correct" and 
            result.get("understanding_level") in ["good", "excellent"] and
            result.get("verdict") == "continue"):
            
            print("🎯 Dean override: Student has correct answer, ending dialogue")
            result["verdict"] = "satisfactory"
            result["reasoning"] = "Answer is correct with good understanding. " + result.get("reasoning", "")
        
        # Override verdict if max iterations reached
        if current_iteration >= max_iterations and result["verdict"] == "continue":
            result["verdict"] = "max_reached"
            result["reasoning"] = f"Maximum iterations ({max_iterations}) reached. " + result.get("reasoning", "")
        
        return result
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Dean agent JSON parsing error: {e}")
        print(f"Raw response: {response}")
        
        # Conservative fallback - end if we've made several attempts
        if current_iteration >= max_iterations:
            return {
                "verdict": "max_reached",
                "understanding_level": "developing",
                "answer_correctness": "unknown",
                "reasoning": "Maximum iterations reached, ending dialogue",
                "key_insights_gained": ["Assessment incomplete due to parsing error"],
                "remaining_gaps": ["Unable to assess due to error"]
            }
        elif current_iteration >= 3:  # Conservative stopping after 3 rounds if parsing fails
            return {
                "verdict": "satisfactory",
                "understanding_level": "developing",
                "answer_correctness": "unknown", 
                "reasoning": "Ending dialogue due to parsing error after multiple rounds",
                "key_insights_gained": ["Some progress observed"],
                "remaining_gaps": ["Assessment incomplete"]
            }
        else:
            return {
                "verdict": "continue",
                "understanding_level": "developing",
                "answer_correctness": "unknown",
                "reasoning": "Continuing dialogue despite parsing error",
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