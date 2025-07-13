# agents/cognitive_state.py

from config.llm_config import chat_with_llm, LLMProvider
from config.personas import persona_traits
from typing import List, Dict, Optional
import json

def generate_cognitive_state(
    persona: str,
    provider: LLMProvider = "gemini",
    model: str = "gemini-2.5-flash", 
    conversation_history: Optional[List[Dict]] = None,
    final_understanding: Optional[str] = None
) -> dict:
    """
    Generate a cognitive state assessment based on the student's learning journey.
    
    Args:
        persona: Student personality type
        provider: LLM provider ("gemini" or "ollama")
        model: Model name
        conversation_history: Full conversation between student and teacher
        final_understanding: Final understanding level assessed by dean
    """
    
    traits = persona_traits[persona]
    
    # Analyze conversation progression if available
    learning_journey = ""
    if conversation_history:
        student_responses = [entry for entry in conversation_history if entry["role"] == "student"]
        learning_journey = "\n\nLearning Journey:\n"
        for i, response in enumerate(student_responses, 1):
            content = response['content'][:150] + "..." if len(response['content']) > 150 else response['content']
            learning_journey += f"Round {i}: {content}\n"
    
    prompt = f"""
You are a cognitive scientist analyzing a student's learning state and mental model development.

Student Profile:
- Persona: {persona}
- Problem Understanding: {traits['Problem Understanding']}
- Instruction Understanding: {traits['Instruction Understanding']}  
- Calculation: {traits['Calculation']}
- Knowledge Mastery: {traits['Knowledge Mastery']}
- Thirst for Learning: {traits['Thirst for Learning']}

Final Understanding Level: {final_understanding or 'Not assessed'}

{learning_journey}

Based on this information, generate a comprehensive cognitive state assessment. Return as JSON:

{{
  "mental_model_development": {{
    "initial_state": "Description of student's starting knowledge state",
    "final_state": "Description of student's ending knowledge state", 
    "key_breakthroughs": ["Moments where understanding clicked"],
    "persistent_misconceptions": ["Concepts still unclear or incorrect"]
  }},
  "learning_patterns": {{
    "preferred_learning_style": "How this student learns best",
    "response_to_guidance": "How well they respond to Socratic questioning",
    "question_asking_behavior": "What types of questions they ask",
    "confidence_progression": "How their confidence changed"
  }},
  "cognitive_skills_demonstrated": {{
    "analytical_thinking": "poor/developing/good/excellent",
    "conceptual_connections": "poor/developing/good/excellent", 
    "self_reflection": "poor/developing/good/excellent",
    "knowledge_application": "poor/developing/good/excellent"
  }},
  "persona_consistency": {{
    "trait_alignment": "How well responses matched expected persona",
    "authentic_behaviors": ["Behaviors that matched the persona"],
    "persona_development": "How the persona evolved during learning"
  }},
  "recommendations": {{
    "next_learning_steps": ["What should this student study next"],
    "teaching_strategies": ["What teaching methods work best for this student"],
    "knowledge_gaps": ["Specific areas needing more work"]
  }},
  "overall_assessment": {{
    "learning_effectiveness": "poor/fair/good/excellent",
    "engagement_level": "low/medium/high", 
    "readiness_for_advanced_topics": "yes/no/partial",
    "summary": "2-3 sentence overall assessment"
  }}
}}

Focus on:
- How the student's thinking evolved throughout the dialogue
- Whether their responses were consistent with their persona
- What cognitive strategies they used
- How effectively they learned from Socratic questioning
"""
    
    response = chat_with_llm(prompt, provider=provider, model=model)
    
    try:
        cognitive_state = json.loads(response)
        return cognitive_state
        
    except json.JSONDecodeError as e:
        print(f"Cognitive state JSON parsing error: {e}")
        print(f"Raw response: {response}")
        
        # Fallback cognitive state
        return {
            "mental_model_development": {
                "initial_state": f"Student began with {traits['Knowledge Mastery'].lower()} knowledge level",
                "final_state": f"Final understanding: {final_understanding or 'unknown'}",
                "key_breakthroughs": ["Assessment incomplete due to parsing error"],
                "persistent_misconceptions": ["Unable to assess"]
            },
            "learning_patterns": {
                "preferred_learning_style": f"Based on {persona} persona characteristics",
                "response_to_guidance": "Unable to assess due to error",
                "question_asking_behavior": "Unknown",
                "confidence_progression": "Unable to track"
            },
            "cognitive_skills_demonstrated": {
                "analytical_thinking": "unknown",
                "conceptual_connections": "unknown",
                "self_reflection": "unknown", 
                "knowledge_application": "unknown"
            },
            "persona_consistency": {
                "trait_alignment": "Unable to assess",
                "authentic_behaviors": ["Assessment incomplete"],
                "persona_development": "Unknown"
            },
            "recommendations": {
                "next_learning_steps": ["Retry assessment with proper JSON formatting"],
                "teaching_strategies": ["Standard Socratic method"],
                "knowledge_gaps": ["Assessment incomplete"]
            },
            "overall_assessment": {
                "learning_effectiveness": "unknown",
                "engagement_level": "unknown",
                "readiness_for_advanced_topics": "unknown",
                "summary": "Cognitive assessment incomplete due to technical error."
            },
            "error": str(e)
        }

# Legacy function for backward compatibility  
def generate_cognitive_state_legacy(persona: str, model: str = "llama3") -> dict:
    """Legacy cognitive state function for backward compatibility"""
    return generate_cognitive_state(persona, "ollama", model)