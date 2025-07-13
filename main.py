# main.py with LLM provider selection

import pandas as pd
import random
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List

from agents.student import student_agent
from agents.teacher import teacher_agent  
from agents.dean import dean_agent
from agents.cognitive_state import generate_cognitive_state
from config.personas import persona_traits
from config.llm_config import LLMProvider

# Load questions
df = pd.read_csv("data/data_science_interview_questions.csv")

# Configuration - Change these to switch providers
LLM_PROVIDER: LLMProvider = "gemini"  # or "ollama"
LLM_MODEL = "gemini-2.5-flash"  # Free Gemini model

# Define shared LangGraph state
class TeachingState(TypedDict):
    question: str
    category: str
    difficulty: str
    persona: str
    llm_provider: str
    llm_model: str
    conversation_history: List[dict]
    current_student_reply: Optional[str]
    current_teacher_reply: Optional[str]
    dean_verdict: Optional[str]
    understanding_level: Optional[str]
    iteration_count: int
    max_iterations: int
    final_assessment: Optional[dict]
    cognitive_state: Optional[dict]

# LangGraph Node Wrappers
def student_node(state: TeachingState) -> TeachingState:
    print(f"=== Student Iteration {state['iteration_count']} ===")
    
    context = state.get("conversation_history", [])
    last_teacher_response = context[-1]["content"] if context and context[-1]["role"] == "teacher" else None
    
    try:
        if last_teacher_response:
            response = student_agent(
                question=state["question"], 
                persona=state["persona"], 
                teacher_guidance=last_teacher_response,
                conversation_history=context,
                provider=state["llm_provider"],
                model=state["llm_model"]
            )
        else:
            response = student_agent(
                question=state["question"], 
                persona=state["persona"], 
                provider=state["llm_provider"],
                model=state["llm_model"]
            )
        
        new_history = context + [{"role": "student", "content": response, "iteration": state["iteration_count"]}]
        
        return {
            **state, 
            "current_student_reply": response,
            "conversation_history": new_history
        }
    except Exception as e:
        print(f"Error in student_node: {e}")
        return {**state, "current_student_reply": f"Error: {str(e)}"}

def teacher_node(state: TeachingState) -> TeachingState:
    print(f"=== Teacher Iteration {state['iteration_count']} ===")
    
    try:
        response = teacher_agent(
            question=state["question"],
            student_reply=state["current_student_reply"],
            conversation_history=state.get("conversation_history", []),
            iteration=state["iteration_count"],
            provider=state["llm_provider"],
            model=state["llm_model"]
        )
        
        new_history = state["conversation_history"] + [{"role": "teacher", "content": response, "iteration": state["iteration_count"]}]
        
        return {
            **state, 
            "current_teacher_reply": response,
            "conversation_history": new_history
        }
    except Exception as e:
        print(f"Error in teacher_node: {e}")
        return {**state, "current_teacher_reply": f"Error: {str(e)}"}

def dean_node(state: TeachingState) -> TeachingState:
    print(f"=== Dean Assessment Iteration {state['iteration_count']} ===")
    
    try:
        result = dean_agent(
            question=state["question"],
            conversation_history=state["conversation_history"],
            current_iteration=state["iteration_count"],
            max_iterations=state["max_iterations"],
            provider=state["llm_provider"],
            model=state["llm_model"]
        )
        
        return {
            **state,
            "dean_verdict": result["verdict"],
            "understanding_level": result["understanding_level"],
            "iteration_count": state["iteration_count"] + 1
        }
    except Exception as e:
        print(f"Error in dean_node: {e}")
        return {
            **state,
            "dean_verdict": "continue",
            "understanding_level": "unknown",
            "iteration_count": state["iteration_count"] + 1
        }

def cognitive_node(state: TeachingState) -> TeachingState:
    print(f"=== Final Cognitive Assessment ===")
    
    try:
        cognitive_state = generate_cognitive_state(
            persona=state["persona"],
            conversation_history=state["conversation_history"],
            final_understanding=state["understanding_level"],
            provider=state["llm_provider"],
            model=state["llm_model"]
        )
        
        final_assessment = {
            "total_iterations": state["iteration_count"] - 1,
            "final_understanding_level": state["understanding_level"],
            "conversation_length": len(state["conversation_history"]),
            "llm_provider": state["llm_provider"],
            "llm_model": state["llm_model"],
            "learning_progression": analyze_learning_progression(state["conversation_history"])
        }
        
        return {
            **state, 
            "cognitive_state": cognitive_state,
            "final_assessment": final_assessment
        }
    except Exception as e:
        print(f"Error in cognitive_node: {e}")
        return {**state, "cognitive_state": {"error": str(e)}}

def analyze_learning_progression(history: List[dict]) -> List[str]:
    """Analyze how understanding progressed through the conversation"""
    student_responses = [entry for entry in history if entry["role"] == "student"]
    return [f"Iteration {r['iteration']}: {r['content'][:100]}..." for r in student_responses]

def should_continue_dialogue(state: TeachingState) -> str:
    """Determine whether to continue the Socratic dialogue or end"""
    
    verdict = state.get("dean_verdict", "continue")
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 5)
    
    print(f"Dean verdict: {verdict}, Iteration: {iteration_count}/{max_iterations}")
    
    if verdict == "satisfactory":
        print("✅ Student has reached satisfactory understanding")
        return "cognitive"
    elif iteration_count >= max_iterations:
        print("⏰ Maximum iterations reached")
        return "cognitive" 
    elif verdict == "continue":
        print("🔄 Continue Socratic dialogue")
        return "student"
    else:
        print("🎯 Moving to final assessment")
        return "cognitive"

# LangGraph Wiring
builder = StateGraph(TeachingState)
builder.add_node("student", student_node)
builder.add_node("teacher", teacher_node) 
builder.add_node("dean", dean_node)
builder.add_node("cognitive", cognitive_node)

builder.set_entry_point("student")
builder.add_edge("student", "teacher")
builder.add_edge("teacher", "dean")
builder.add_conditional_edges(
    "dean",
    should_continue_dialogue,
    {
        "student": "student",
        "cognitive": "cognitive"
    }
)
builder.add_edge("cognitive", END)

graph = builder.compile()

# Run Socratic dialogues
print(f"🤖 Using {LLM_PROVIDER.upper()} with model: {LLM_MODEL}")
print("=" * 50)
import time
results = []
for idx, row in df.iterrows():
    if idx == 50:
        break
    persona = random.choice(list(persona_traits.keys()))
    initial_state: TeachingState = {
        "question": row["question"],
        "category": row["category"], 
        "difficulty": row["difficulty"],
        "persona": persona,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "conversation_history": [],
        "current_student_reply": None,
        "current_teacher_reply": None,
        "dean_verdict": None,
        "understanding_level": None,
        "iteration_count": 1,
        "max_iterations": 5,  # Reasonable limit for free API
        "final_assessment": None,
        "cognitive_state": None
    }
    
    print(f"\n🎓 Starting Socratic Dialogue Q{idx+1}: {row['question']}")
    print(f"👤 Student Persona: {persona}")
    
    try:
        final_state = graph.invoke(initial_state)
        results.append(final_state)
        # Save results with provider info
        output_file = f"outputs/socratic_results_{LLM_PROVIDER}.jsonl"
        with open(output_file, "a") as f:
            f.write(json.dumps(final_state, default=str) + "\n")

        print(f"✅ Completed Q{idx+1} after {final_state['final_assessment']['total_iterations']} iterations")
        print(f"📊 Final understanding: {final_state['understanding_level']}")
        
    except Exception as e:
        print(f"❌ Error processing Q{idx+1}: {e}")
        continue
    time.sleep(60)
print(f"\n🎉 All Socratic dialogues completed using {LLM_PROVIDER.upper()}!")
print(f"📁 Results saved to: outputs/socratic_results_{LLM_PROVIDER}.jsonl")