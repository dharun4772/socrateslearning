# main.py

import pandas as pd
import random
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

from agents.student import student_agent
from agents.teacher import teacher_agent
from agents.dean import dean_agent
from agents.cognitive_state import generate_cognitive_state
from config.personas import persona_traits

# Load questions
df = pd.read_csv("data/data_science_interview_questions.csv")

# Define shared LangGraph state with proper TypedDict
class TeachingState(TypedDict):
    question: str
    category: str
    difficulty: str
    persona: str
    llm_model: str
    student_reply: Optional[str]
    teacher_reply: Optional[str]
    dean_verdict: Optional[str]
    revised_teacher_reply: Optional[str]
    cognitive_state: Optional[dict]

# LangGraph Node Wrappers
def student_node(state: TeachingState) -> TeachingState:
    print(f"Student node received state: {state}")
    try:
        response = student_agent(state["question"], state["persona"], model=state["llm_model"])
        print(response)
        # Return a new state dict with the added response
        return {**state, "student_reply": response}
    except Exception as e:
        print(f"Error in student_node: {e}")
        return {**state, "student_reply": f"Error: {str(e)}"}

def teacher_node(state: TeachingState) -> TeachingState:
    print(f"Teacher node received state keys: {state.keys()}")
    try:
        response = teacher_agent(state["question"], state["student_reply"], model=state["llm_model"])
        return {**state, "teacher_reply": response}
    except Exception as e:
        print(f"Error in teacher_node: {e}")
        return {**state, "teacher_reply": f"Error: {str(e)}"}

def dean_node(state: TeachingState) -> TeachingState:
    print(f"Dean node received state keys: {state.keys()}")
    try:
        result = dean_agent(state["question"], state["student_reply"], state["teacher_reply"], model=state["llm_model"])
        return {
            **state, 
            "dean_verdict": result["verdict"],
            "revised_teacher_reply": result["revision"]
        }
    except Exception as e:
        print(f"Error in dean_node: {e}")
        return {
            **state,
            "dean_verdict": f"Error: {str(e)}",
            "revised_teacher_reply": state.get("teacher_reply", "")
        }

def cognitive_node(state: TeachingState) -> TeachingState:
    print(f"Cognitive node received state keys: {state.keys()}")
    try:
        cognitive_state = generate_cognitive_state(state["persona"])
        return {**state, "cognitive_state": cognitive_state}
    except Exception as e:
        print(f"Error in cognitive_node: {e}")
        return {**state, "cognitive_state": {"error": str(e)}}

# LangGraph Wiring
builder = StateGraph(TeachingState)
builder.add_node("Student", student_node)
builder.add_node("Teacher", teacher_node)
builder.add_node("Dean", dean_node)
builder.add_node("Cognitive", cognitive_node)

builder.set_entry_point("Student")
builder.add_edge("Student", "Teacher")
builder.add_edge("Teacher", "Dean")
builder.add_edge("Dean", "Cognitive")
builder.add_edge("Cognitive", END)

graph = builder.compile()

# Run on all questions
results = []
for idx, row in df.iterrows():
    persona = random.choice(list(persona_traits.keys()))
    initial_state: TeachingState = {
        "question": row["question"],
        "category": row["category"],
        "difficulty": row["difficulty"],
        "persona": persona,
        "llm_model": "llama3",  # change if needed
        "student_reply": None,
        "teacher_reply": None,
        "dean_verdict": None,
        "revised_teacher_reply": None,
        "cognitive_state": None
    }
    
    print(f"Starting Q{idx+1} with initial state: {initial_state}")
    
    try:
        final_state = graph.invoke(initial_state)
        results.append(final_state)
        break
    
        # Save to disk continuously
        with open("outputs/results.jsonl", "a") as f:
            f.write(json.dumps(final_state, default=str) + "\n")

        print(f"✅ Completed Q{idx+1}: {row['question']}")
        
    except Exception as e:
        print(f"❌ Error processing Q{idx+1}: {e}")
        # Continue with next question instead of crashing
        continue

print("🎉 All questions processed.")