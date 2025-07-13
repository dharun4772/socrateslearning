# utils/ollama_client.py

import subprocess
import json

def ollama_chat(prompt: str, model: str = "llama3", system: str = "") -> str:
    """
    Sends a prompt to a local Ollama model and returns the generated response.
    Requires Ollama to be running locally.
    """
    try:
        command = ["ollama", "run", model]
        full_prompt = f"{system}\n{prompt}" if system else prompt

        # Use subprocess to run the command with the prompt piped in
        result = subprocess.run(
            command,
            input=full_prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        output = result.stdout.decode("utf-8")
        return output.strip()

    except Exception as e:
        return f"ERROR: {e}"
