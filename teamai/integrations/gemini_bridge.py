import json
import os
from pathlib import Path
from google import genai
from google.genai import types

def execute_gemini_handoff(payload_path: Path, patch_path: Path, model_name: str = "gemini-2.5-pro"):
    """Reads the local scout's payload, fires it to Gemini, and writes the resulting patch."""
    if not os.environ.get("GEMINI_API_KEY"):
        print('{"error": "GEMINI_API_KEY is not set. Export it before running."}')
        return

    client = genai.Client()
    
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f'{{"error": "[Errno 2] No such file or directory: \'{payload_path}\'"}}')
        return

    # Reconstruct the exact semantic context for the execution engine
    prompt = f"""TASK: {payload.get('original_task')}
    
CORE DEPENDENCIES:
{json.dumps(payload.get('core_dependencies', []), indent=2)}

DISTILLED CONTEXT:
{json.dumps(payload.get('distilled_context', {}), indent=2)}
"""

    # The ruthlessly strict system prompt to ensure we get a clean patch
    system_instruction = (
        "You are the execution engine for a multi-agent orchestrator.\n"
        "Your constraints:\n"
        "- Read the task and context.\n"
        "- Return ONLY a strict git unified diff patch.\n"
        "- Every file change MUST begin with standard git headers (e.g., `--- a/filepath` and `+++ b/filepath`).\n"
        "- Do not wrap the code in conversational filler. Just output the code block."
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0, # 0.0 for strict, deterministic code generation
            ),
        )
        raw_text = response.text
        
        # Sanitize the output just in case the model adds markdown formatting
        patch_text = raw_text.strip()
        if patch_text.startswith("```patch"):
            patch_text = patch_text[8:]
        elif patch_text.startswith("```diff"):
            patch_text = patch_text[7:]
        elif patch_text.startswith("```"):
            patch_text = patch_text[3:]
        if patch_text.endswith("```"):
            patch_text = patch_text[:-3]
            
        patch_text = patch_text.strip() + "\n"

        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch_text, encoding="utf-8")
        
    except Exception as e:
        print(f'{{"error": "Gemini API Request failed: {str(e)}"}}')