"""
Module 2: core_gen — Core Orchestrator & Gemini Gateway

Takes consolidated context from Module 1, sends it to Gemini for analysis,
decomposes the goal into subtasks, and assigns them to specialized agents.
Agents execute locally by writing files into _workspaces/.
"""

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai

WORKSPACE_DIR = Path(__file__).parent / "_workspaces"

# ──────────────────────────────────────────────
# Gemini interaction
# ──────────────────────────────────────────────

PLANNING_PROMPT = textwrap.dedent("""\
You are OmniFlow's planning engine. The user wants to build a web application.
Your job is to analyze their requirements (text + optional sketch) and produce
a structured JSON task list that local coding agents will execute.

## Agent Roster
- **Boss**: Creates the initial plan and final review.
- **Jordan**: Frontend specialist (HTML, CSS, JS). Writes files under `frontend/`.
- **Alex**: Backend specialist (Python, Node, APIs). Writes files under `backend/`.
- **Sam**: DevOps / config. Writes deployment configs, README, etc.

## Output Format
Return ONLY a JSON array (no markdown fences). Each element:
```
{
  "agent": "Boss" | "Jordan" | "Alex" | "Sam",
  "task": "<concise task title>",
  "description": "<detailed instructions including exact file paths and content hints>",
  "depends_on": ["<agent whose task must finish first>"],
  "tools": ["write_file", "run_command"],
  "status": "pending"
}
```

Rules:
1. Boss always has the first task (plan) and last task (review).
2. Jordan and Alex tasks run in parallel where possible.
3. Sam handles firebase.json, README, and any config.
4. Be specific about file paths relative to the workspace root.
5. Generate real, working code instructions — not placeholders.

## User Requirements
{context}
""")

CODING_PROMPT = textwrap.dedent("""\
You are agent **{agent}** in the OmniFlow system.
Your task: {task}

Detailed instructions:
{description}

Generate the COMPLETE file contents needed. Return ONLY a JSON object:
```
{{
  "files": {{
    "relative/path/to/file.ext": "full file content here",
    ...
  }},
  "commands": ["optional shell commands to run"],
  "notes": "any notes for the next agent"
}}
```

Rules:
- Write production-quality code, not stubs.
- Use semantic HTML5 + modern CSS (flexbox/grid) for frontend.
- For backend, prefer lightweight solutions (Flask / Express).
- Return ONLY valid JSON, no markdown fences.
""")


def _configure_gemini(api_key: str) -> genai.GenerativeModel:
    """Configure and return a Gemini model instance."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    return model


def _parse_json_response(text: str) -> Any:
    """Extract JSON from Gemini's response, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return json.loads(cleaned)


def _build_multimodal_parts(
    context_text: str, image_path: Optional[str] = None
) -> list:
    """Build content parts for Gemini, optionally including an image."""
    parts = []
    if image_path and Path(image_path).exists():
        import PIL.Image
        img = PIL.Image.open(image_path)
        parts.append(img)
        parts.append(
            "\n[The above image is the user's hand-drawn sketch/wireframe. "
            "Analyze it to understand the intended UI layout.]\n\n"
        )
    parts.append(context_text)
    return parts


# ──────────────────────────────────────────────
# Phase 1: Planning
# ──────────────────────────────────────────────

def orchestrate_planning(
    multimodal_context: Dict[str, Any],
    api_key: str,
    image_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Uses Gemini to generate a structured multi-agent task list.

    Args:
        multimodal_context: Output from multi_ingest.ingest_requirements()
        api_key: Gemini API key
        image_path: Optional path to sketch image for visual analysis

    Returns:
        List of task dicts assigned to agents.
    """
    model = _configure_gemini(api_key)

    consolidated = multimodal_context.get("consolidated_context", "")
    prompt_text = PLANNING_PROMPT.format(context=consolidated)

    parts = _build_multimodal_parts(prompt_text, image_path)

    print("[core_gen] Sending planning request to Gemini...")
    response = model.generate_content(parts)
    print(f"[core_gen] Gemini responded ({len(response.text)} chars)")

    tasks = _parse_json_response(response.text)

    # Mark Boss's first task as complete (planning itself)
    if tasks and tasks[0].get("agent") == "Boss":
        tasks[0]["status"] = "complete"

    print(f"[core_gen] Plan generated: {len(tasks)} tasks")
    for i, t in enumerate(tasks):
        deps = ", ".join(t.get("depends_on", [])) or "none"
        print(f"  [{i+1}] {t['agent']:8s} | {t['task']} (deps: {deps})")

    return tasks


# ──────────────────────────────────────────────
# Phase 2: Code Generation (Agent Execution)
# ──────────────────────────────────────────────

def execute_agent_task(
    task: Dict[str, Any],
    api_key: str,
    workspace: Path,
) -> Dict[str, Any]:
    """
    Have Gemini act as the specified agent and generate code/files.

    Args:
        task: A single task dict from the plan.
        api_key: Gemini API key.
        workspace: Path to write output files.

    Returns:
        Agent output dict with files written and any commands.
    """
    model = _configure_gemini(api_key)

    prompt = CODING_PROMPT.format(
        agent=task["agent"],
        task=task["task"],
        description=task.get("description", task["task"]),
    )

    print(f"\n[core_gen] Agent {task['agent']} executing: {task['task']}")
    response = model.generate_content(prompt)

    try:
        output = _parse_json_response(response.text)
    except json.JSONDecodeError:
        # Fallback: save raw response as a file
        print(f"[core_gen] Warning: Agent {task['agent']} returned non-JSON, saving raw output")
        fallback_path = workspace / f"{task['agent'].lower()}_output.txt"
        fallback_path.write_text(response.text, encoding="utf-8")
        return {"files": {str(fallback_path): "(raw output)"}, "commands": [], "notes": "non-JSON response"}

    # Write files to workspace
    files_written = []
    for rel_path, content in output.get("files", {}).items():
        full_path = workspace / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        files_written.append(str(rel_path))
        print(f"  -> Wrote: {rel_path}")

    task["status"] = "complete"

    return {
        "agent": task["agent"],
        "files": files_written,
        "commands": output.get("commands", []),
        "notes": output.get("notes", ""),
    }


def execute_all_tasks(
    tasks: List[Dict[str, Any]],
    api_key: str,
) -> List[Dict[str, Any]]:
    """
    Execute all agent tasks sequentially, respecting dependencies.
    Writes all output to _workspaces/.

    Returns:
        List of execution results per agent.
    """
    workspace = WORKSPACE_DIR
    workspace.mkdir(parents=True, exist_ok=True)

    results = []
    completed_agents = set()

    for task in tasks:
        # Skip Boss's initial planning task (already complete)
        if task.get("status") == "complete":
            completed_agents.add(task["agent"])
            results.append({
                "agent": task["agent"],
                "task": task["task"],
                "status": "complete (planning)",
                "files": [],
            })
            continue

        # Check dependencies
        deps = task.get("depends_on", [])
        unmet = [d for d in deps if d not in completed_agents]
        if unmet:
            print(f"[core_gen] Warning: {task['agent']} has unmet deps: {unmet}, proceeding anyway")

        result = execute_agent_task(task, api_key, workspace)
        completed_agents.add(task["agent"])
        results.append(result)

    print(f"\n[core_gen] All {len(results)} tasks executed.")
    print(f"[core_gen] Workspace: {workspace.resolve()}")

    return results


# ──────────────────────────────────────────────
# CLI test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in .env")
        exit(1)

    # Quick test with text-only input
    mock_context = {
        "consolidated_context": (
            "## User Requirements (Text)\n"
            "Build a simple portfolio website with a hero section, "
            "project cards, and a contact form."
        )
    }

    tasks = orchestrate_planning(mock_context, api_key)
    results = execute_all_tasks(tasks, api_key)

    print("\n=== Results ===")
    for r in results:
        print(f"  {r.get('agent', '?'):8s} | files: {r.get('files', [])}")
