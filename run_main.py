"""
AutoLogic — run_main.py
=======================
Main entry point that chains the three modules:

  Module 1 (multi_ingest)  ->  Multimodal input processing
  Module 2 (core_gen)      ->  Gemini planning + agent code generation
  Module 3 (auto_deploy)   ->  Package & deploy to Firebase

Usage:
  python run_main.py
  python run_main.py --text "Build a todo app"
  python run_main.py --image sketch.png --audio notes.mp3 --text "portfolio site"
  python run_main.py --skip-deploy   # generate code only, no Firebase deploy
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")


def banner():
    print(r"""
    ╔═══════════════════════════════════════════════╗
    ║          A U T O L O G I C   v 0.1            ║
    ║   Multimodal -> Plan -> Code -> Deploy        ║
    ╚═══════════════════════════════════════════════╝
    """)


def parse_args():
    parser = argparse.ArgumentParser(description="AutoLogic: Multimodal-to-Deployment Pipeline")
    parser.add_argument("--image", type=str, help="Path to sketch/wireframe image")
    parser.add_argument("--audio", type=str, help="Path to voice note audio file")
    parser.add_argument("--text", type=str, help="Text description of requirements")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip Firebase deployment")
    parser.add_argument("--project-id", type=str, help="Firebase project ID (overrides .env)")
    return parser.parse_args()


def main():
    banner()
    args = parse_args()

    # ── Validate API key ──
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set. Create a .env file with your key.")
        print("        Get one at: https://aistudio.google.com/apikey")
        sys.exit(1)

    # ── Interactive mode if no args ──
    text_prompt = args.text
    image_path = args.image
    audio_path = args.audio

    if not any([text_prompt, image_path, audio_path]):
        print("[AutoLogic] No inputs provided. Entering interactive mode.\n")
        text_prompt = input("Describe what you want to build:\n> ").strip()
        if not text_prompt:
            print("[ERROR] No input provided. Exiting.")
            sys.exit(1)

        img_input = input("\nPath to sketch image (or press Enter to skip):\n> ").strip()
        if img_input:
            image_path = img_input

        audio_input = input("\nPath to voice note (or press Enter to skip):\n> ").strip()
        if audio_input:
            audio_path = audio_input

    # ══════════════════════════════════════════
    # MODULE 1: Multimodal Ingestion
    # ══════════════════════════════════════════
    print("\n" + "=" * 50)
    print("  MODULE 1: Multimodal Ingestion")
    print("=" * 50)
    t0 = time.time()

    from multi_ingest import ingest_requirements

    context = ingest_requirements(
        image_path=image_path,
        audio_path=audio_path,
        text_prompt=text_prompt,
    )

    print(f"\n  Module 1 complete ({time.time() - t0:.1f}s)")

    # ══════════════════════════════════════════
    # MODULE 2: Gemini Planning + Code Gen
    # ══════════════════════════════════════════
    print("\n" + "=" * 50)
    print("  MODULE 2: Planning & Code Generation")
    print("=" * 50)
    t1 = time.time()

    from core_gen import execute_all_tasks, orchestrate_planning

    # Phase 2a: Planning
    tasks = orchestrate_planning(
        multimodal_context=context,
        api_key=api_key,
        image_path=context.get("image_path"),
    )

    # Phase 2b: Agent execution (code generation)
    results = execute_all_tasks(tasks, api_key)

    print(f"\n  Module 2 complete ({time.time() - t1:.1f}s)")
    print(f"  Tasks executed: {len(results)}")
    for r in results:
        files = r.get("files", [])
        print(f"    {r.get('agent', '?'):8s} -> {len(files)} file(s)")

    # ══════════════════════════════════════════
    # MODULE 3: Deploy
    # ══════════════════════════════════════════
    workspace_dir = str(Path(__file__).parent / "_workspaces")

    if args.skip_deploy:
        print("\n" + "=" * 50)
        print("  MODULE 3: Deployment SKIPPED (--skip-deploy)")
        print("=" * 50)
        print(f"\n  Generated files are in: {workspace_dir}")
    else:
        print("\n" + "=" * 50)
        print("  MODULE 3: Packaging & Deployment")
        print("=" * 50)
        t2 = time.time()

        from auto_deploy import deploy_to_firebase, generate_deploy_report

        project_id = args.project_id or os.getenv("FIREBASE_PROJECT_ID", "")

        if not project_id:
            print("\n[WARNING] No Firebase project ID set.")
            project_id = input("Enter Firebase project ID (or press Enter to skip deploy):\n> ").strip()

        if project_id:
            deploy_result = deploy_to_firebase(workspace_dir, project_id)
            report = generate_deploy_report(deploy_result, workspace_dir)
            print(report)
            print(f"\n  Module 3 complete ({time.time() - t2:.1f}s)")
        else:
            print("\n  Deployment skipped. Files are in: {workspace_dir}")

    # ── Summary ──
    total = time.time() - t0
    print("\n" + "=" * 50)
    print(f"  PIPELINE COMPLETE ({total:.1f}s total)")
    print("=" * 50)
    print(f"  Workspace: {workspace_dir}")
    print()


if __name__ == "__main__":
    main()
