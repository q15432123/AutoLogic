"""
AutoLogic — run_main.py
=======================
Main entry point using the new modular engine architecture.

  python run_main.py
  python run_main.py --text "Build a todo app"
  python run_main.py --image sketch.png --audio notes.mp3
  python run_main.py --skip-deploy
  python run_main.py --server    # Launch WebUI mode
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


def banner():
    print(r"""
    ╔═══════════════════════════════════════════════╗
    ║          A U T O L O G I C   v 0.2            ║
    ║   Multimodal -> Plan -> Code -> Deploy        ║
    ║                                               ║
    ║   Designed by Google Gemini                   ║
    ║   Written by Anthropic Claude Opus            ║
    ╚═══════════════════════════════════════════════╝
    """)


def parse_args():
    parser = argparse.ArgumentParser(description="AutoLogic: Multimodal AI Development Pipeline")
    parser.add_argument("--image", type=str, help="Path to sketch/wireframe image")
    parser.add_argument("--audio", type=str, help="Path to voice note audio file")
    parser.add_argument("--text", type=str, help="Text description of requirements")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip Firebase deployment")
    parser.add_argument("--project-id", type=str, help="Firebase project ID")
    parser.add_argument("--server", action="store_true", help="Launch WebUI server mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()


async def run_pipeline(args):
    from autologic.config import AutoLogicConfig
    from autologic.engine import AutoLogicEngine
    from autologic.models import PipelineContext
    from autologic.logger import setup_logger

    config = AutoLogicConfig.from_file(args.config)
    logger = setup_logger("autologic", config.log_level)

    # Build context from user input
    context = PipelineContext()

    text_prompt = args.text
    if not any([args.text, args.image, args.audio]):
        print("[AutoLogic] No inputs provided. Entering interactive mode.\n")
        text_prompt = input("Describe what you want to build:\n> ").strip()
        if not text_prompt:
            print("[ERROR] No input provided. Exiting.")
            sys.exit(1)

    if text_prompt:
        await context.set("text_prompt", text_prompt)
    if args.image:
        await context.set("image_path", args.image)
    if args.audio:
        await context.set("audio_path", args.audio)
    if args.skip_deploy:
        await context.set("skip_deploy", True)
    if args.project_id:
        await context.set("firebase_project_id", args.project_id)

    # Create engine with default pipeline
    engine = AutoLogicEngine.default_pipeline(config)

    # Register progress handlers
    def on_node_start(node_name, **kwargs):
        print(f"\n{'='*50}")
        print(f"  NODE: {node_name}")
        print(f"{'='*50}")

    def on_node_complete(node_name, result, **kwargs):
        status = "OK" if result.status == "success" else "FAIL"
        print(f"  [{status}] {node_name} ({result.duration_seconds:.1f}s)")

    def on_node_error(node_name, error, **kwargs):
        print(f"  [ERROR] {node_name}: {error}")

    engine.on("node_start", on_node_start)
    engine.on("node_complete", on_node_complete)
    engine.on("node_error", on_node_error)

    # Run pipeline
    result = await engine.run(context)

    # Summary
    print(f"\n{'='*50}")
    print(f"  PIPELINE {'COMPLETE' if result.status == 'success' else 'FAILED'} ({result.total_duration:.1f}s)")
    print(f"{'='*50}")
    for nr in result.node_results:
        s = "OK" if nr.status == "success" else ("SKIP" if nr.status == "skipped" else "FAIL")
        print(f"  [{s}] {nr.node_name:20s} {nr.duration_seconds:.1f}s")
    print(f"\n  Workspace: {result.workspace_dir}")


def main():
    banner()
    args = parse_args()

    if args.server:
        import uvicorn
        from autologic.config import AutoLogicConfig
        config = AutoLogicConfig.from_file(args.config)
        print(f"[AutoLogic] Starting WebUI server on {config.server_host}:{config.server_port}")
        uvicorn.run("server:app", host=config.server_host, port=config.server_port, reload=True)
    else:
        asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
