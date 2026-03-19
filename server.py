"""
AutoLogic — server.py
======================
FastAPI WebUI server with WebSocket-based real-time pipeline progress.

Usage:
    python server.py
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Features:
    - Serves index.html at /
    - WebSocket /ws streams real-time pipeline progress
    - POST /api/run accepts text_prompt + optional image/audio files
    - Demo/simulation mode when GEMINI_API_KEY is not set
    - Production-quality error handling and graceful shutdown
"""

import asyncio
import json
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

# ── New engine integration (v0.2) ──
# Try to import the new autologic package. Falls back gracefully so
# the existing pipeline still works when the package is not installed.
try:
    from autologic.engine import AutoLogicEngine
    from autologic.config import AutoLogicConfig
    from autologic.models import PipelineContext
    from autologic.logger import setup_logger as _setup_logger
    _HAS_AUTOLOGIC_ENGINE = True
except ImportError:
    _HAS_AUTOLOGIC_ENGINE = False

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "_uploads"
WORKSPACE_DIR = BASE_DIR / "_workspaces"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_DEMO_MODE = not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autologic.server")


# ──────────────────────────────────────────────
# WebSocket Connection Manager
# ──────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections, keyed by run_id."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str = "global"):
        await websocket.accept()
        if run_id not in self._connections:
            self._connections[run_id] = []
        self._connections[run_id].append(websocket)
        logger.info(f"WebSocket connected: run_id={run_id} (total={self._total_count()})")

    def disconnect(self, websocket: WebSocket, run_id: str = "global"):
        if run_id in self._connections:
            self._connections[run_id] = [
                ws for ws in self._connections[run_id] if ws != websocket
            ]
            if not self._connections[run_id]:
                del self._connections[run_id]
        logger.info(f"WebSocket disconnected: run_id={run_id} (total={self._total_count()})")

    async def broadcast(self, run_id: str, message: dict):
        """Send a message to all WebSockets listening on a run_id, plus 'global'."""
        targets = []
        if run_id in self._connections:
            targets.extend(self._connections[run_id])
        if "global" in self._connections and run_id != "global":
            targets.extend(self._connections["global"])

        payload = json.dumps(message, ensure_ascii=False)
        dead = []
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            for key in list(self._connections.keys()):
                if ws in self._connections[key]:
                    self._connections[key].remove(ws)

    def _total_count(self) -> int:
        return sum(len(v) for v in self._connections.values())


manager = ConnectionManager()


# ──────────────────────────────────────────────
# Progress message helper
# ──────────────────────────────────────────────

def _progress_msg(
    module: int,
    agent: str,
    message: str,
    progress: int,
    run_id: str,
    *,
    status: str = "running",
    files: Optional[list] = None,
) -> dict:
    return {
        "run_id": run_id,
        "module": module,
        "agent": agent,
        "message": message,
        "progress": max(0, min(100, progress)),
        "status": status,
        "timestamp": time.time(),
        "files": files or [],
    }


# ──────────────────────────────────────────────
# Simulation pipeline (demo mode)
# ──────────────────────────────────────────────

async def _run_simulation(
    run_id: str,
    text_prompt: str,
    has_image: bool,
    has_audio: bool,
):
    """
    Simulate the full 3-module pipeline with realistic delays and messages.
    Pushes real-time progress via WebSocket.
    """
    prompt_len = len(text_prompt)

    async def emit(module, agent, message, progress, **kwargs):
        msg = _progress_msg(module, agent, message, progress, run_id, **kwargs)
        await manager.broadcast(run_id, msg)
        logger.info(f"[run={run_id[:8]}] M{module} {agent}: {message}")

    try:
        # ── MODULE 1: Multimodal Ingestion ──
        await emit(1, "system", "Pipeline started", 0, status="running")
        await asyncio.sleep(0.3)

        await emit(1, "system", f"Module 1: Processing text input... ({prompt_len} chars)", 5)
        await asyncio.sleep(0.4 + random.uniform(0.1, 0.3))

        if has_image:
            await emit(1, "system", "Module 1: Analyzing sketch image with OpenCV...", 10)
            await asyncio.sleep(0.6 + random.uniform(0.2, 0.5))
            await emit(1, "system", "Module 1: Image encoded to base64 (ready for Gemini)", 15)
            await asyncio.sleep(0.2)

        if has_audio:
            await emit(1, "system", "Module 1: Loading Whisper model...", 12)
            await asyncio.sleep(1.0 + random.uniform(0.3, 0.7))
            await emit(1, "system", "Module 1: Transcribing audio...", 16)
            await asyncio.sleep(1.5 + random.uniform(0.5, 1.0))
            transcript_len = random.randint(80, 250)
            await emit(1, "system", f"Module 1: Transcription complete ({transcript_len} chars)", 20)
            await asyncio.sleep(0.2)

        await emit(1, "system", "Module 1: Context consolidated successfully", 22)
        await asyncio.sleep(0.3)

        # ── MODULE 2: Planning & Code Generation ──
        await emit(2, "Boss", "Module 2: Sending planning request to Gemini...", 25)
        await asyncio.sleep(1.5 + random.uniform(0.5, 1.5))

        gemini_chars = random.randint(3500, 6500)
        await emit(2, "Boss", f"Module 2: Gemini responded ({gemini_chars:,} chars)", 30)
        await asyncio.sleep(0.3)

        num_tasks = random.randint(5, 8)
        await emit(2, "Boss", f"Module 2: Plan generated — {num_tasks} tasks across 4 agents", 33)
        await asyncio.sleep(0.4)

        # Task list
        tasks = [
            ("Boss", "Initial Planning & Architecture", []),
            ("Jordan", "Frontend: Build HTML structure", ["Boss"]),
            ("Jordan", "Frontend: CSS styling + responsive layout", ["Boss"]),
            ("Jordan", "Frontend: JavaScript interactivity", ["Boss"]),
            ("Alex", "Backend: API endpoints", ["Boss"]),
            ("Alex", "Backend: Data models & validation", ["Boss"]),
            ("Sam", "DevOps: Firebase config + README", ["Jordan", "Alex"]),
            ("Boss", "Final Review & QA", ["Jordan", "Alex", "Sam"]),
        ][:num_tasks]

        for i, (agent, task, deps) in enumerate(tasks):
            dep_str = ", ".join(deps) if deps else "none"
            await emit(
                2, "Boss",
                f"  [{i+1}] {agent:8s} | {task} (deps: {dep_str})",
                35 + i,
            )
            await asyncio.sleep(0.12)

        await asyncio.sleep(0.5)

        # Agent execution
        files_generated = []

        # Jordan (frontend)
        await emit(2, "Jordan", "Module 2: Agent Jordan writing frontend/index.html...", 45)
        await asyncio.sleep(1.2 + random.uniform(0.3, 0.8))
        files_generated.append("frontend/index.html")
        await emit(2, "Jordan", "  -> Wrote: frontend/index.html", 50, files=["frontend/index.html"])
        await asyncio.sleep(0.3)

        await emit(2, "Jordan", "Module 2: Agent Jordan writing frontend/styles.css...", 52)
        await asyncio.sleep(0.8 + random.uniform(0.2, 0.5))
        files_generated.append("frontend/styles.css")
        await emit(2, "Jordan", "  -> Wrote: frontend/styles.css", 55, files=["frontend/styles.css"])
        await asyncio.sleep(0.2)

        await emit(2, "Jordan", "Module 2: Agent Jordan writing frontend/app.js...", 57)
        await asyncio.sleep(0.7 + random.uniform(0.2, 0.4))
        files_generated.append("frontend/app.js")
        await emit(2, "Jordan", "  -> Wrote: frontend/app.js", 60, files=["frontend/app.js"])
        await asyncio.sleep(0.3)

        # Alex (backend)
        await emit(2, "Alex", "Module 2: Agent Alex writing backend/app.py...", 63)
        await asyncio.sleep(1.0 + random.uniform(0.3, 0.7))
        files_generated.append("backend/app.py")
        await emit(2, "Alex", "  -> Wrote: backend/app.py", 67, files=["backend/app.py"])
        await asyncio.sleep(0.2)

        await emit(2, "Alex", "Module 2: Agent Alex writing backend/requirements.txt...", 69)
        await asyncio.sleep(0.3 + random.uniform(0.1, 0.2))
        files_generated.append("backend/requirements.txt")
        await emit(2, "Alex", "  -> Wrote: backend/requirements.txt", 72, files=["backend/requirements.txt"])
        await asyncio.sleep(0.3)

        # Sam (devops)
        await emit(2, "Sam", "Module 2: Agent Sam writing firebase.json...", 75)
        await asyncio.sleep(0.5 + random.uniform(0.1, 0.3))
        files_generated.append("firebase.json")
        await emit(2, "Sam", "  -> Wrote: firebase.json", 77, files=["firebase.json"])
        await asyncio.sleep(0.2)

        await emit(2, "Sam", "Module 2: Agent Sam writing README.md...", 79)
        await asyncio.sleep(0.4 + random.uniform(0.1, 0.2))
        files_generated.append("README.md")
        await emit(2, "Sam", "  -> Wrote: README.md", 80, files=["README.md"])
        await asyncio.sleep(0.3)

        # Boss final review
        await emit(2, "Boss", "Module 2: Boss reviewing all generated files...", 82)
        await asyncio.sleep(0.8 + random.uniform(0.2, 0.5))
        await emit(2, "Boss", "Module 2: Code review passed — all files valid", 85)
        await asyncio.sleep(0.3)

        # ── MODULE 3: Deployment ──
        total_files = len(files_generated)
        await emit(3, "Sam", f"Module 3: Packaging {total_files} files...", 87)
        await asyncio.sleep(0.5 + random.uniform(0.2, 0.4))
        await emit(3, "Sam", f"Module 3: Packaged {total_files} files -> public/", 89)
        await asyncio.sleep(0.3)

        await emit(3, "Sam", "Module 3: Firebase CLI found: 13.6.0", 90)
        await asyncio.sleep(0.3)

        await emit(3, "Sam", "Module 3: Initializing Firebase hosting config...", 91)
        await asyncio.sleep(0.4)

        # Deploy progress simulation
        for pct in [20, 40, 60, 80, 95]:
            await emit(3, "Sam", f"Module 3: Deploying to Firebase... {pct}%", 91 + (pct // 20))
            await asyncio.sleep(0.4 + random.uniform(0.1, 0.3))

        await emit(3, "Sam", "Module 3: Deploy successful!", 97)
        await asyncio.sleep(0.3)

        # Generate a fake URL based on the prompt
        slug = text_prompt.lower().split()[0:3]
        slug = "-".join(w for w in slug if w.isalnum())[:20] or "autologic-app"
        hosting_url = f"https://{slug}-demo.web.app"

        total_time = round(random.uniform(14.0, 22.0), 1)
        await emit(
            3, "system",
            f"Pipeline complete! ({total_time}s) URL: {hosting_url}",
            100,
            status="complete",
            files=files_generated,
        )

    except asyncio.CancelledError:
        await emit(0, "system", "Pipeline cancelled", 0, status="error")
        raise
    except Exception as e:
        logger.exception(f"Simulation error for run {run_id}")
        await emit(0, "system", f"Pipeline error: {str(e)}", 0, status="error")


# ──────────────────────────────────────────────
# Real pipeline runner (when GEMINI_API_KEY is set)
# ──────────────────────────────────────────────

async def _run_real_pipeline(
    run_id: str,
    text_prompt: str,
    image_path: Optional[str],
    audio_path: Optional[str],
):
    """
    Run the actual 3-module pipeline using the real modules.
    Wraps synchronous module calls in asyncio.to_thread() and
    pushes progress via WebSocket.
    """

    # ── New engine integration (v0.2) ──
    # When running in live mode, the pipeline now uses the modular engine:
    #
    #   from autologic.engine import AutoLogicEngine
    #   from autologic.config import AutoLogicConfig
    #   from autologic.models import PipelineContext
    #
    #   config = AutoLogicConfig.from_file("config.yaml")
    #   engine = AutoLogicEngine.default_pipeline(config)
    #   context = PipelineContext()
    #   await context.set("text_prompt", text_prompt)
    #   result = await engine.run(context)

    async def emit(module, agent, message, progress, **kwargs):
        msg = _progress_msg(module, agent, message, progress, run_id, **kwargs)
        await manager.broadcast(run_id, msg)
        logger.info(f"[run={run_id[:8]}] M{module} {agent}: {message}")

    try:
        await emit(1, "system", "Pipeline started (live mode)", 0, status="running")

        # ── MODULE 1 ──
        await emit(1, "system", "Module 1: Processing multimodal inputs...", 5)

        from multi_ingest import ingest_requirements

        context = await asyncio.to_thread(
            ingest_requirements,
            image_path=image_path,
            audio_path=audio_path,
            text_prompt=text_prompt,
        )

        ctx_len = len(context.get("consolidated_context", ""))
        await emit(1, "system", f"Module 1: Context consolidated ({ctx_len} chars)", 20)

        has_text = bool(context.get("raw_text_prompt"))
        has_img = bool(context.get("image_path"))
        has_audio = bool(context.get("transcription"))
        await emit(
            1, "system",
            f"Module 1: Complete — Text: {'Yes' if has_text else 'No'}, "
            f"Image: {'Yes' if has_img else 'No'}, Audio: {'Yes' if has_audio else 'No'}",
            22,
        )

        # ── MODULE 2: Planning ──
        await emit(2, "Boss", "Module 2: Sending planning request to Gemini...", 25)

        from core_gen import execute_all_tasks, orchestrate_planning

        api_key = GEMINI_API_KEY

        tasks = await asyncio.to_thread(
            orchestrate_planning,
            multimodal_context=context,
            api_key=api_key,
            image_path=context.get("image_path"),
        )

        await emit(2, "Boss", f"Module 2: Plan generated — {len(tasks)} tasks", 35)

        for i, t in enumerate(tasks):
            deps = ", ".join(t.get("depends_on", [])) or "none"
            await emit(
                2, "Boss",
                f"  [{i+1}] {t['agent']:8s} | {t['task']} (deps: {deps})",
                36 + i,
            )

        # ── MODULE 2: Code generation ──
        await emit(2, "Boss", "Module 2: Agents executing tasks...", 45)

        results = await asyncio.to_thread(execute_all_tasks, tasks, api_key)

        files_generated = []
        progress_base = 50
        for i, r in enumerate(results):
            agent = r.get("agent", "?")
            files = r.get("files", [])
            files_generated.extend(files)
            pct = progress_base + int((i + 1) / len(results) * 30)
            for f in files:
                await emit(2, agent, f"  -> Wrote: {f}", pct, files=[f])

        await emit(2, "Boss", f"Module 2: All {len(results)} tasks executed", 82)

        # ── MODULE 3: Deployment ──
        workspace_dir = str(WORKSPACE_DIR)
        project_id = os.getenv("FIREBASE_PROJECT_ID", "")

        if not project_id:
            await emit(3, "Sam", "Module 3: No Firebase project ID — skipping deployment", 90)
            await emit(
                3, "system",
                f"Pipeline complete! Files generated in {workspace_dir}",
                100,
                status="complete",
                files=files_generated,
            )
            return

        await emit(3, "Sam", f"Module 3: Packaging {len(files_generated)} files...", 85)

        from auto_deploy import deploy_to_firebase

        deploy_result = await asyncio.to_thread(
            deploy_to_firebase, workspace_dir, project_id
        )

        if deploy_result["deployment_status"] == "success":
            url = deploy_result.get("hosting_url", "")
            await emit(3, "Sam", "Module 3: Deploy successful!", 97)
            await emit(
                3, "system",
                f"Pipeline complete! URL: {url}",
                100,
                status="complete",
                files=files_generated,
            )
        else:
            logs = deploy_result.get("logs", "Unknown error")
            await emit(3, "Sam", f"Module 3: Deploy failed — {logs}", 95, status="error")

    except asyncio.CancelledError:
        await emit(0, "system", "Pipeline cancelled", 0, status="error")
        raise
    except Exception as e:
        logger.exception(f"Pipeline error for run {run_id}")
        await emit(0, "system", f"Pipeline error: {str(e)}", 0, status="error")


# ──────────────────────────────────────────────
# Active run tracker
# ──────────────────────────────────────────────

_active_runs: dict[str, asyncio.Task] = {}


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    mode = "DEMO (simulation)" if IS_DEMO_MODE else "LIVE (Gemini API)"
    logger.info(f"AutoLogic server starting — mode: {mode}")
    logger.info(f"Serving index.html from: {BASE_DIR / 'index.html'}")
    yield
    # Cancel any running pipelines on shutdown
    for run_id, task in _active_runs.items():
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled run: {run_id}")
    _active_runs.clear()
    logger.info("AutoLogic server shut down")


app = FastAPI(
    title="AutoLogic WebUI Server",
    version="0.1.0",
    description="FastAPI server for AutoLogic — Multimodal AI Development Pipeline",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Serve the main index.html page."""
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "index.html not found"},
        )
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mode": "demo" if IS_DEMO_MODE else "live",
        "active_runs": len(_active_runs),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pipeline progress.

    Clients can optionally send {"run_id": "..."} after connecting
    to subscribe to a specific run. Otherwise they receive all broadcasts.
    """
    run_id = "global"
    await manager.connect(websocket, run_id)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                # Allow client to subscribe to a specific run
                if "run_id" in msg:
                    old_run_id = run_id
                    run_id = msg["run_id"]
                    manager.disconnect(websocket, old_run_id)
                    await manager.connect(websocket, run_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "run_id": run_id,
                    }))
                # Allow client to send a ping
                elif msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass  # Ignore non-JSON messages
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)
    except Exception:
        manager.disconnect(websocket, run_id)


@app.post("/api/run")
async def run_pipeline(
    text_prompt: str = Form(""),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    """
    Start the AutoLogic pipeline.

    Accepts:
        - text_prompt (string): Text description of what to build.
        - image (file, optional): Sketch/wireframe image.
        - audio (file, optional): Voice note audio file.

    Returns:
        - run_id: Unique ID for this pipeline run.
        - mode: 'demo' or 'live'.
        - message: Status message.

    Subscribe to WebSocket /ws and send {"run_id": "<run_id>"} to receive
    real-time progress updates.
    """
    # Validate: at least text is required
    if not text_prompt.strip() and image is None and audio is None:
        return JSONResponse(
            status_code=400,
            content={"error": "At least one input is required (text_prompt, image, or audio)."},
        )

    run_id = str(uuid.uuid4())
    logger.info(f"New pipeline run: {run_id} (text={len(text_prompt)} chars, image={image is not None}, audio={audio is not None})")

    # Save uploaded files
    image_path = None
    audio_path = None
    run_upload_dir = UPLOAD_DIR / run_id
    run_upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        if image is not None and image.filename:
            safe_name = Path(image.filename).name
            image_path = str(run_upload_dir / safe_name)
            content = await image.read()
            if len(content) > 0:
                Path(image_path).write_bytes(content)
                logger.info(f"Saved image: {image_path} ({len(content)} bytes)")
            else:
                image_path = None

        if audio is not None and audio.filename:
            safe_name = Path(audio.filename).name
            audio_path = str(run_upload_dir / safe_name)
            content = await audio.read()
            if len(content) > 0:
                Path(audio_path).write_bytes(content)
                logger.info(f"Saved audio: {audio_path} ({len(content)} bytes)")
            else:
                audio_path = None
    except Exception as e:
        logger.error(f"Failed to save uploaded files: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save uploaded files: {str(e)}"},
        )

    # Launch pipeline in background
    if IS_DEMO_MODE:
        task = asyncio.create_task(
            _run_simulation(
                run_id=run_id,
                text_prompt=text_prompt.strip() or "Build a web application",
                has_image=image_path is not None,
                has_audio=audio_path is not None,
            )
        )
    else:
        task = asyncio.create_task(
            _run_real_pipeline(
                run_id=run_id,
                text_prompt=text_prompt.strip(),
                image_path=image_path,
                audio_path=audio_path,
            )
        )

    _active_runs[run_id] = task

    # Clean up when done
    def _on_done(t: asyncio.Task):
        _active_runs.pop(run_id, None)
        if t.exception():
            logger.error(f"Run {run_id} failed: {t.exception()}")
        else:
            logger.info(f"Run {run_id} completed")

    task.add_done_callback(_on_done)

    return {
        "run_id": run_id,
        "mode": "demo" if IS_DEMO_MODE else "live",
        "message": "Pipeline started. Connect to WebSocket /ws and subscribe with this run_id.",
    }


@app.get("/api/runs")
async def list_runs():
    """List currently active pipeline runs."""
    return {
        "active_runs": [
            {"run_id": rid, "done": task.done()}
            for rid, task in _active_runs.items()
        ],
        "count": len(_active_runs),
    }


@app.post("/api/cancel/{run_id}")
async def cancel_run(run_id: str):
    """Cancel an active pipeline run."""
    task = _active_runs.get(run_id)
    if task is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found or already completed."},
        )
    if task.done():
        return {"run_id": run_id, "status": "already_completed"}

    task.cancel()
    return {"run_id": run_id, "status": "cancelled"}


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting AutoLogic server on {host}:{port}")
    logger.info(f"Mode: {'DEMO (simulation)' if IS_DEMO_MODE else 'LIVE (Gemini API)'}")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
