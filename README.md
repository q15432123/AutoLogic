
<div align="center">

```
                   ___         __       __                _
                  / _ | __ __ / /_ ___ / /  ___  ___ _ (_)___
                 / __ |/ // // __// _ \/ /__/ _ \/ _ `// / __/
                /_/ |_|\_,_/ \__/ \___/____/\___/\_, //_/\__/
                                                /___/

     █████╗ ██╗   ██╗████████╗ ██████╗ ██╗      ██████╗  ██████╗ ██╗ ██████╗
    ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██║     ██╔═══██╗██╔════╝ ██║██╔════╝
    ███████║██║   ██║   ██║   ██║   ██║██║     ██║   ██║██║  ███╗██║██║
    ██╔══██║██║   ██║   ██║   ██║   ██║██║     ██║   ██║██║   ██║██║██║
    ██║  ██║╚██████╔╝   ██║   ╚██████╔╝███████╗╚██████╔╝╚██████╔╝██║╚██████╗
    ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝ ╚═════╝
```

### *The first multimodal pipeline that turns your whiteboards into Firebase-ready apps.*

---

**Designed by Google Gemini** | **Written by Anthropic Claude Opus**

---

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22d3ee?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Pro-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Claude](https://img.shields.io/badge/Claude_Opus-Anthropic-cc785c?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMiA3bDEwIDUgMTAtNS0xMC01ek0yIDE3bDEwIDUgMTAtNU0yIDEybDEwIDUgMTAtNSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSJub25lIi8+PC9zdmc+)

</div>

---

## The Pipeline

```
                           A U T O L O G I C    P I P E L I N E
  ╔══════════════════════════════════════════════════════════════════════════════════╗
  ║                                                                                ║
  ║   Sketch / Audio / Text                                                        ║
  ║          |                                                                     ║
  ║          v                                                                     ║
  ║   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐        ║
  ║   │                 │      │                 │      │                 │        ║
  ║   │   MODULE 1      │      │   MODULE 2      │      │   MODULE 3      │        ║
  ║   │   Ingest        │ ───> │   Gemini Brain  │ ───> │   Deploy        │        ║
  ║   │                 │      │                 │      │                 │        ║
  ║   │  multi_ingest   │      │   core_gen      │      │  auto_deploy    │        ║
  ║   │                 │      │                 │      │                 │        ║
  ║   └─────────────────┘      └─────────────────┘      └─────────────────┘        ║
  ║    OpenCV + Whisper          Gemini 1.5 Pro           Firebase Hosting          ║
  ║    Image validation          Multi-Agent Plan         Package + Deploy          ║
  ║    Audio transcription       Code Generation          Live URL output           ║
  ║                                        |                                       ║
  ║                                        v                                       ║
  ║                                  ╔═══════════╗                                 ║
  ║                                  ║ LIVE APP  ║                                 ║
  ║                                  ╚═══════════╝                                 ║
  ║                                                                                ║
  ╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Features

| | Feature | Description |
|---|---|---|
| :art: | **Multimodal Input** | Feed it a napkin sketch, a voice memo, or raw text. Hand-drawn wireframes via OpenCV, voice notes via Whisper, plain English descriptions -- all at once or any combination. |
| :brain: | **Gemini 1.5 Pro Planning** | The Gemini Brain analyzes your multimodal context, understands layout from sketches, extracts intent from transcriptions, and produces a structured JSON execution plan. |
| :busts_in_silhouette: | **Multi-Agent Code Generation** | Four specialized AI agents -- **Boss** (orchestrator), **Jordan** (frontend), **Alex** (backend), **Sam** (DevOps) -- execute tasks in dependency order, writing production-quality code. |
| :rocket: | **One-Click Firebase Deploy** | Generated code is auto-packaged into a `public/` directory, Firebase configs are written programmatically, and `firebase deploy` runs non-interactively. You get a live URL. |
| :tv: | **Real-Time WebUI Monitoring** | The bundled `index.html` provides a cyberpunk-styled dashboard with an interactive terminal simulator showing the full pipeline execution flow. |
| :wrench: | **Pre-Processing Layer** | Image validation and enhancement via OpenCV, audio denoising and transcription via OpenAI Whisper, base64 encoding for Gemini's multimodal API -- all handled before the brain even fires. |
| :jigsaw: | **Agent Marketplace** | Drop-in `_team/` templates let you define custom agent personas, swap specialists, or add new roles. The architecture is built for extensibility. |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/q15432123/AutoLogic.git
cd AutoLogic
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
GEMINI_API_KEY=your-gemini-api-key-here
FIREBASE_PROJECT_ID=your-firebase-project-id
```

> Get a Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 3. Run

```bash
# Interactive mode (prompts you for input)
python run_main.py

# Direct text input
python run_main.py --text "Build a todo app with dark mode"

# Full multimodal
python run_main.py --image sketch.png --audio notes.mp3 --text "portfolio site"

# Generate code only, skip deployment
python run_main.py --text "landing page" --skip-deploy
```

---

## Agent DNA -- The `_team/` System

AutoLogic's multi-agent architecture is driven by **Agent DNA templates** -- structured role definitions that tell each agent *who they are*, *what they own*, and *how they write code*.

```
_team/
  boss.yaml        # Orchestrator: decomposes goals, assigns tasks, reviews output
  jordan.yaml      # Frontend: HTML5, CSS3, JS -- pixel-perfect UI from sketches
  alex.yaml        # Backend: Flask, Express, APIs -- server logic & data models
  sam.yaml         # DevOps: Firebase, configs, README -- deployment pipeline
  custom/          # YOUR agents go here
    designer.yaml  # Example: a Figma-to-code specialist
    qa.yaml        # Example: automated testing agent
```

Each template defines:

| Field | Purpose |
|---|---|
| `role` | Agent's identity and expertise area |
| `owns` | File paths and directories the agent is responsible for |
| `style` | Coding conventions, frameworks, and patterns to follow |
| `depends_on` | Which agents must complete before this one runs |
| `tools` | Available actions: `write_file`, `run_command`, etc. |

> **Want a new specialist?** Drop a YAML file in `_team/custom/` and the planning engine picks it up automatically. Swap Jordan for a React specialist. Add a QA agent that writes tests. The roster is yours to define.

---

## Architecture

AutoLogic is a **three-module sequential pipeline**. Each module is a standalone Python file that can be tested independently.

### Module 1: `multi_ingest.py` -- Multimodal Ingestion

```
Input Sources                    Processing                     Output
─────────────────────────────────────────────────────────────────────────
 [Image]  .png/.jpg  ──>  OpenCV validate + base64 encode  ──┐
 [Audio]  .mp3/.wav  ──>  Whisper transcribe to text       ──┤──> Context Dict
 [Text]   string     ──>  Pass through                     ──┘
```

- Validates image dimensions and readability via OpenCV
- Transcribes audio using OpenAI Whisper (`base` model, ~140MB)
- Encodes images to base64 for Gemini's multimodal input
- Returns a unified context dictionary with all processed inputs

### Module 2: `core_gen.py` -- Gemini Brain + Agent Execution

```
Context Dict ──> Gemini 1.5 Pro ──> JSON Task Plan ──> Agent Loop ──> Files on Disk
                     |                                     |
                     |  "Analyze sketch + text,            |  Boss -> Jordan -> Alex -> Sam
                     |   decompose into agent tasks"       |  (dependency-ordered execution)
                     |                                     |
                     v                                     v
              Structured Plan                     _workspaces/
              with dependencies                     frontend/
                                                    backend/
                                                    firebase.json
```

- Sends multimodal context (text + optional image) to Gemini 1.5 Pro
- Gemini returns a JSON task plan with agent assignments and dependencies
- Each agent receives its task as a prompt and generates complete file contents
- All files are written to `_workspaces/` with proper directory structure

### Module 3: `auto_deploy.py` -- Package & Deploy

```
_workspaces/ ──> Package to public/ ──> firebase.json + .firebaserc ──> firebase deploy ──> LIVE URL
```

- Scans workspace for deployable web assets (HTML, CSS, JS, images)
- Copies everything into a `public/` directory for Firebase Hosting
- Programmatically generates `firebase.json` and `.firebaserc` (no interactive `firebase init`)
- Runs `firebase deploy --non-interactive` and extracts the hosting URL
- Generates a deployment report in `DEPLOY_REPORT.md`

---

## Why AutoLogic?

| | Traditional Workflow | CodeSnap / Copilot | **AutoLogic** |
|---|---|---|---|
| Input | Text prompts only | Screenshot of code | **Sketch + Voice + Text** (true multimodal) |
| Planning | You plan everything | Suggests next line | **Gemini decomposes the entire project** |
| Execution | You write every file | Autocomplete fragments | **4 agents write all files in parallel** |
| Output | Code on your machine | Code on your machine | **Live deployed app with URL** |
| Deployment | Manual Firebase setup | Not included | **Fully automated, zero-config** |
| Time | Hours to days | Faster typing | **Sketch to live site in under 60 seconds** |

<br>

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   CodeSnap helps you show code.                                        ║
║                                                                        ║
║   AutoLogic helps you SHUT UP AND BUILD.                               ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## Project Structure

```
AutoLogic/
  ├── run_main.py          # Entry point -- chains all 3 modules
  ├── multi_ingest.py      # Module 1: Multimodal ingestion engine
  ├── core_gen.py          # Module 2: Gemini planning + agent code gen
  ├── auto_deploy.py       # Module 3: Firebase packaging & deployment
  ├── index.html           # Cyberpunk WebUI dashboard
  ├── requirements.txt     # Python dependencies
  ├── .env.example         # Environment variable template
  ├── .gitignore
  └── _workspaces/         # Generated output (gitignored)
```

---

## Requirements

```
opencv-python >= 4.8.0
openai-whisper >= 20231117
google-generativeai >= 0.8.0
python-dotenv >= 1.0.0
Pillow >= 10.0.0
```

Plus: [Firebase CLI](https://firebase.google.com/docs/cli) (`npm install -g firebase-tools`) for deployment.

---

## License

```
MIT License

Copyright (c) 2025 AutoLogic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

```
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   Architecture & Conceptual Design ... Google Gemini        │
  │   Implementation & Documentation ..... Anthropic Claude     │
  │                                                             │
  │   Two AIs. One pipeline. Zero excuses.                      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

**Built with the combined intelligence of [Google Gemini](https://deepmind.google/technologies/gemini/) and [Anthropic Claude](https://www.anthropic.com/claude)**

*From whiteboard to production. No humans were mass-deployed in the making of this pipeline.*

</div>
