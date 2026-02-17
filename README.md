# ◈ J.A.R.V.I.S. (Just A Rather Very Intelligent System)
### Offline-First AI Ecosystem for Engineers, Researchers & Creators

**JARVIS** is a high-performance, offline-first AI ecosystem designed to provide seamless assistance across voice interaction, computer vision, 3D modeling, and local automation. Built on a sophisticated **V24.5 Cognitive Pipeline**, it ensures every action is grounded in context, memory, and continuous learning—all while remaining 100% local for maximum privacy and low latency.

---

## 🏛️ Core Architecture

JARVIS is engineered as a distributed system of specialized engines working in orchestration:

- **Frontend**: A high-fidelity React application utilizing **Three.js** for 3D visualization and **Zustand** for ultra-fast state management, wrapped in **Electron** for native desktop integrations.
- **Backend Hub**: A **FastAPI** hub that orchestrates worker threads, streams real-time data via **WebSockets**, and manages the cognitive pipeline.
- **Cognitive Brain**: Powered by **Qwen 2.5 Coder** (via Ollama), providing reasoning, planning, and synthesis capabilities.
- **Knowledge Engine**: **ChromaDB**-based vector memory for infinite long-term recall of facts, preferences, and past experiences.

---

## 🔄 The V24.5 Cognitive Pipeline
The "Master Loop" consists of 6 distinct stages that ensure JARVIS behaves intelligently rather than just reactively:

1.  **Step 0: Memory Pre-fetch**: Before planning, the Brain queries the vector database for relevant facts, past lessons, and user preferences based on the input.
2.  **Step 1: Planning (Memory-Informed)**: The Brain analyzes the request + memory context and selects the appropriate specialized worker (Web, Vision, etc.).
3.  **Step 2: Execution**: The selected worker performs the task (opening an app, searching the web, or generating a 3D model).
4.  **Step 3: Review**: After execution, the Brain critiques the outcome for quality, accuracy, and potential errors.
5.  **Step 4: Update Memory**: Lessons learned and mistakes made are committed back to **ChromaDB** to improve future performance.
6.  **Step 5: Synthesis**: A final, emotive, and professional response is generated and delivered via TTS.

---

## 🛠️ Specialized Engine Workers

### 👁️ Vision Engine (OmniParser)
Uses a custom Vision-Language Model pipeline to capture and understand your screen state. It identifies buttons, icons, and UI hierarchies, enabling JARVIS to "see" what you're working on and provide contextual help.

### 🏗️ 3D Generation Engine
A custom procedural engine that translates natural language descriptions into complex 3D hierarchies. It supports:
- **Geometry Snapping**: Forces 1mm/5mm alignment for clean engineering models.
- **Symmetry Guards**: Automatically mirrors left/right components (e.g., robotic legs, wings).
- **Physics Bias**: Intelligently places heavy components (engines) at the bottom and rotors at the top.

### 🎙️ Speech (Voice & Audio)
- **STT**: `whisper.cpp` (CoreML optimized for Mac) or `Vosk` for near-instant low-latency transcription.
- **TTS**: `Piper` (Ryan High voice) and `Coqui V3` for natural, British-accented professional feedback.
- **Wake Word**: `openWakeWord` for local, low-power detection ("Hey Jarvis").

### 🖼️ Image Generation
Deep integration with **ComfyUI** and **SDXL** for generating high-fidelity assets directly from the chat interface.

### 🛰️ Automation
Full system-level control via **PyAutoGUI** and OS-native scripts to launch apps, type text, and manipulate system settings without an internet connection.

---

## 🚀 Quick Start (Installation)

### 1. Clone & Setup
```bash
git clone https://github.com/yousef469/jarvis-workflow-system.git jarvis
cd jarvis
```

### 2. Environment & Dependencies
```bash
# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Setup Frontend
npm install
```

### 3. Automated Asset Setup
JARVIS requires large model files (Vosk, OmniParser, Piper) that are too heavy for Git. Run the following to set them up automatically:
```bash
python3 download_assets.py
```

### 4. LLM Requirement
Install [Ollama](https://ollama.com/) and pull the core reasoning model:
```bash
ollama pull qwen2.5-coder:3b
```

### 5. Launch
```bash
npm run start-all
```

---

## 📈 Technical Specifications (Benchmarks)

| Metric | Performance (v24.5) |
|--------|-----------------------|
| **Planning Latency** | ~350ms |
| **Memory Access** | <10ms |
| **RAM Footprint** | ~800MB (Base System) |
| **STT Accuracy** | ~98.4% |
| **Synthesis Speed** | ~45 tokens/sec |

---

## 📁 System Inventory

```text
jarvis/
├── src/                # React Frontend (TypeScript, Three.js)
├── backend/            # Python FastAPI Server & Cognitive Logic
│   ├── jarvis_brain_v245.py    # Main Cognitive Pipeline
│   ├── jarvis_dispatcher_v245.py    # Worker Routing
│   ├── server.py              # Socket.io Hub & Entry Point
│   └── workers/               # Specialized Task Specialists
└── download_assets.py  # Automated Model Downloader
```

*“Precision Engineering. Infinite Intelligence.”*
*JARVIS v24.5 - Definitive Archive*
