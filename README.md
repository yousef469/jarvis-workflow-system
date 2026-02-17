# ◈ ATLAS (Formerly J.A.R.V.I.S.)
### Offline AI Assistant for Engineers, Researchers & Creators

Atlas is a sophisticated, offline-first AI ecosystem designed to provide high-performance assistance without relying on external cloud services. It integrates real-time voice interaction, computer vision, 3D modeling, and automation into a unified cognitive architecture.

---

## 🏛️ Core Architecture

Atlas is built on a distributed engine model that prioritizes low latency and local execution:

- **Frontend**: A sleek, high-fidelity React application (TypeScript, Three.js) wrapped in Electron for native desktop capabilities.
- **Backend**: A FastAPI-powered Python hub orchestrating state-of-the-art AI models and worker threads.
- **Communication**: Real-time bidirectional streaming via WebSocket (Client ↔ Server).

---

## 🔄 The V24.5 Cognitive Loop

The heart of Atlas is the **6-Step Cognitive Pipeline**, which ensures every action is grounded in context and memory:

1.  **Step 0: Memory Pre-fetch**: Loads relevant facts, preferences, and past lessons from ChromaDB before planning.
2.  **Step 1: Planning**: A memory-informed routing step where the Brain selects the appropriate specialized worker.
3.  **Step 2: Execution**: The selected specialized worker performs the task (Web, Vision, Automation).
4.  **Step 3: Review**: The Brain critiques the worker's output for quality and accuracy.
5.  **Step 4: Update Memory**: Lessons learned and mistakes made are committed to long-term memory.
6.  **Step 5: Synthesis**: A final, emotive response is generated for the user.

---

## 🛠️ Specialized Engines

### 🧠 The Brain (Reasoning & Memory)
Powered by **Qwen 2.5 Coder** (via Ollama). Atlas features **Infinite Long-Term Memory** via ChromaDB—it remembers every fact, preference, and lesson learned across sessions.

### 🎙️ Speech (Voice & Audio)
- **STT**: `whisper.cpp` (CoreML optimized) or `Vosk` for near-instant transcription.
- **TTS**: `Piper` and `Coqui V3` for natural, emotive vocal feedback.
- **Wake Word**: `openWakeWord` for local, low-power detection ("Hey Jarvis").

### 👁️ Vision (OmniParser)
Utilizes a custom Vision-Language Model pipeline to understand the user's screen, identify UI elements, and perform OCR with spatial awareness.

### 🖼️ Image Generation (ComfyUI & SDXL)
Integrates with **ComfyUI** and **SDXL** to generate high-fidelity images locally, handling complex prompt engineering.

### 🏗️ 3D Generation
A custom procedural modeling engine that converts natural language descriptions into valid 3D hierarchies and GLTF models.

---

## 📁 Project Structure

```text
jarvis/
├── src/                # React Frontend (TypeScript, Three.js, Zustand)
├── backend/            # Python FastAPI Server & Cognitive Logic
│   ├── jarvis_brain_v245.py    # Cognitive Pipeline
│   ├── jarvis_dispatcher_v245.py    # Worker Orchestrator
│   ├── server.py              # Main Entry Point
│   └── workers/               # Specialized Task Workers
├── electron/           # Desktop App configuration
└── public/             # Static assets
```

---

## 🚀 Quick Start (Installation)

To get Atlas running on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yousef469/jarvis-workflow-system.git jarvis
cd jarvis
```

### 2. Set Up Environment
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
# Install backend requirements
pip install -r backend/requirements.txt

# Install frontend dependencies
npm install
```

### 4. Download AI Models & Assets
Atlas requires several large models (Vosk, OmniParser weights, etc.) to be present in the `backend/` directory. We've provided a script to automate this:
```bash
python3 download_assets.py
```

### 5. Install Ollama
Ensure [Ollama](https://ollama.com/) is installed and running, then pull the required model:
```bash
ollama pull qwen2.5-coder:3b
```

### 6. Launch the System
```bash
npm run start-all
```
This command starts both the Electron frontend and the Python backend concurrently.

---

## 📊 Technical Performance (v24.5)
| Metric | Performance |
|--------|-------------|
| **Planning Latency** | ~350ms |
| **Memory Access** | <10ms |
| **RAM Footprint** | ~800MB (Excl. Models) |
| **STT Accuracy** | ~98% |

---

*“Precision Engineering. Infinite Intelligence.”*
