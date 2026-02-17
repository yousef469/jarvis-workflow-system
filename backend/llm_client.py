"""Local LLM client - 100% Ollama for both Text and Vision."""
import os
import json
import asyncio
from typing import Optional
import ollama

LLAVA_MODEL = "qwen2-vl"        # Best local vision model
TEXT_MODEL = "qwen2.5-coder:3b" # Best local coding model (Aligned with 3B)

SYSTEM_PROMPT = """You are "AI Hub Assistant" — a helpful AI coding assistant in a web-based code editor.

CRITICAL: Output ONLY a valid JSON array. No markdown, no explanations, no code blocks.

AVAILABLE ACTIONS:
1. say - Chat/explain: {"action":"say","text":"response here"}
2. suggest_fix - Modify/enhance EXISTING code: {"action":"suggest_fix","fixed":"COMPLETE improved code","explanation":"what changed"}
3. create_file - Create NEW file: {"action":"create_file","filename":"name.ext","content":"file content"}
4. run_code - Execute current code: {"action":"run_code"}

DECISION RULES:
- Questions/explanations only → "say"
- Fix/enhance/improve/modify/add to CURRENT code → "suggest_fix" (MUST include full improved code)
- Create/make/build NEW files → "create_file"
- Run/execute → "run_code"

IMPORTANT - MODIFYING CODE:
When user says "enhance", "improve", "add", "make it better", "yes do it", "add more":
- You MUST use "suggest_fix" action with the COMPLETE improved code
- NEVER just explain what could be done - actually DO IT
- The "fixed" field must contain the ENTIRE file with improvements applied
- Don't ask for confirmation, just make the improvements

IMPORTANT - FILE CREATION:
When user says "make", "create", "build" a website/app/project:
- You MUST use "create_file" action for EACH file
- NEVER just output HTML/code as text in "say"
- Include complete working code in "content" field

EXAMPLE - "enhance this code" or "make it better" or "yes add that":
[{"action":"suggest_fix","fixed":"/* Enhanced CSS */\\n* { margin: 0; padding: 0; box-sizing: border-box; }\\nbody { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 40px; }\\nh1 { color: #fff; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }\\np { color: #f0f0f0; line-height: 1.6; }\\n.container { max-width: 1200px; margin: 0 auto; }\\n.card { background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }","explanation":"Added gradient background, modern typography, cards, shadows, and responsive container"}]

EXAMPLE - "make a basic website":
[{"action":"create_file","filename":"index.html","content":"<!DOCTYPE html>\\n<html>\\n<head>\\n<title>My Site</title>\\n<link rel=\\"stylesheet\\" href=\\"style.css\\">\\n</head>\\n<body>\\n<h1>Welcome</h1>\\n<script src=\\"script.js\\"></script>\\n</body>\\n</html>"},{"action":"create_file","filename":"style.css","content":"* { margin: 0; padding: 0; }\\nbody { font-family: Arial; padding: 20px; }"},{"action":"say","text":"Created your website files!"}]

EXAMPLE - "what is a variable":
[{"action":"say","text":"A variable is a named container for storing data values."}]

Output ONLY the JSON array, nothing else."""


class VisionClient:
    """Unified client: 100% Local (via Ollama)."""
    
    def __init__(self):
        self.vision_model = LLAVA_MODEL
        self.text_model = TEXT_MODEL
        print(f"[LLM] Unified Local Brain ready (Text: {self.text_model}, Vision: {self.vision_model})")
    
    def _is_complex_task(self, transcript: str) -> bool:
        """Heuristic to determine intent."""
        transcript_lower = transcript.lower()
        complex_keywords = [
            'explain', 'analyze', 'create', 'build', 'make', 'design',
            'write code', 'fix code', 'debug', 'refactor', 'optimize',
            'suggest', 'improve', 'enhance', 'help me with',
            'how do i', 'how to', 'what is', 'why', 'compare'
        ]
        for keyword in complex_keywords:
            if keyword in transcript_lower:
                return True
        return len(transcript.split()) > 10
    
    async def get_actions(self, transcript: str, context: dict, image_b64: Optional[str] = None) -> list:
        """Route to Local LLaVA for vision or Local Phi-3 for text."""
        if image_b64:
            print(f"[LLM] Using Local LLaVA (Vision)")
            return await self._call_ollama_vision(transcript, context, image_b64)
        
        print(f"[LLM] Using Local {self.text_model} (Text)")
        return await self._call_ollama_text(transcript, context)

    async def _call_ollama_text(self, transcript: str, context: dict) -> list:
        """Call local Ollama for text tasks."""
        try:
            prompt = self._build_prompt(transcript, context)
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.text_model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.3}
            )
            return self._parse_response(response['message']['content'])
        except Exception as e:
            print(f"Ollama text error: {e}")
            return [{"action": "error", "reason": "ollama_error", "message": str(e)}]

    async def _call_ollama_vision(self, transcript: str, context: dict, image_b64: str) -> list:
        """Call local LLaVA via Ollama for vision analysis."""
        try:
            prompt = self._build_prompt(transcript, context)
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.vision_model,
                prompt=f"{SYSTEM_PROMPT}\n\n{prompt}",
                images=[image_b64]
            )
            return self._parse_response(response['response'])
        except Exception as e:
            print(f"LLaVA error: {e}")
            return [{"action": "error", "reason": "llava_error", "message": str(e)}]

    def _build_prompt(self, transcript: str, context: dict) -> str:
        """Build prompt with context."""
        ctx = json.dumps(context, indent=2) if context else "{}"
        return f"Context:\n{ctx}\n\nUser request:\n\"{transcript}\"\n\nReturn ONLY the JSON array."
    
    def _parse_response(self, text: str) -> list:
        """Parse JSON from LLM response."""
        text = text.strip()
        try:
            result = json.loads(text)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass
        
        import re
        if "```" in text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                    return result if isinstance(result, list) else [result]
                except: pass
        
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                result = json.loads(match.group(0))
                return result if isinstance(result, list) else [result]
            except: pass
        
        return [{"action": "say", "text": text}]
    
    async def get_code_suggestions(self, code_context: dict, image_b64: Optional[str] = None) -> list:
        """Get code suggestions using local brain."""
        prompt = f"Suggest improvements for this {code_context.get('language', 'code')}:\n{code_context.get('current_line', '')}"
        return await self.get_actions(prompt, code_context, image_b64)
