import os
import re
import time
import json
import base64
import requests
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import asyncio

# Prevent Render RAM overload during high traffic bursts
inference_semaphore = asyncio.Semaphore(3)

# ─── LOCAL MODEL SETUP ───────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    import numpy as np
    from backend.utils.image_utils import preprocess_image
    from backend.utils.remedies import REMEDIES
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: tflite_runtime/Numpy import failed: {e}")
    TENSORFLOW_AVAILABLE = False
    tflite = np = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if TENSORFLOW_AVAILABLE:
    TFLITE_PATH = os.path.join(BASE_DIR, "..", "model", "plant_model.tflite")
    try:
        interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"Warning: Failed to load TFLite model: {e}")
        interpreter = None
else:
    interpreter = None

CLASS_NAMES = [
    "Early_Blight",
    "Healthy",
    "Late_Blight",
    "Leaf_Mold",
    "Septoria_Leaf_Spot"
]

# ─── OPENROUTER CONFIG ────────────────────────────────────────────────────────
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Primary model — Gemma 3 27B (vision-capable, free, excellent at structured JSON)
PRIMARY_MODEL  = "google/gemma-3-27b-it:free"

# Fallback model — Llama 3.3 70B (free, very strong reasoning)
FALLBACK_MODEL_VISION = "meta-llama/llama-3.3-70b-instruct:free"

# Emergency catch-all — auto-routes to any available free model
EMERGENCY_MODEL = "openrouter/auto"

# ─── AI DOCTOR CHAT MODEL STACK (PARALLEL RACING) ──────────────────────────
RACING_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",  # Fast, confirmed live
    "qwen/qwen-2.5-7b-instruct",          # Strong reasoning, confirmed live
    "meta-llama/llama-3.2-1b-instruct",  # Lightweight speed fallback
]

# Chat model routing (used by choose_model())
CHAT_PRIMARY_MODEL = "meta-llama/llama-3.2-3b-instruct"   # Fast for simple queries
CHAT_COMPLEX_MODEL = "qwen/qwen-2.5-7b-instruct"          # Stronger for complex queries
CHAT_FAST_MODEL    = "meta-llama/llama-3.2-1b-instruct"   # Ultra-fast for weather insights

CONNECT_TIMEOUT = 4    # aggressive connect timeout to fail-fast
READ_TIMEOUT    = 15   # seconds between chunks; aborts hanging models quickly

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": os.environ.get("APP_URL", "https://florentix-ai.vercel.app"),
    "X-Title": "Florentix AI"
}

# ─── RESPONSE SCHEMA ──────────────────────────────────────────────────────────
JSON_SCHEMA_HINT = """{
  "plant": "<common plant name, e.g. Tomato, Rose, Monstera>",
  "condition": "<specific disease name, e.g. Early Blight, Leaf Curl, or 'Healthy'>",
  "confidence": <integer between 50 and 100>,
  "summary": "<2-3 sentences: what you see, why this diagnosis is likely, progression risk>",
  "symptoms": [
    "<Observable symptom 1, specific to image>",
    "<Observable symptom 2>",
    "<Observable symptom 3>"
  ],
  "treatment": [
    "<Step 1 — Immediate action: e.g. Remove all visibly infected leaves and dispose away from the plant>",
    "<Step 2 — Treatment method: e.g. Apply a copper-based fungicide every 7-10 days until symptoms clear>",
    "<Step 3 — Recovery support: e.g. Ensure proper spacing for airflow and reduce leaf wetness>"
  ],
  "prevention": [
    "<Practical prevention tip 1>",
    "<Practical prevention tip 2>"
  ],
  "severity": "<Low | Medium | High>",
  "optimal_temperature": "<e.g., 20-30°C>",
  "light_requirement": "<e.g., Moderate (8,000–15,000 lux)>",
  "model_used": "AI Vision Scan"
}"""

# Smart routing for chat models is handled dynamically in the backend stream later

# ─── PROMPT TEMPLATES ────────────────────────────────────────────────────────
def build_prompt(strict: bool = False) -> str:
    """
    Build the prompt for the vision model.
    strict=True produces an even more constrained retry prompt.
    """
    preamble = (
        "CRITICAL: Return ONLY valid JSON. No explanations, no markdown, no backticks. "
        "The very first character must be '{' and the last must be '}'.\n\n"
        if strict else ""
    )

    return f"""{preamble}You are a certified plant pathologist with 20 years of field experience diagnosing crop diseases.

Analyze the plant leaf image provided and return a precise, structured diagnosis.

STRICT RULES:
- Your ENTIRE response must be valid JSON only — no text before or after.
- DO NOT wrap output in markdown code blocks or backticks.
- DO NOT hallucinate chemical names, fake products, or non-existent diseases.
- NEVER return empty strings or null values.
- NEVER say "Unknown", "Unidentified", or "Unavailable" — always give your best diagnosis.
- If healthy: set condition to "Healthy" and severity to "Low".
- Confidence must be an integer between 50 and 100.
  - Use 50-65 if image quality is poor or you are uncertain.
  - Use 75-100 if diagnosis is clear and confident.
- Treatments and preventions must be specific, actionable sentences.
- VERY IMPORTANT: Every single item in the "treatment" and "prevention" arrays MUST strictly follow the format "Title: Detailed description". You must include a colon ':' to separate the short title from the description. Example: "Pruning: Remove all visibly infected leaves immediately."
- If confidence is below 65, include this note at the end of summary:
  "Low confidence prediction — consider retaking with better lighting or a closer shot."
- All field names must exactly match the schema below.

Return ONLY this JSON structure:
{JSON_SCHEMA_HINT}"""


# ─── JSON PARSE + SANITIZE ───────────────────────────────────────────────────
def extract_and_parse_json(raw: str) -> dict | None:
    """
    Attempt to extract and parse a JSON object from the raw LLM response.
    Returns the parsed dict or None if it cannot be recovered.
    """
    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r'^```[a-z]*\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)

    # Grab content between outermost { }
    start = text.find('{')
    end   = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    text = text[start:end + 1]

    # Remove trailing commas before ] or }
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: replace unescaped newlines inside string literals
    text = re.sub(r'(?<!\\)\n', ' ', text)
    text = re.sub(r'(?<!\\)\r', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_output(data: dict) -> list[str]:
    """
    Returns a list of validation errors. Empty list = valid.
    """
    errors = []
    BAD_VALUES = {"unknown", "unidentified", "unavailable", "n/a", "", "none", "null"}

    plant     = str(data.get("plant", "")).strip().lower()
    condition = str(data.get("condition", "")).strip().lower()
    conf      = data.get("confidence", 0)
    symptoms  = data.get("symptoms", [])
    treatment = data.get("treatment", [])

    if not plant or plant in BAD_VALUES:
        errors.append("plant is empty or invalid")
    if not condition or condition in BAD_VALUES:
        errors.append("condition is empty or invalid")
    try:
        conf_val = int(float(str(conf)))
        if conf_val <= 0:
            errors.append("confidence is 0 or negative")
    except (ValueError, TypeError):
        errors.append("confidence is not a number")
    if not isinstance(symptoms, list) or len(symptoms) < 2:
        errors.append("symptoms has fewer than 2 entries")
    if not isinstance(treatment, list) or len(treatment) < 2:
        errors.append("treatment has fewer than 2 entries")

    return errors


def normalise_output(data: dict, model_label: str = "AI Vision Scan") -> dict:
    """
    Ensure all required keys exist and values are non-null strings/lists.
    Merges 'condition' into 'disease' alias for frontend compatibility.
    Appends low-confidence note if appropriate.
    """
    # Coerce confidence to int
    raw_conf = data.get("confidence", 75)
    try:
        conf_int = max(0, min(100, int(float(str(raw_conf)))))
    except (ValueError, TypeError):
        conf_int = 75

    # Merge condition -> disease (frontend uses 'prediction')
    condition = data.get("condition") or data.get("disease") or "No visible disease detected"
    summary   = data.get("summary") or data.get("analysis") or ""

    if conf_int < 60 and "low confidence" not in summary.lower():
        summary += (" Low confidence prediction — consider retaking with better lighting or a closer view of the affected area.")

    severity  = data.get("severity") or data.get("risk_level") or "Medium"

    return {
        "plant":      data.get("plant", "Plant"),
        "prediction": condition,
        "confidence": str(conf_int),
        "analysis":   summary,
        "symptoms":   data.get("symptoms") or ["Visual symptoms present"],
        "treatment":  data.get("treatment") or ["Consult a plant specialist for advanced treatment options."],
        "prevention": data.get("prevention") or [],
        "risk_level": severity,
        "optimal_temperature": data.get("optimal_temperature", "Unknown"),
        "light_requirement": data.get("light_requirement", "Unknown"),
        "notes":      data.get("notes") or "This is an AI-assisted diagnosis. For critical cases, consult an agronomist.",
        "model_used": model_label
    }


# ─── OPENROUTER CALL ─────────────────────────────────────────────────────────
def call_openrouter(model: str, image_url: str, prompt: str, timeout: int = 50) -> str | None:
    """
    Call OpenRouter and return the raw content string, or None on failure.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "temperature": 0.1,      # very low temperature for more deterministic output
        "max_tokens": 1200,
    }

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers=HEADERS,
            json=payload,
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        print(f"[OpenRouter] Error calling {model}: {e}")
        return None


# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(title="Florentix AI API")

# NOTE: /config endpoint removed — API keys must NEVER be exposed to the client.
# All API key usage is server-side only (backend → OpenRouter).

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CHAT CONFIG ─────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are FloraSense, the expert AI Plant Doctor for the Florentix app.

PERSONALITY:
- Friendly, warm, and conversational. You are a helpful gardening companion.
- For greetings (hi, hello, etc.) or personal questions (name, who are you), respond naturally and briefly.
- For general plant info (e.g., "tell me about roses"), provide a conversational, informative summary.
- ONLY use the structured diagnostic format if the user describes a specific health PROBLEM (spots, yellowing, pests, dying).

STRUCTURED DIAGNOSTIC FORMAT (Only for illness/problems):
🌱 **Problem Summary**: Brief overview.
🔍 **Likely Causes**: Ranked 1-2 probable causes.
🛠️ **Recommended Actions**: Step-by-step treatment.
⚠️ **Mistakes to Avoid**: Common errors.
📌 **Care Tip**: One prevention tip.

BEHAVIOR RULES:
- If no problem is mentioned, DO NOT force the diagnostic format.
- Stick to plant-related topics. Politely redirect others.
- Your name is FloraSense. You are part of the Florentix system.
- Be concise but expert."""

# ─── SMART QUERY CLASSIFIER ──────────────────────────────────────────────────

_CASUAL_TRIGGERS = {
    "hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you",
    "what is your name", "what's your name", "who are you", "your name",
    "how are you", "good morning", "good evening", "good night", "ok", "okay",
    "lol", "haha", "nice", "cool", "great", "awesome", "sure", "yes", "no",
    "what can you do", "help", "how do you work"
}

_PLANT_KEYWORDS = {
    "plant", "leaf", "leaves", "root", "soil", "water", "watering", "sunlight",
    "fertilize", "fertilizer", "disease", "pest", "bug", "insect", "fungus",
    "mold", "blight", "rot", "wilt", "yellow", "brown", "spot", "drop",
    "grow", "prune", "repot", "seed", "flower", "fruit", "tomato", "mango",
    "rose", "pothos", "orchid", "cactus", "succulent", "basil", "mint",
    "crop", "harvest", "garden", "spray", "nitrogen", "phosphorus",
    "compost", "mulch", "ph", "humidity", "temperature", "overwater",
    "underwater", "nutrient", "deficiency", "strawberry", "mold",
}

def is_health_issue(text: str) -> bool:
    """Returns True if the user is describing a plant problem that needs diagnosis."""
    text = text.lower()
    problem_words = {
        "yellow", "brown", "black", "spot", "wilt", "die", "dying", "sick", "pest",
        "insect", "bug", "mold", "fungus", "rot", "drop", "fallen", "fall", "dry",
        "crispy", "shrivel", "curling", "bite", "hole", "burned", "scorched", "blight",
        "mildew", "rust", "stunted", "slow growth", "pale", "limp", "soft", "mushy"
    }
    return any(word in text for word in problem_words)

def is_plant_query(text: str) -> bool:
    """Returns True if the message is generally about plants."""
    lower = text.lower().strip()
    for trigger in _CASUAL_TRIGGERS:
        if lower == trigger or lower.startswith(trigger + " ") or lower.endswith(" " + trigger):
            return False
    for word in _PLANT_KEYWORDS:
        if word in lower:
            return True
    return False


# ─── IN-MEMORY CHAT STORE (Production-schema; swap .get() for DB later) ──────
# Schema: { user_id: [ { role, content, timestamp } ] }
_chat_store: dict = {}

def get_user_history(user_id: str) -> list:
    """Return the chat history for a user. Replace body with DB call when ready."""
    return _chat_store.get(user_id, [])

def append_to_user_history(user_id: str, role: str, content: str) -> None:
    """Append a message to the in-memory store."""
    import datetime
    if user_id not in _chat_store:
        _chat_store[user_id] = []
    _chat_store[user_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })


def build_plant_context_block(plant_ctx: dict) -> str:
    """Compressed plain-text context block from one scan — max 6 lines, no raw JSON."""
    if not plant_ctx:
        return ""
    lines = [
        "━━━ PLANT CONTEXT ━━━",
        f"Plant: {plant_ctx.get('plant', 'Unknown')}",
        f"Disease: {plant_ctx.get('prediction', 'Unknown')} ({plant_ctx.get('confidence', '?')}% confidence)",
        f"Risk: {plant_ctx.get('risk_level', 'Unknown')}",
    ]
    symptoms = plant_ctx.get("symptoms", [])
    if symptoms:
        lines.append("Key Symptoms: " + ", ".join(s.strip().lstrip("-•* ") for s in symptoms[:3]))
    treatment = plant_ctx.get("treatment", [])
    if treatment:
        lines.append("Treatment: " + "; ".join(t.strip().lstrip("-•* ") for t in treatment[:3]))
    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def build_global_context_summary(all_scans: list) -> str:
    """Concise multi-plant one-liner summary — no raw JSON dump."""
    if not all_scans:
        return ""
    lines = ["━━━ PLANT HEALTH SUMMARY ━━━"]
    for scan in all_scans[:8]:
        plant = scan.get("plant", "Unknown")
        pred  = scan.get("prediction", "Unknown")
        conf  = scan.get("confidence", "?")
        risk  = scan.get("risk_level", "Medium")
        lines.append(f"  • {plant} -> {pred} ({risk} risk, {conf}%)")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


# ─── SMART MODEL ROUTER ──────────────────────────────────────────────────────
COMPLEX_KEYWORDS = {
    "compare", "analyse", "analyze", "analysis", "explain", "difference",
    "what causes", "why does", "how does", "vs", "versus", "strategy",
    "prevention plan", "treatment plan", "full plan", "detail", "detailed",
    "history", "summarize", "summary", "evaluate", "assessment", "diagnose",
    "step by step", "step-by-step", "comprehensive", "thorough", "elaborate",
}

def is_complex_query(text: str) -> bool:
    """Heuristic logic for production-grade model routing."""
    text_lower = text.lower()
    return (
        len(text.split()) > 25 or
        "compare" in text_lower or
        "difference" in text_lower or
        "explain in detail" in text_lower
    )

def choose_model(messages: list) -> str:
    """Pick primary model based on query complexity."""
    if not messages:
        return CHAT_PRIMARY_MODEL
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return CHAT_COMPLEX_MODEL if is_complex_query(last_user) else CHAT_PRIMARY_MODEL


# ─── CHAT ENDPOINT (SSE STREAMING) ────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    plant_context: Optional[dict] = None
    all_scans: Optional[list] = None
    mode: str = "plant"


@app.post("/chat")
async def chat_with_doctor(req: ChatRequest):
    """
    Elite Parallel Racing Architecture: fires all RACING_MODELS simultaneously.
    First valid streaming response wins; others are cancelled safely.
    Includes first-byte timeout, empty-stream validation, and retry fallback.
    """
    import json as _json
    import asyncio
    import anyio
    import httpx

    context_block = ""
    p_ctx = req.plant_context
    if req.mode == "plant" and isinstance(p_ctx, dict) and p_ctx:
        try:
            plant_name    = str(p_ctx.get("plant", "Unknown"))
            disease_name  = str(p_ctx.get("disease", p_ctx.get("prediction", "Unknown")))
            conf_val      = str(p_ctx.get("confidence", "?"))
            symptoms_list = p_ctx.get("symptoms", [])
            symptoms_str  = ", ".join(symptoms_list) if isinstance(symptoms_list, list) else "None"
            context_block = (
                "Plant: " + plant_name + "\n" +
                "Disease: " + disease_name + "\n" +
                "Confidence: " + conf_val + "\n" +
                "Symptoms: " + symptoms_str
            )
        except Exception:
            context_block = "Context parsing failed. Proceed with general knowledge."
    elif req.mode == "global" and isinstance(req.all_scans, list) and req.all_scans:
        context_block = "User has " + str(len(req.all_scans)) + " previous scans. Provide general botanical advice."

    full_system = CHAT_SYSTEM_PROMPT
    if context_block:
        full_system += "\n\nCONTEXT:\n" + context_block

    user_query = req.messages[-1].content if req.messages else ""
    
    if is_health_issue(user_query):
        # Only wrap if it's a specific PROBLEM
        instruction = "The user is reporting a plant health problem. Please diagnose and use the structured format."
        enhanced_query = f"{instruction}\n\nUser Query: {user_query}"
    else:
        # Keep it natural for general info or casual chat
        enhanced_query = user_query

    or_messages = [{"role": "system", "content": full_system}]
    for msg in req.messages[:-1][-7:]:
        or_messages.append({"role": msg.role, "content": msg.content})
    or_messages.append({"role": "user", "content": enhanced_query})

    base_payload = {
        "messages":    or_messages,
        "temperature": 0.4,
        "max_tokens":  512,
        "stream":      True,
    }

    async def stream_generator():
        import json as _json
        import asyncio
        import anyio
        import httpx

        try:
            yield "event: ping\ndata: {}\n\n"

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=4.0, read=15.0, write=5.0, pool=5.0)
            ) as client:

                async def fetch_model_stream(model_name):
                    """Returns (model_name, response, byte_iterator, first_chunk) or None."""
                    payload = dict(base_payload, model=model_name)
                    try:
                        with anyio.fail_after(3.0):
                            req_obj  = client.build_request("POST", OPENROUTER_API_URL, json=payload, headers=HEADERS)
                            response = await client.send(req_obj, stream=True)
                            if response.status_code != 200:
                                await response.aclose()
                                return None
                            # Create ONE persistent iterator and peek first chunk
                            byte_iter   = response.aiter_bytes().__aiter__()
                            first_chunk = await byte_iter.__anext__()
                            if first_chunk and first_chunk.strip():
                                return (model_name, response, byte_iter, first_chunk)
                            await response.aclose()
                            return None
                    except StopAsyncIteration:
                        return None
                    except Exception:
                        return None

                async def retry_once():
                    await asyncio.sleep(1.0)
                    return await fetch_model_stream("meta-llama/llama-3.2-3b-instruct")

                winner      = None
                pending_tasks = []

                with anyio.move_on_after(8.0):
                    tasks         = [asyncio.create_task(fetch_model_stream(m)) for m in RACING_MODELS]
                    pending_tasks = list(tasks)

                    while pending_tasks and not winner:
                        done, pending_tasks = await asyncio.wait(
                            pending_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for t in done:
                            res = t.result()
                            if res is not None:
                                winner = res
                                break

                # Cancel remaining tasks safely
                for task in pending_tasks:
                    task.cancel()
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)

                if not winner:
                    print("[AI Racing] All parallel models failed retrying with llama-3.2-3b...")
                    winner = await retry_once()

                if not winner:
                    raise Exception("All models exhausted including retry.")

                winner_model, winner_resp, byte_iter, first_chunk = winner
                print("[AI Racing] Winner: " + winner_model)

                sse_buffer = ""

                def process_buffer(buf):
                    events    = buf.split("\n\n")
                    remaining = events.pop()
                    tokens    = []
                    done      = False
                    for event_block in events:
                        for ln in event_block.strip().split("\n"):
                            ln = ln.strip()
                            if not ln.startswith("data:"):
                                continue
                            data_str = ln[5:].strip()
                            if data_str == "[DONE]":
                                done = True
                                break
                            try:
                                dj  = _json.loads(data_str)
                                tok = dj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if tok:
                                    tokens.append(_json.dumps({"content": tok}))
                            except Exception:
                                pass
                        if done:
                            break
                    return tokens, done, remaining

                # Process peeked first chunk
                sse_buffer += first_chunk.decode("utf-8", errors="replace")
                tokens, done, sse_buffer = process_buffer(sse_buffer)
                for tok in tokens:
                    yield "data: " + tok + "\n\n"
                if done:
                    yield "data: [DONE]\n\n"
                    return

                # Continue streaming using the SAME iterator (no StreamConsumed)
                async for chunk in byte_iter:
                    if not chunk:
                        continue
                    sse_buffer += chunk.decode("utf-8", errors="replace")
                    tokens, done, sse_buffer = process_buffer(sse_buffer)
                    for tok in tokens:
                        yield "data: " + tok + "\n\n"
                    if done:
                        yield "data: [DONE]\n\n"
                        return

                yield "data: [DONE]\n\n"

        except Exception as major_exc:
            print("[AI Racing] Fatal failure: " + str(major_exc))
            err_payload = _json.dumps({
                "message": (
                    "All AI providers are currently busy or rate-limited. "
                    "Please wait a few seconds and try again. "
                    "Your message has been saved."
                )
            })
            yield "event: error\ndata: " + err_payload + "\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/history")
async def get_chat_history(user_id: str = ""):
    """
    Returns the chat history for a user.
    Currently backed by in-memory store; swap get_user_history() for DB later.
    Response schema is final and will not change.
    """
    if not user_id:
        return {"user_id": None, "history": [], "count": 0}
    history = get_user_history(user_id)
    return {
        "user_id": user_id,
        "history": history,
        "count":   len(history)
    }

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "message": "Florentix AI API is live",
        "primary_model": PRIMARY_MODEL,
        "local_model_ready": interpreter is not None
    }


@app.post("/predict_local")
async def predict_disease_local(file: UploadFile = File(...)):
    """Ultra-fast local model execution for the Vercel Orchestrator."""
    if not interpreter or not TENSORFLOW_AVAILABLE:
        return {"error": "local_model_offline", "message": "TensorFlow/TFLite is not running."}

    image_bytes = await file.read()
    
    async with inference_semaphore: # Prevents hitting RAM limits on Free/Starter tiers
        try:
            processed_image = preprocess_image(image_bytes)
            dtype = input_details[0]["dtype"]
            interpreter.set_tensor(input_details[0]["index"], processed_image.astype(dtype))
            interpreter.invoke()
            local_preds = interpreter.get_tensor(output_details[0]["index"])
            confidence = float(np.max(local_preds))
            local_class = CLASS_NAMES[int(np.argmax(local_preds))]

            lookup_key = "Healthy" if "healthy" in local_class.lower() else next(
                (k for k in REMEDIES if k.lower() == local_class.lower()), "Healthy"
            )
            local_info = REMEDIES.get(lookup_key, REMEDIES["Healthy"])

            return {
                "plant": "Tomato",
                "condition": f"{local_class.replace('_', ' ')} (Expert Model)",
                "prediction": f"{local_class.replace('_', ' ')} (Expert Model)",
                "confidence": str(round(confidence * 100, 1)),
                "symptoms": [local_info.get("description", "Identified by tomato-specific expert model.")],
                "treatment": local_info.get("remedies", []) + local_info.get("care_tips", []),
                "prevention": local_info.get("care_tips", []),
                "risk_level": "Low" if "healthy" in local_class.lower() else "Medium",
                "model_used": "Tomato-Specific AI Model"
            }
        except Exception as e:
            return {"error": "inference_failed", "message": str(e)}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    # ── 0. File validation ─────────────────────────────────────────────────
    file_size = getattr(file, "size", None)
    if file_size is not None and file_size > 8 * 1024 * 1024:
        return {"error": "file_too_large", "message": "Max file size is 8MB."}

    image_bytes = await file.read()
    start_time  = time.time()

    mime_type = file.content_type if file.content_type else "image/jpeg"
    b64       = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{b64}"

    # ── 1. Try Primary Model ───────────────────────────────────────────────
    parsed_data  = None
    model_used   = "AI Vision Scan"
    attempt_log  = []

    prompt_v1 = build_prompt(strict=False)
    raw = call_openrouter(PRIMARY_MODEL, image_url, prompt_v1, timeout=50)

    if raw:
        parsed_data = extract_and_parse_json(raw)
        attempt_log.append(f"Primary model ({PRIMARY_MODEL}) — parse {'OK' if parsed_data else 'FAILED'}")

        # ── 2. If parse failed OR validation fails -> retry with strict prompt ──
        if parsed_data:
            errs = validate_output(parsed_data)
            if errs:
                print(f"[Validation] Primary output failed: {errs}. Retrying with strict prompt...")
                attempt_log.append(f"Validation errors: {errs}")
                parsed_data = None  # force retry

        if not parsed_data:
            prompt_v2 = build_prompt(strict=True)
            raw2 = call_openrouter(PRIMARY_MODEL, image_url, prompt_v2, timeout=50)
            if raw2:
                parsed_data = extract_and_parse_json(raw2)
                attempt_log.append(f"Primary model retry — parse {'OK' if parsed_data else 'FAILED'}")
            else:
                attempt_log.append("Primary model retry — no response")
    else:
        attempt_log.append(f"Primary model ({PRIMARY_MODEL}) — no response")

    # ── 3. Validate retry result; trigger fallback if still bad ───────────
    if parsed_data:
        errs = validate_output(parsed_data)
        if errs:
            print(f"[Validation] Retry output still invalid: {errs}. Triggering fallback model...")
            attempt_log.append(f"Retry validation errors: {errs} — switching to fallback")
            parsed_data = None

    # ── 4. Fallback Model ──────────────────────────────────────────────────
    if not parsed_data:
        prompt_fallback = build_prompt(strict=True)
        raw_fb = call_openrouter(FALLBACK_MODEL_VISION, image_url, prompt_fallback, timeout=55)
        if raw_fb:
            parsed_data = extract_and_parse_json(raw_fb)
            attempt_log.append(f"Fallback model ({FALLBACK_MODEL_VISION}) — parse {'OK' if parsed_data else 'FAILED'}")
        else:
            attempt_log.append(f"Fallback model ({FALLBACK_MODEL_VISION}) — no response")

    # ── 4b. Emergency Model — auto-routes to any available free model ─────
    if not parsed_data:
        raw_em = call_openrouter(EMERGENCY_MODEL, image_url, build_prompt(strict=True), timeout=60)
        if raw_em:
            parsed_data = extract_and_parse_json(raw_em)
            attempt_log.append(f"Emergency model ({EMERGENCY_MODEL}) — parse {'OK' if parsed_data else 'FAILED'}")
        else:
            attempt_log.append(f"Emergency model ({EMERGENCY_MODEL}) — no response")

    # ── 5. Final sanity: if still None, build a neutral response ──────────
    if not parsed_data:
        print(f"[Pipeline] All attempts failed. Attempts: {attempt_log}")
        return {
            "error": "parsing_failed",
            "message": (
                "The AI could not produce a structured response after multiple attempts. "
                "Please try again with a clearer, well-lit image of the leaf."
            ),
            "debug_attempts": attempt_log
        }

    # ── 6. Confidence-gated fallback: if confidence too low or "Unknown" ──
    try:
        raw_conf = int(float(str(parsed_data.get("confidence", 50))))
    except (ValueError, TypeError):
        raw_conf = 50

    condition_text = str(parsed_data.get("condition") or parsed_data.get("disease", "")).lower()

    if raw_conf < 40 or "unknown" in condition_text:
        print(f"[Confidence] Low confidence ({raw_conf}%) or unknown condition. Trying fallback...")
        attempt_log.append("Low confidence / unknown — re-triggering fallback")
        raw_fb2 = call_openrouter(FALLBACK_MODEL_VISION, image_url, build_prompt(strict=True), timeout=55)
        if raw_fb2:
            fb_parsed = extract_and_parse_json(raw_fb2)
            if fb_parsed and not validate_output(fb_parsed):
                parsed_data = fb_parsed
                attempt_log.append("Fallback improved result — using fallback output")

    # ── 7. Smart Routing: if plant is Tomato -> run local TFLite ──────────
    plant_name = str(parsed_data.get("plant", "")).strip()
    used_local = False

    if "tomato" in plant_name.lower() and TENSORFLOW_AVAILABLE and interpreter is not None:
        try:
            processed_image = preprocess_image(image_bytes)
            dtype = input_details[0]["dtype"]
            interpreter.set_tensor(input_details[0]["index"], processed_image.astype(dtype))
            interpreter.invoke()
            local_preds      = interpreter.get_tensor(output_details[0]["index"])
            local_confidence = float(np.max(local_preds))
            local_class      = CLASS_NAMES[int(np.argmax(local_preds))]

            parsed_data["condition"] = f"{local_class.replace('_', ' ')} (Expert Model)"
            parsed_data["confidence"] = str(round(local_confidence * 100, 1))

            lookup_key = "Healthy" if "healthy" in local_class.lower() else next(
                (k for k in REMEDIES if k.lower() == local_class.lower()), "Healthy"
            )
            local_info = REMEDIES.get(lookup_key, REMEDIES["Healthy"])

            parsed_data["symptoms"]  = [local_info.get("description", "Identified by tomato-specific expert model.")]
            parsed_data["treatment"] = local_info.get("remedies", []) + local_info.get("care_tips", [])
            used_local = True

        except Exception as local_err:
            print(f"[LocalModel] Routing failed, using cloud result. Error: {local_err}")

    model_label = "Tomato-Specific AI Model" if used_local else "General Plant AI Model"
    result      = normalise_output(parsed_data, model_label)
    result["inference_time"] = round(time.time() - start_time, 3)
    return result



# ─── LIVE ENVIRONMENT (WEATHER + AI) ──────────────────────────────────────────
import httpx
import json
from cachetools import TTLCache
from typing import Optional
from fastapi import Query
import time

weather_cache = TTLCache(maxsize=100, ttl=600)   # 10 mins raw weather
insight_cache = TTLCache(maxsize=100, ttl=900)   # 15 mins AI insight

@app.get("/api/weather/raw")
async def get_weather_raw(lat: float, lon: float):
    cache_key = f"{round(lat, 3)}_{round(lon, 3)}"
    if cache_key in weather_cache:
        return weather_cache[cache_key]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,uv_index&hourly=temperature_2m,relative_humidity_2m&daily=sunrise,sunset&timezone=auto"
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            weather_cache[cache_key] = data
            return data
        except Exception as e:
            return {"error": str(e)}

@app.get("/api/weather/insights")
async def get_weather_insights(temp: float, humidity: float, wind: float, uv: float, rain: float):
    cache_key = f"{temp}_{humidity}_{wind}_{uv}_{rain}"
    if cache_key in insight_cache:
        return insight_cache[cache_key]
        
    prompt = f"""You are a plant environment analysis system.

STRICT RULES:
- Only use the provided weather data
- Do NOT assume missing values
- Do NOT hallucinate
- Keep output strictly under 2 sentences. Max 30 words.

Weather Data:
Temperature: {temp}°C
Humidity: {humidity}%
Wind Speed: {wind} km/h
UV Index: {uv}
Precipitation: {rain} mm

Generate 1-2 actionable plant care insights."""

    try:
        # Use existing OPENROUTER logic
        payload = {
            "model": CHAT_FAST_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, # Low temperature for strictness
            "max_tokens": 100
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                OPENROUTER_API_URL,
                headers=HEADERS,
                json=payload,
                timeout=15.0
            )
            raw = resp.json()
            reply = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not reply:
                raise ValueError("Empty response")
                
            res = {"insight": reply}
            insight_cache[cache_key] = res
            return res
            
    except Exception as e:
        return {"insight": "Monitor plants closely as environment shifts.", "error": str(e)}

@app.get("/api/weather/combined")
async def get_weather_combined(lat: float, lon: float):
    raw_data = await get_weather_raw(lat, lon)
    if "error" in raw_data:
        return raw_data
        
    try:
        current = raw_data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]
        uv = current.get("uv_index", 0.0)
        rain = current["precipitation"]
        
        insight_data = await get_weather_insights(temp, humidity, wind, uv, rain)
        raw_data["plant_insight"] = insight_data["insight"]
    except Exception as e:
        print("[Weather Integration Error]", e)
        raw_data["plant_insight"] = "Monitor plants closely as environment shifts."
        
    return raw_data


# ─── STATIC FILES ─────────────────────────────────────────────────────────────
# Note: In the hybrid deployment, the frontend is hosted statically on Vercel.
# The FastAPI backend acts strictly as an API server and no longer serves HTML.

