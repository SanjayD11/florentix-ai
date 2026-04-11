"""
Florentix AI — Vercel Serverless Orchestrator (predict.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Intelligent routing layer deployed on Vercel that:
  1. Tries the Render/Railway TFLite backend (3s timeout, 1 fast retry)
  2. Evaluates confidence — pivots to OpenRouter Vision if < 70%
  3. Normalises every response into a unified schema
  4. Never returns an empty or broken response to the user
"""

import os
import re
import time
import json
import base64
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Florentix AI Cloud Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIGURATION (all secrets via Vercel Environment Variables) ────────────
RENDER_BACKEND_URL = os.environ.get("RENDER_BACKEND_URL", "http://localhost:8000")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Vision-capable models for plant diagnosis fallback
PRIMARY_VISION_MODEL = "google/gemma-3-27b-it:free"
FALLBACK_VISION_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
EMERGENCY_VISION_MODEL = "openrouter/auto"

# Timeout configuration (seconds)
RENDER_TIMEOUT_FIRST  = 3   # First attempt — aggressive
RENDER_TIMEOUT_RETRY  = 2   # Fast retry — even more aggressive
OPENROUTER_TIMEOUT    = 55  # Cloud AI — generous (complex inference)
CONFIDENCE_THRESHOLD  = 70  # Below this → pivot to cloud

# ─── RESPONSE SCHEMA HINT ────────────────────────────────────────────────────
JSON_SCHEMA_HINT = """{
  "plant": "<common plant name>",
  "condition": "<specific disease name or 'Healthy'>",
  "confidence": <integer 50-100>,
  "summary": "<2-3 sentence diagnosis summary>",
  "symptoms": ["<Symptom 1>", "<Symptom 2>", "<Symptom 3>"],
  "treatment": ["<Title: Detail>", "<Title: Detail>", "<Title: Detail>"],
  "prevention": ["<Tip 1>", "<Tip 2>"],
  "severity": "<Low | Medium | High>",
  "optimal_temperature": "<e.g. 20-30°C>",
  "light_requirement": "<e.g. Moderate (8,000–15,000 lux)>"
}"""


def build_vision_prompt(strict: bool = False) -> str:
    """Build the prompt for OpenRouter vision models."""
    preamble = (
        "CRITICAL: Return ONLY valid JSON. No explanations, no markdown, no backticks. "
        "The very first character must be '{' and the last must be '}'.\n\n"
        if strict else ""
    )
    return f"""{preamble}You are a certified plant pathologist with 20 years of field experience.

Analyze the plant leaf image and return a precise diagnostic JSON.

STRICT RULES:
- Response must be valid JSON only — no text before or after.
- DO NOT wrap output in markdown code blocks.
- DO NOT hallucinate diseases or products.
- NEVER return empty strings or "Unknown".
- Confidence: integer 50-100 (50-65 if uncertain, 75-100 if confident).
- Treatment items MUST follow "Title: Detail" format with a colon separator.
- If healthy: set condition to "Healthy" and severity to "Low".

Return ONLY this JSON:
{JSON_SCHEMA_HINT}"""


# ─── JSON PARSING ────────────────────────────────────────────────────────────
def extract_json(raw: str) -> dict | None:
    """Extract and parse JSON from potentially messy LLM output."""
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r'^```[a-z]*\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)

    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    text = text[start:end + 1]

    # Fix trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: strip unescaped newlines inside strings
    text = re.sub(r'(?<!\\)\n', ' ', text)
    text = re.sub(r'(?<!\\)\r', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalise_response(data: dict, source: str = "cloud") -> dict:
    """Normalise any response into the unified Florentix schema."""
    raw_conf = data.get("confidence", 75)
    try:
        conf = max(0, min(100, int(float(str(raw_conf)))))
    except (ValueError, TypeError):
        conf = 75

    condition = (
        data.get("condition")
        or data.get("prediction")
        or data.get("disease")
        or "Analysis pending"
    )
    summary = data.get("summary") or data.get("analysis") or ""
    severity = data.get("severity") or data.get("risk_level") or "Medium"

    if conf < 60 and "low confidence" not in summary.lower():
        summary += " Low confidence — consider retaking with better lighting."

    return {
        "source":       source,
        "is_fallback":  source != "local",
        "plant":        data.get("plant", "Plant"),
        "prediction":   condition,
        "confidence":   str(conf),
        "analysis":     summary,
        "symptoms":     data.get("symptoms") or ["Visual symptoms present"],
        "treatment":    data.get("treatment") or ["Consult a specialist."],
        "prevention":   data.get("prevention") or [],
        "risk_level":   severity,
        "optimal_temperature": data.get("optimal_temperature", ""),
        "light_requirement":   data.get("light_requirement", ""),
        "model_used":   data.get("model_used", "AI Vision Scan"),
    }


# ─── OPENROUTER VISION CALL ─────────────────────────────────────────────────
def call_openrouter_vision(
    model: str, image_b64: str, mime_type: str, strict: bool = False
) -> dict | None:
    """Call an OpenRouter vision model and return parsed dict or None."""
    if not OPENROUTER_API_KEY:
        print("[Orchestrator] OPENROUTER_API_KEY is not set!")
        return None

    prompt = build_vision_prompt(strict=strict)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{image_b64}"
                    }}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1200,
    }

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://florentix-ai.vercel.app",
                "X-Title": "Florentix AI",
            },
            json=payload,
            timeout=OPENROUTER_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return extract_json(raw)
    except Exception as e:
        print(f"[OpenRouter] {model} failed: {e}")
        return None


# ─── RENDER (TFLITE) CALL ────────────────────────────────────────────────────
def try_render(image_bytes: bytes, filename: str, mime_type: str) -> dict | None:
    """
    Try the Render/Railway TFLite backend with retry.
    Returns parsed JSON dict on success, None on failure.
    """
    render_endpoint = f"{RENDER_BACKEND_URL}/predict_local"
    files_payload = {"file": (filename, image_bytes, mime_type)}

    # Attempt 1: primary timeout
    try:
        res = requests.post(
            render_endpoint, files=files_payload, timeout=RENDER_TIMEOUT_FIRST
        )
        if res.status_code == 200:
            data = res.json()
            if "error" not in data:
                return data
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        print("[Orchestrator] Render attempt 1 timed out — fast retrying...")
    except Exception as e:
        print(f"[Orchestrator] Render attempt 1 error: {e}")

    # Attempt 2: fast retry with tighter timeout
    try:
        files_payload_retry = {"file": (filename, image_bytes, mime_type)}
        res = requests.post(
            render_endpoint, files=files_payload_retry, timeout=RENDER_TIMEOUT_RETRY
        )
        if res.status_code == 200:
            data = res.json()
            if "error" not in data:
                return data
    except Exception as e:
        print(f"[Orchestrator] Render fast retry failed: {e}")

    return None


# ─── CLOUD FALLBACK CASCADE ─────────────────────────────────────────────────
def cloud_fallback(image_b64: str, mime_type: str) -> dict:
    """
    Multi-layer OpenRouter fallback:
      1. Primary vision model (strict prompt)
      2. Fallback vision model
      3. Emergency auto-router
      4. Hard-coded safe response (never fails)
    """
    attempts = []

    # Layer 1: Primary vision model
    result = call_openrouter_vision(PRIMARY_VISION_MODEL, image_b64, mime_type, strict=False)
    if result:
        attempts.append(f"{PRIMARY_VISION_MODEL} → OK")
        return normalise_response(result, source="cloud")
    attempts.append(f"{PRIMARY_VISION_MODEL} → FAIL")

    # Layer 1b: Retry with strict prompt
    result = call_openrouter_vision(PRIMARY_VISION_MODEL, image_b64, mime_type, strict=True)
    if result:
        attempts.append(f"{PRIMARY_VISION_MODEL} (strict) → OK")
        return normalise_response(result, source="cloud")
    attempts.append(f"{PRIMARY_VISION_MODEL} (strict) → FAIL")

    # Layer 2: Fallback vision model
    result = call_openrouter_vision(FALLBACK_VISION_MODEL, image_b64, mime_type, strict=True)
    if result:
        attempts.append(f"{FALLBACK_VISION_MODEL} → OK")
        return normalise_response(result, source="cloud")
    attempts.append(f"{FALLBACK_VISION_MODEL} → FAIL")

    # Layer 3: Emergency auto-router
    result = call_openrouter_vision(EMERGENCY_VISION_MODEL, image_b64, mime_type, strict=True)
    if result:
        attempts.append(f"{EMERGENCY_VISION_MODEL} → OK")
        return normalise_response(result, source="cloud")
    attempts.append(f"{EMERGENCY_VISION_MODEL} → FAIL")

    # Layer 4: Hard-coded safe response (NEVER fails)
    print(f"[Orchestrator] All cloud models exhausted. Attempts: {attempts}")
    return {
        "source":      "error",
        "is_fallback":  True,
        "plant":        "Unknown",
        "prediction":   "Unable to Diagnose",
        "confidence":   "0",
        "analysis":     (
            "All AI systems are currently overloaded. "
            "Please try again in a few moments with a clear, well-lit image."
        ),
        "symptoms":     [],
        "treatment":    ["Retry: Try scanning again in 30 seconds."],
        "prevention":   [],
        "risk_level":   "Unknown",
        "model_used":   "None (all exhausted)",
    }


# ─── HEALTH ENDPOINT ─────────────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    """Quick health probe for uptime monitors."""
    return {
        "status": "Vercel Orchestrator Active",
        "render_url_set": RENDER_BACKEND_URL != "http://localhost:8000",
        "openrouter_key_set": bool(OPENROUTER_API_KEY),
    }


# ─── MAIN PREDICTION ENDPOINT ────────────────────────────────────────────────
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Intelligent prediction orchestrator:
      1. Try Render TFLite backend (3s + 2s retry)
      2. If success AND confidence ≥ 70 → return local result
      3. Else → cascade through OpenRouter vision models
      4. Guarantee: ALWAYS returns a valid response
    """
    start = time.time()
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    filename = file.filename or "upload.jpg"

    # ── Step 1: Try Render (TFLite) ──────────────────────────────────────
    render_data = try_render(image_bytes, filename, mime_type)

    if render_data:
        # Check confidence
        try:
            conf = float(render_data.get("confidence", 0))
        except (ValueError, TypeError):
            conf = 0

        if conf >= CONFIDENCE_THRESHOLD:
            # Local model succeeded with high confidence
            result = normalise_response(render_data, source="local")
            result["inference_time"] = round(time.time() - start, 3)
            print(f"[Orchestrator] ✓ Local result (conf={conf}%) in {result['inference_time']}s")
            return JSONResponse(result)
        else:
            print(f"[Orchestrator] Local confidence too low ({conf}%), pivoting to cloud...")

    # ── Step 2: Cloud Fallback ───────────────────────────────────────────
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    cloud_result = cloud_fallback(image_b64, mime_type)
    cloud_result["inference_time"] = round(time.time() - start, 3)
    print(f"[Orchestrator] Cloud result in {cloud_result['inference_time']}s")
    return JSONResponse(cloud_result)
