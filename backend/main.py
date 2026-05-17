"""
Thai-English Translator Backend V3
===================================
FastAPI server with preloaded AI models for real-time translation.
Supports both REST API (solo mode) and WebSocket (room mode).
"""

import os
import base64
import tempfile
import logging
import io
import time
import asyncio
import secrets
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import scipy.io.wavfile as wavfile
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== USER MANAGEMENT ====================
USERS_FILE = Path(__file__).parent / "users.json"

def load_users() -> dict:
    """Load users from JSON file"""
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except:
            return {}
    return {}

def save_users(users: dict):
    """Save users to JSON file"""
    USERS_FILE.write_text(json.dumps(users, indent=2))

def hash_password(password: str) -> str:
    """Simple SHA256 hash for passwords"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_auth_token() -> str:
    """Generate a random auth token"""
    return secrets.token_hex(32)

def validate_token(authorization: str) -> str:
    """Validate token and return username. Raises HTTPException if invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.replace("Bearer ", "")
    users = load_users()

    for username, data in users.items():
        if data.get("token") == token:
            return username

    raise HTTPException(status_code=401, detail="Invalid token")

def validate_token_simple(token: str) -> str:
    """Validate token without Bearer prefix. Returns username or raises exception."""
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    users = load_users()
    for username, data in users.items():
        if data.get("token") == token:
            return username

    raise HTTPException(status_code=401, detail="Invalid token")

# ==================== GLOBAL MODELS ====================
whisper_model = None
nllb_model = None
nllb_tokenizer = None
tts_thai_model = None
tts_thai_processor = None
tts_eng_model = None
tts_eng_processor = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_CODES = {"en": "eng_Latn", "th": "tha_Thai"}

# ==================== ROOM MANAGEMENT ====================
@dataclass
class Room:
    code: str
    host_ws: WebSocket
    host_name: str
    guest_ws: Optional[WebSocket] = None
    guest_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)

rooms: dict[str, Room] = {}
model_lock = asyncio.Lock()
executor = ThreadPoolExecutor(max_workers=1)

def generate_room_code() -> str:
    """Generate a 4-digit room code"""
    while True:
        code = str(secrets.randbelow(9000) + 1000)
        if code not in rooms:
            return code

# ==================== MODEL LOADING ====================
def load_all_models():
    """Load all models at startup"""
    global whisper_model, nllb_model, nllb_tokenizer
    global tts_thai_model, tts_thai_processor, tts_eng_model, tts_eng_processor

    total_start = time.time()

    # Whisper
    logger.info("[1/4] Loading Whisper Large-v3...")
    start = time.time()
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info(f"       Done in {time.time() - start:.2f}s")

    # NLLB
    logger.info("[2/4] Loading NLLB-200...")
    start = time.time()
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    logger.info(f"       Done in {time.time() - start:.2f}s")

    # MMS-TTS Thai
    logger.info("[3/4] Loading MMS-TTS Thai...")
    start = time.time()
    from transformers import VitsModel, AutoProcessor
    tts_thai_processor = AutoProcessor.from_pretrained("facebook/mms-tts-tha")
    tts_thai_model = VitsModel.from_pretrained(
        "facebook/mms-tts-tha",
        torch_dtype=torch.float16
    ).to("cuda")
    logger.info(f"       Done in {time.time() - start:.2f}s")

    # MMS-TTS English
    logger.info("[4/4] Loading MMS-TTS English...")
    start = time.time()
    tts_eng_processor = AutoProcessor.from_pretrained("facebook/mms-tts-eng")
    tts_eng_model = VitsModel.from_pretrained(
        "facebook/mms-tts-eng",
        torch_dtype=torch.float16
    ).to("cuda")
    logger.info(f"       Done in {time.time() - start:.2f}s")

    # Warmup
    logger.info("[*] Warming up GPU...")
    with torch.no_grad():
        dummy = torch.randn(1, 100).to("cuda")
        _ = dummy * 2
    torch.cuda.synchronize()

    total_time = time.time() - total_start
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"ALL MODELS LOADED in {total_time:.2f}s")
    logger.info("=" * 50)
    logger.info("Server ready! Waiting for requests...")
    logger.info("")

# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("")
    logger.info("=" * 50)
    logger.info("THAI-ENGLISH TRANSLATOR V3")
    logger.info("=" * 50)
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info("")

    load_all_models()

    yield

    # Shutdown
    logger.info("Shutting down...")
    executor.shutdown(wait=False)

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Thai-English Translator V3",
    description="Real-time bidirectional translation with voice - Solo & Room modes",
    lifespan=lifespan
)

# CORS - allow all for ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== REQUEST/RESPONSE MODELS ====================
class TranslateRequest(BaseModel):
    audio: str  # base64 encoded audio

class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    detected_lang: str
    target_lang: str
    audio: str  # base64 encoded WAV
    timing: dict  # timing breakdown

class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    token: str
    username: str

# ==================== AUTH ENDPOINTS ====================
@app.post("/register", response_model=AuthResponse)
async def register(request: AuthRequest):
    """Register a new user account"""
    username = request.username.strip().lower()
    password = request.password

    if not username or len(username) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 characters")
    if not password or len(password) < 3:
        raise HTTPException(status_code=400, detail="Password must be at least 3 characters")

    users = load_users()

    if username in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    token = generate_auth_token()
    users[username] = {
        "password": hash_password(password),
        "token": token,
        "created_at": time.time()
    }
    save_users(users)

    logger.info(f"New user registered: {username}")
    return AuthResponse(token=token, username=username)

@app.post("/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """Login with existing account"""
    username = request.username.strip().lower()
    password = request.password

    users = load_users()

    if username not in users:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if users[username]["password"] != hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Generate new token on each login
    token = generate_auth_token()
    users[username]["token"] = token
    users[username]["last_login"] = time.time()
    save_users(users)

    logger.info(f"User logged in: {username}")
    return AuthResponse(token=token, username=username)

@app.get("/validate")
async def validate_session(authorization: str = Header(None)):
    """Validate a session token"""
    username = validate_token(authorization)
    return {"valid": True, "username": username}

# ==================== CORE FUNCTIONS ====================
def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    """Transcribe audio using Whisper, return (text, language)"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        torch.cuda.synchronize()
        segments, info = whisper_model.transcribe(temp_path, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        torch.cuda.synchronize()
        return text, info.language
    finally:
        os.unlink(temp_path)

def translate_text(text: str, source: str, target: str) -> str:
    """Translate text using NLLB"""
    nllb_tokenizer.src_lang = LANG_CODES[source]
    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    torch.cuda.synchronize()
    with torch.no_grad():
        generated = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(LANG_CODES[target]),
            max_length=256
        )
    torch.cuda.synchronize()

    return nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def text_to_speech(text: str, language: str) -> bytes:
    """Generate speech using MMS-TTS, return WAV bytes"""
    if language == "th":
        processor = tts_thai_processor
        model = tts_thai_model
    else:
        processor = tts_eng_processor
        model = tts_eng_model

    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    torch.cuda.synchronize()
    with torch.no_grad():
        output = model(**inputs)
    torch.cuda.synchronize()

    waveform = output.waveform[0].cpu().numpy()
    sample_rate = model.config.sampling_rate

    # Normalize
    waveform = waveform / np.max(np.abs(waveform)) * 0.9

    # Convert to WAV bytes
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, (waveform * 32767).astype(np.int16))
    return buffer.getvalue()

def process_audio_sync(audio_bytes: bytes) -> dict:
    """Synchronous audio processing for executor"""
    # STT
    original_text, detected_lang = transcribe(audio_bytes)
    if not original_text.strip():
        return None

    # Determine direction
    if detected_lang in ["en", "english"]:
        source, target = "en", "th"
    else:
        source, target = "th", "en"

    # Translate
    translated_text = translate_text(original_text, source, target)

    # TTS
    audio_output = text_to_speech(translated_text, target)
    audio_base64 = base64.b64encode(audio_output).decode()

    return {
        "original_text": original_text,
        "translated_text": translated_text,
        "detected_lang": detected_lang,
        "source": source,
        "target": target,
        "audio": audio_base64
    }

# ==================== ROOM AUDIO PROCESSING ====================
async def process_room_audio(room_code: str, is_host: bool, audio_base64: str, sender_name: str):
    """Process audio from room participant and send to partner"""
    room = rooms.get(room_code)
    if not room:
        return

    try:
        audio_bytes = base64.b64decode(audio_base64)
        logger.info(f"[Room {room_code}] {sender_name}: {len(audio_bytes)/1024:.1f} KB audio")

        # Process in executor to not block event loop
        loop = asyncio.get_event_loop()
        async with model_lock:
            result = await loop.run_in_executor(executor, process_audio_sync, audio_bytes)

        if not result:
            logger.info(f"[Room {room_code}] No speech detected")
            return

        logger.info(f"[Room {room_code}] {sender_name}: '{result['original_text'][:30]}...' -> '{result['translated_text'][:30]}...'")

        # Send transcript to BOTH users
        transcript_msg = {
            "type": "transcript",
            "from": sender_name,
            "original": result["original_text"],
            "translated": result["translated_text"],
            "lang_from": result["source"],
            "lang_to": result["target"],
            "timestamp": int(time.time() * 1000)
        }

        # Send to host
        try:
            await room.host_ws.send_json(transcript_msg)
        except:
            pass

        # Send to guest
        if room.guest_ws:
            try:
                await room.guest_ws.send_json(transcript_msg)
            except:
                pass

        # Send TTS audio to PARTNER only (not to sender)
        audio_msg = {
            "type": "audio_for_you",
            "data": result["audio"],
            "from": sender_name
        }

        if is_host and room.guest_ws:
            try:
                await room.guest_ws.send_json(audio_msg)
            except:
                pass
        elif not is_host:
            try:
                await room.host_ws.send_json(audio_msg)
            except:
                pass

    except Exception as e:
        logger.error(f"[Room {room_code}] Error processing audio: {e}")

# ==================== WEBSOCKET ENDPOINT ====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_room: Optional[str] = None
    is_host = False
    user_name: Optional[str] = None

    logger.info("WebSocket connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "create_room":
                # Host creates a new room - requires authentication
                token = data.get("token")
                if not token:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Login required to create room"
                    })
                    continue

                try:
                    host_username = validate_token_simple(token)
                except:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid token - please login again"
                    })
                    continue

                user_name = data.get("name", "Host")
                code = generate_room_code()
                rooms[code] = Room(
                    code=code,
                    host_ws=websocket,
                    host_name=user_name
                )
                current_room = code
                is_host = True
                logger.info(f"Room {code} created by {user_name} (account: {host_username})")
                await websocket.send_json({
                    "type": "room_created",
                    "code": code
                })

            elif msg_type == "join_room":
                # Guest joins existing room - NO authentication required
                code = data.get("code")
                user_name = data.get("name", "Guest")

                if code not in rooms:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Room does not exist"
                    })
                    continue

                room = rooms[code]
                if room.guest_ws is not None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Room is full"
                    })
                    continue

                # Join the room
                room.guest_ws = websocket
                room.guest_name = user_name
                current_room = code
                is_host = False
                logger.info(f"Room {code}: {user_name} joined")

                # Notify host
                await room.host_ws.send_json({
                    "type": "user_joined",
                    "name": user_name
                })

                # Notify guest
                await websocket.send_json({
                    "type": "room_ready",
                    "partner": room.host_name,
                    "code": code
                })

            elif msg_type == "audio":
                # Audio data from participant
                if current_room and current_room in rooms:
                    audio_data = data.get("data")
                    if audio_data:
                        # Process asynchronously
                        asyncio.create_task(
                            process_room_audio(current_room, is_host, audio_data, user_name)
                        )

            elif msg_type == "leave_room":
                # User leaves room
                if current_room and current_room in rooms:
                    room = rooms[current_room]
                    partner_ws = room.guest_ws if is_host else room.host_ws

                    # Notify partner
                    if partner_ws:
                        try:
                            await partner_ws.send_json({
                                "type": "user_left",
                                "name": user_name
                            })
                        except:
                            pass

                    # Clean up room
                    if is_host:
                        del rooms[current_room]
                        logger.info(f"Room {current_room} closed by host")
                    else:
                        room.guest_ws = None
                        room.guest_name = None
                        logger.info(f"Room {current_room}: guest left")

                    current_room = None

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {user_name}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup on disconnect
        if current_room and current_room in rooms:
            room = rooms[current_room]
            partner_ws = room.guest_ws if is_host else room.host_ws

            # Notify partner
            if partner_ws:
                try:
                    await partner_ws.send_json({
                        "type": "user_left",
                        "name": user_name or "Unknown"
                    })
                except:
                    pass

            # Clean up
            if is_host:
                del rooms[current_room]
                logger.info(f"Room {current_room} closed (host disconnected)")
            else:
                room.guest_ws = None
                room.guest_name = None
                logger.info(f"Room {current_room}: guest disconnected")

# ==================== REST ENDPOINTS ====================
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "ready",
        "service": "Thai-English Translator V3",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": whisper_model is not None,
        "active_rooms": len(rooms)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "models_loaded": whisper_model is not None}

@app.post("/translate", response_model=TranslateResponse)
async def translate_audio(request: TranslateRequest, authorization: str = Header(None)):
    """
    Translate audio from one language to another (Solo mode).
    Automatically detects source language (EN or TH) and translates to the other.
    Requires authentication token.
    """
    # Validate user token
    username = validate_token(authorization)

    total_start = time.time()
    timing = {}

    try:
        audio_bytes = base64.b64decode(request.audio)
        logger.info(f"[Solo] {username}: {len(audio_bytes) / 1024:.1f} KB audio")

        # Process with lock
        async with model_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, process_audio_sync, audio_bytes)

        if not result:
            raise HTTPException(status_code=400, detail="No speech detected")

        timing["total_ms"] = int((time.time() - total_start) * 1000)
        logger.info(f"[Solo] Done in {timing['total_ms']}ms")

        return TranslateResponse(
            original_text=result["original_text"],
            translated_text=result["translated_text"],
            detected_lang=result["detected_lang"],
            target_lang=result["target"],
            audio=result["audio"],
            timing=timing
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
