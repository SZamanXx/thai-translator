# Thai ↔ English Real-time Voice Translator

A bidirectional voice translator that runs **100% locally on a single GPU** — no Anthropic, no ElevenLabs, no Google, no per-call cost, no audio ever leaves the box. STT, MT, and TTS are all open-weight models served from one FastAPI process. The phone is just a microphone and a speaker; the GPU does the work.

Two modes in one backend:
- **Solo** — one person, one phone. Speak into your phone in either language, hear it back in the other. REST `POST /translate`.
- **Room** — two phones, two people, real conversation. Host opens a room (4-digit code + QR), guest scans, both speak naturally. Each side hears the *other* side's audio translated into their own language. WebSocket `/ws`.

The Room mode is the reason I built this. Solo voice translation is a solved problem on every phone in 2026 — but pointing a translator app at yourself and showing your screen to a stranger across a table is not a conversation, it's a checkpoint. Two-phone room mode is what an actual conversation between two languages feels like.

---

## Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │      Backend (Python, FastAPI, port 8001)    │
                  │                                              │
   Phone A        │   Whisper Large-v3   ─ STT                   │
  (English)       │       │                                      │
      ──── ngrok ─┼─→ NLLB-200-distilled  ─ MT (EN ↔ TH)         │
                  │       │                                      │
   Phone B        │   MMS-TTS-tha / -eng  ─ TTS                  │
   (Thai)         │       │                                      │
      ──── ngrok ─┼─→ Auth + Room manager (in-memory + SQLite)   │
                  │                                              │
                  │   All four models live on GPU                │
                  │   one ThreadPoolExecutor, one model lock     │
                  └──────────────────────────────────────────────┘

Solo:    Phone  →  POST /translate  (base64 WAV in, base64 WAV out)
Room:    Phone  ⇄  WS   /ws         (audio + transcript fan-out)
```

---

## Requirements

**Backend**
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (~4 GB VRAM at fp16 with all four models resident)
- Tested on RTX 5090 Laptop. Works on smaller GPUs with `compute_type="int8"` on Whisper and matching dtype downgrades on NLLB / MMS, with a real-time penalty.
- ~5 GB free disk for model weights (Hugging Face cache)

**Mobile**
- Node.js 18+
- Expo Go on iOS or Android
- An HTTPS tunnel to your backend — ngrok, cloudflared, tailscale-funnel, your choice

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python main.py
```

First boot downloads the four models from Hugging Face (~5 GB) and loads them onto the GPU. Subsequent boots are ~25-30 s of model load. Server binds on `0.0.0.0:8001`.

You can also double-click `start_backend.bat` on Windows.

### 2. Expose the backend

```bash
ngrok http 8001
```

Copy the HTTPS URL (e.g. `https://abc123.ngrok-free.app`).

### 3. Mobile

Open `mobile/App.js`, set `BACKEND_URL` at the top of the file to your ngrok URL. Then:

```bash
cd mobile
npm install
npx expo start --tunnel
```

Scan the QR code with Expo Go.

### 4. Use it

- **Solo**: register → login → tap *Solo* → speak. VAD detects end-of-utterance and sends; you hear the translation. Loop.
- **Room (host)**: register → login → tap *Create room* → screen shows a 4-digit code and a QR.
- **Room (guest)**: open Expo Go on the second phone, scan the QR (or tap *Join room* and enter the code). No account needed for the guest.

---

## Models

| Task         | Model                          | Size   | License                          |
| ---          | ---                            | ---    | ---                              |
| STT          | `faster-whisper` Large-v3      | ~3 GB  | MIT (Whisper) / MIT (CT2)        |
| Translation  | `facebook/nllb-200-distilled-600M` | ~1.2 GB | CC-BY-NC-4.0                  |
| TTS Thai     | `facebook/mms-tts-tha`         | ~140 MB | CC-BY-NC-4.0                    |
| TTS English  | `facebook/mms-tts-eng`         | ~140 MB | CC-BY-NC-4.0                    |

A note on the NLLB / MMS licenses: they are **CC-BY-NC** (non-commercial). For production / commercial use, swap NLLB for a permissively-licensed MT model (MADLAD-400, M2M-100 100M, or a fine-tuned NLLB checkpoint released under a different license) and MMS for VITS / Piper / Coqui XTTS. The pipeline shape is the same; only the `from_pretrained()` ids change.

---

## Benchmark (RTX 5090 Laptop, fp16)

Real numbers from `test_voice_benchmark.py` on a ~3-second utterance, measured end-to-end on the GPU side (no network):

| Stage         | EN → TH   | TH → EN   |
| ---           | ---       | ---       |
| Whisper STT   | 2 771 ms  | 1 126 ms  |
| NLLB MT       | 518 ms    | 127 ms    |
| MMS TTS       | 491 ms    | 377 ms    |
| **Total**     | **3.78 s** | **1.63 s** |

Cold-start (load all four models from HF cache): ~26 s. After that, the next inference is hot.

The asymmetry is real and worth knowing — English audio is harder for Whisper than Thai audio in this configuration, mostly because the test utterance was longer in English. Both directions are well below the threshold where the conversation feels like a translator-in-the-middle, but EN → TH is the one you would tune first.

---

## API

All routes are on the backend (`:8001`).

| Method | Path        | Purpose                                                  | Auth         |
| ---    | ---         | ---                                                      | ---          |
| GET    | `/`         | Service info + GPU + active room count                   | none         |
| GET    | `/health`   | Liveness                                                 | none         |
| POST   | `/register` | Create account (`username`, `password`) → returns token  | none         |
| POST   | `/login`    | Returns a new bearer token                               | none         |
| GET    | `/validate` | Confirms a token is valid                                | bearer       |
| POST   | `/translate`| Solo mode — base64 WAV in, full result + base64 WAV out  | bearer       |
| WS     | `/ws`       | Room mode — `create_room` / `join_room` / `audio` / `leave_room` / `ping` | per-message |

Auth model:
- Passwords are SHA-256 hashed (this is **fine for a demo, not for production** — bcrypt/argon2 is the real answer; see "What I cut").
- A new bearer token is issued on every login. Tokens are not expiring.
- The `users.json` file holds the hashed-password + token store. It is `.gitignore`d so your local users do not leak in a fork.
- **Creating a room** requires a token. **Joining a room** does not — guests are anonymous on purpose, so you can hand someone the QR code without onboarding them.

WebSocket message shapes are documented inline in `backend/main.py` (search for `msg_type ==`). Short version:

```json
// → host
{"type": "create_room", "token": "...", "name": "Wojciech"}
// ← server
{"type": "room_created", "code": "1234"}

// → guest
{"type": "join_room", "code": "1234", "name": "Sapir"}
// ← both
{"type": "room_ready", "partner": "Wojciech", "code": "1234"}

// → either side, repeatedly during the call
{"type": "audio", "data": "<base64 WAV>"}
// ← both sides receive transcript + translated transcript
{"type": "transcript", "from": "Wojciech", "original": "...", "translated": "...", "lang_from": "en", "lang_to": "th", "timestamp": 1737200000000}
// ← only the *partner* of the sender receives this:
{"type": "audio_for_you", "data": "<base64 WAV>", "from": "Wojciech"}
```

---

## Key decisions and tradeoffs

**1. Everything on one GPU, in one process, behind one model lock.**

GPU memory is the bottleneck, not throughput at small scale. All four models are loaded once, kept resident in VRAM at fp16, and served from a single `ThreadPoolExecutor(max_workers=1)` guarded by an `asyncio.Lock`. This serializes requests intentionally — two simultaneous Whisper transcriptions on the same GPU is slower wall-clock than back-to-back, and contention on VRAM-resident weights is not worth the complexity for a 2-person conversation. If you scale this to N rooms, you either run N backends behind a load balancer, or you split into separate STT / MT / TTS services and put a queue in front of each.

**2. Whisper Large-v3 instead of `medium` or `turbo`.**

The honest reason is that Thai is a tonal language and `medium` makes a noticeable share of mistakes that `large-v3` does not, particularly on short utterances and on accented English from non-native speakers (a common case for this exact app). The Whisper cost in the table is the single largest stage of the pipeline — that is also the stage where downgrading the model is most visible to the user. I keep large-v3 and live with 2.7 s on EN inputs rather than ship a fast translator that mistranscribes.

**3. NLLB-200 distilled-600M, not the 1.3B or 3.3B variants.**

The 600M variant is plenty for everyday EN ↔ TH at conversational length. The bigger NLLB checkpoints help on long technical text or rare language pairs; neither is the case here.

**4. MMS-TTS for both directions, not Coqui XTTS or ElevenLabs.**

XTTS is multilingual but does not natively support Thai. ElevenLabs would solve quality but it is a paid API and the whole point of this project is offline. MMS is single-speaker, no cloning, robotic on long utterances, but it covers both EN and TH locally with the same `VitsModel` interface — and the audio is good enough that you understand the speaker the first time. If TTS quality is the part you want to improve, swap MMS for Piper (English) + a fine-tuned VAJA / Thai-specific VITS checkpoint (Thai), and accept the per-language model split.

**5. Solo over REST, Room over WebSocket.**

Solo mode is request/response — record, send, get audio back. REST is the simplest protocol for that and it survives reconnects with zero state. Room mode is bidirectional and fan-out (two clients, each receives both transcripts and the other side's audio), so it has to be a persistent connection. They share one `process_audio_sync` core function; only the transport differs.

**6. SHA-256 password hashing instead of bcrypt/argon2.**

This is a demo-grade choice and I want to be honest about it. SHA-256 is fast — which is exactly the problem for password storage — and there is no salt in this implementation. For anything past a demo, swap `hash_password()` in `backend/main.py` to `passlib.hash.argon2.hash()` / `verify()` and re-issue tokens on first login.

**7. No HTTPS termination in-process.**

Backend binds plain HTTP on `0.0.0.0:8001`. HTTPS is the tunnel's job — ngrok / cloudflared / nginx in front. This keeps the server uvicorn-only and removes one moving part (cert management) from anything you might want to run quickly on a laptop.

---

## Project layout

```
thai-translator/
├── backend/
│   ├── main.py              # FastAPI app — auth, /translate, /ws, room manager
│   ├── requirements.txt     # CUDA-pinned PyTorch + faster-whisper + transformers
│   └── users.json           # runtime credential store (gitignored, created on first /register)
├── mobile/
│   ├── App.js               # Expo app — auth, mode picker, Solo, Room, QR
│   ├── app.json
│   ├── babel.config.js
│   ├── index.js
│   ├── package.json
│   └── package-lock.json
├── test_e2e.py              # end-to-end pipeline test (ElevenLabs ref audio → local stack)
├── test_realtime.py         # realtime / streaming smoke test
├── test_voice_benchmark.py  # produced the numbers in this README
├── start_backend.bat
├── start_mobile.bat
├── .gitignore
└── README.md
```

To run the tests, set `ELEVENLABS_API_KEY` in your environment first — the test scripts use ElevenLabs *only* to generate clean reference audio; the runtime app does not.

---

## What I cut

- **bcrypt / argon2 password hashing.** SHA-256 is in. Demo-grade. See note above.
- **HTTPS / TLS in the backend.** Tunnel handles it.
- **Per-room model isolation.** All rooms share one GPU lock, so two rooms speaking at the same time will serialize. Fine for the 1-2 room scale this is built for.
- **Streaming STT.** Whisper transcribes whole utterances post-VAD. Streaming Whisper exists (`whisper-streaming`, `faster-whisper` server mode) but adds real complexity and the VAD cut-point pattern is good enough that nobody using it felt the latency as a problem.
- **Voice cloning on TTS.** MMS is single-speaker. Sometimes I want my voice back; today, no.
- **Rate limiting** on `/register` and `/login`. The threat model for a self-hosted local app is small; for anything public, add slowapi or nginx limits.

## What I would add with more time

1. **Streaming TTS playback** — emit the first second of TTS audio while the rest is still being generated. MMS-TTS supports this with a small refactor; the win is perceived-latency, not throughput.
2. **A second STT pass on the recording** — the same dual-transcript pattern I used in ARIA. Whisper + a smaller domain-specific model voting on disagreements, with low-confidence words flagged in the transcript view. Useful when one side is a non-native speaker of the target language.
3. **Per-room language pinning.** Right now the source language is auto-detected per utterance via Whisper. That works fine in clean conditions but produces occasional swaps when a Thai speaker code-switches a single English word. Letting the room *declare* the two languages and force STT into one or the other would fix that.
4. **A proper account/credential rewrite.** argon2 hashing, refresh + access tokens with expiry, optional 2FA. Or — just drop auth entirely for the local self-hosted use case, which is honestly more honest about what the deployment story is.
5. **Multi-language room.** Right now it is rigidly EN ↔ TH. NLLB-200 supports 200 languages; adding a per-side language picker on the client would let the same backend handle TH ↔ JA, EN ↔ PL, anything.

---

## How I used AI tools

- **Claude Code (CLI)** wrote the first scaffold of the FastAPI app, the WebSocket room manager, and the Expo screens. I edited every diff. Architecture decisions (one GPU lock, REST vs WS split, MMS for both directions) are mine, decided before any code.
- **No fine-tuning, no custom weights, no training.** Every model in `backend/main.py` is a stock Hugging Face checkpoint loaded as-is.

---

## Author

Wojciech Szymański — built solo, on my own machine, because I wanted a translator that did not phone home.
GitHub: [@SZamanXx](https://github.com/SZamanXx)
