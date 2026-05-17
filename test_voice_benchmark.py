"""
Voice Benchmark Test - Thai-English Translator
===============================================
Measures ONLY inference times (models already cached).
Flow: ElevenLabs Audio -> Whisper STT -> NLLB Translation -> MMS TTS
"""

import os
import time
import tempfile
import requests
import io

import torch
import numpy as np
import scipy.io.wavfile as wavfile

# ElevenLabs is used here ONLY to synthesize reference speech for benchmarking
# the local pipeline (Whisper STT -> NLLB MT -> MMS TTS). The runtime app does
# not use ElevenLabs. Set ELEVENLABS_API_KEY in your environment.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "bIHbv24MWmeRgasZH58o"  # Will - multilingual

# Test phrases
TEST_EN = "Hello, my name is Michael. Nice to meet you. How are you today?"
TEST_TH = "สวัสดีครับ ผมชื่อไมเคิล ยินดีที่ได้รู้จักครับ"

print("=" * 70)
print("VOICE BENCHMARK - INFERENCE TIMES ONLY")
print("=" * 70)

# ===================== CUDA INFO =====================
print(f"\nCUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ===================== LOAD MODELS (from cache) =====================
print("\n" + "=" * 70)
print("PHASE 1: LOADING MODELS FROM CACHE")
print("=" * 70)

load_times = {}

# Whisper
print("\n[1/4] Loading Whisper Large-v3...")
t0 = time.time()
from faster_whisper import WhisperModel
whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
load_times["whisper"] = time.time() - t0
print(f"      Done: {load_times['whisper']:.2f}s")

# NLLB
print("\n[2/4] Loading NLLB-200...")
t0 = time.time()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    torch_dtype=torch.float16,
    device_map="cuda"
)
load_times["nllb"] = time.time() - t0
print(f"      Done: {load_times['nllb']:.2f}s")

# MMS-TTS Thai
print("\n[3/4] Loading MMS-TTS Thai...")
t0 = time.time()
from transformers import VitsModel, AutoProcessor
tts_th_proc = AutoProcessor.from_pretrained("facebook/mms-tts-tha")
tts_th_model = VitsModel.from_pretrained("facebook/mms-tts-tha", torch_dtype=torch.float16).to("cuda")
load_times["mms_thai"] = time.time() - t0
print(f"      Done: {load_times['mms_thai']:.2f}s")

# MMS-TTS English
print("\n[4/4] Loading MMS-TTS English...")
t0 = time.time()
tts_en_proc = AutoProcessor.from_pretrained("facebook/mms-tts-eng")
tts_en_model = VitsModel.from_pretrained("facebook/mms-tts-eng", torch_dtype=torch.float16).to("cuda")
load_times["mms_eng"] = time.time() - t0
print(f"      Done: {load_times['mms_eng']:.2f}s")

print(f"\n>>> TOTAL MODEL LOAD TIME: {sum(load_times.values()):.2f}s")

# GPU warmup
torch.cuda.synchronize()

# ===================== FUNCTIONS =====================
LANG_CODES = {"en": "eng_Latn", "th": "tha_Thai"}

def elevenlabs_tts(text):
    """Generate audio via ElevenLabs API"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    data = {"text": text, "model_id": "eleven_v3", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}

    t0 = time.time()
    resp = requests.post(url, headers=headers, json=data)
    elapsed = time.time() - t0

    if resp.status_code == 200:
        return resp.content, elapsed
    else:
        raise Exception(f"ElevenLabs error: {resp.status_code}")

def whisper_stt(audio_bytes):
    """Transcribe audio with Whisper - returns text, lang, time"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    torch.cuda.synchronize()
    t0 = time.time()
    segments, info = whisper.transcribe(temp_path, beam_size=5)
    text = " ".join([s.text for s in segments])
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    os.unlink(temp_path)
    return text.strip(), info.language, elapsed

def nllb_translate(text, src, tgt):
    """Translate with NLLB - returns translated text, time"""
    nllb_tokenizer.src_lang = LANG_CODES[src]
    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        gen = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(LANG_CODES[tgt]),
            max_length=256
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    translated = nllb_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translated, elapsed

def mms_tts(text, lang):
    """Generate speech with MMS-TTS - returns waveform, duration, time"""
    proc = tts_th_proc if lang == "th" else tts_en_proc
    model = tts_th_model if lang == "th" else tts_en_model

    inputs = proc(text=text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = model(**inputs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    waveform = output.waveform[0].cpu().numpy()
    duration = len(waveform) / model.config.sampling_rate
    return waveform, duration, elapsed

# ===================== GENERATE AUDIO FROM ELEVENLABS =====================
print("\n" + "=" * 70)
print("PHASE 2: GENERATING TEST AUDIO (ElevenLabs)")
print("=" * 70)

print(f"\n[EN] '{TEST_EN[:50]}...'")
audio_en, el_time_en = elevenlabs_tts(TEST_EN)
print(f"     Generated {len(audio_en)/1024:.1f} KB in {el_time_en:.2f}s")

print(f"\n[TH] '{TEST_TH}'")
audio_th, el_time_th = elevenlabs_tts(TEST_TH)
print(f"     Generated {len(audio_th)/1024:.1f} KB in {el_time_th:.2f}s")

# ===================== BENCHMARK: ENGLISH -> THAI =====================
print("\n" + "=" * 70)
print("BENCHMARK 1: ENGLISH -> THAI")
print("=" * 70)

results_en_th = {}

# STT
print("\n[STT] Whisper transcribing English audio...")
text_en, lang_en, stt_time = whisper_stt(audio_en)
results_en_th["stt"] = stt_time
print(f"      Detected: {lang_en}")
print(f"      Text: '{text_en[:60]}...'")
print(f"      >>> TIME: {stt_time*1000:.0f}ms")

# Translation
print("\n[TRANSLATE] NLLB: English -> Thai...")
translated_th, trans_time = nllb_translate(text_en, "en", "th")
results_en_th["translate"] = trans_time
print(f"      Output: '{translated_th[:60]}...'")
print(f"      >>> TIME: {trans_time*1000:.0f}ms")

# TTS
print("\n[TTS] MMS-TTS generating Thai speech...")
wav_th, dur_th, tts_time = mms_tts(translated_th, "th")
results_en_th["tts"] = tts_time
print(f"      Audio: {dur_th:.2f}s")
print(f"      >>> TIME: {tts_time*1000:.0f}ms")

results_en_th["total"] = sum(results_en_th.values())

# ===================== BENCHMARK: THAI -> ENGLISH =====================
print("\n" + "=" * 70)
print("BENCHMARK 2: THAI -> ENGLISH")
print("=" * 70)

results_th_en = {}

# STT
print("\n[STT] Whisper transcribing Thai audio...")
text_th, lang_th, stt_time = whisper_stt(audio_th)
results_th_en["stt"] = stt_time
print(f"      Detected: {lang_th}")
print(f"      Text: '{text_th}'")
print(f"      >>> TIME: {stt_time*1000:.0f}ms")

# Translation
print("\n[TRANSLATE] NLLB: Thai -> English...")
translated_en, trans_time = nllb_translate(text_th, "th", "en")
results_th_en["translate"] = trans_time
print(f"      Output: '{translated_en}'")
print(f"      >>> TIME: {trans_time*1000:.0f}ms")

# TTS
print("\n[TTS] MMS-TTS generating English speech...")
wav_en, dur_en, tts_time = mms_tts(translated_en, "en")
results_th_en["tts"] = tts_time
print(f"      Audio: {dur_en:.2f}s")
print(f"      >>> TIME: {tts_time*1000:.0f}ms")

results_th_en["total"] = sum(results_th_en.values())

# ===================== FINAL REPORT =====================
print("\n" + "=" * 70)
print("FINAL REPORT - INFERENCE TIMES ONLY")
print("=" * 70)

print("\n┌────────────────────────────────────────────────────────────────┐")
print("│                    INFERENCE TIMES (ms)                        │")
print("├──────────────────┬──────────────────┬──────────────────────────┤")
print("│ Component        │ EN -> TH         │ TH -> EN                 │")
print("├──────────────────┼──────────────────┼──────────────────────────┤")
print(f"│ Whisper STT      │ {results_en_th['stt']*1000:>10.0f} ms    │ {results_th_en['stt']*1000:>10.0f} ms            │")
print(f"│ NLLB Translation │ {results_en_th['translate']*1000:>10.0f} ms    │ {results_th_en['translate']*1000:>10.0f} ms            │")
print(f"│ MMS TTS          │ {results_en_th['tts']*1000:>10.0f} ms    │ {results_th_en['tts']*1000:>10.0f} ms            │")
print("├──────────────────┼──────────────────┼──────────────────────────┤")
print(f"│ TOTAL            │ {results_en_th['total']*1000:>10.0f} ms    │ {results_th_en['total']*1000:>10.0f} ms            │")
print("└──────────────────┴──────────────────┴──────────────────────────┘")

avg_total = (results_en_th["total"] + results_th_en["total"]) / 2
print(f"\n>>> AVERAGE TOTAL INFERENCE: {avg_total*1000:.0f}ms ({avg_total:.3f}s)")
print(f">>> This is the latency user will experience (excluding network)")

# Save report
report_path = "D:/python/thai-translator/reports/07_voice_benchmark.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Voice Benchmark Report\n\n")
    f.write(f"**GPU:** {torch.cuda.get_device_name(0)}\n\n")
    f.write("## Model Load Times (from cache)\n\n")
    f.write("| Model | Time |\n|-------|------|\n")
    for k, v in load_times.items():
        f.write(f"| {k} | {v:.2f}s |\n")
    f.write(f"| **TOTAL** | **{sum(load_times.values()):.2f}s** |\n\n")

    f.write("## Inference Times\n\n")
    f.write("| Component | EN -> TH | TH -> EN |\n")
    f.write("|-----------|----------|----------|\n")
    f.write(f"| Whisper STT | {results_en_th['stt']*1000:.0f}ms | {results_th_en['stt']*1000:.0f}ms |\n")
    f.write(f"| NLLB Translation | {results_en_th['translate']*1000:.0f}ms | {results_th_en['translate']*1000:.0f}ms |\n")
    f.write(f"| MMS TTS | {results_en_th['tts']*1000:.0f}ms | {results_th_en['tts']*1000:.0f}ms |\n")
    f.write(f"| **TOTAL** | **{results_en_th['total']*1000:.0f}ms** | **{results_th_en['total']*1000:.0f}ms** |\n\n")

    f.write(f"## Summary\n\n")
    f.write(f"- **Average inference time:** {avg_total*1000:.0f}ms\n")
    f.write(f"- **Real-time factor:** {avg_total:.3f}s processing for ~3s speech\n")

print(f"\nReport saved: {report_path}")
print("\n" + "=" * 70)
