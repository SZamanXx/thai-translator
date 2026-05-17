"""
Real-time Performance Test - Thai-English Translator
=====================================================
Preloads all models, then waits for user input to measure actual inference times.
"""

import os
import sys
import time
import io
import tempfile
from pathlib import Path

import torch
import numpy as np

print("=" * 60)
print("LOADING MODELS (one-time startup)")
print("=" * 60)

# Check CUDA
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== LOAD WHISPER ====================
print("\n[1/4] Loading Whisper Large-v3...")
start = time.time()
from faster_whisper import WhisperModel
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"      Loaded in {time.time() - start:.2f}s")

# ==================== LOAD NLLB ====================
print("\n[2/4] Loading NLLB-200 Translation...")
start = time.time()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
nllb_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    nllb_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)
print(f"      Loaded in {time.time() - start:.2f}s")

# ==================== LOAD MMS-TTS THAI ====================
print("\n[3/4] Loading MMS-TTS Thai...")
start = time.time()
from transformers import VitsModel, AutoProcessor
tts_thai_processor = AutoProcessor.from_pretrained("facebook/mms-tts-tha")
tts_thai_model = VitsModel.from_pretrained("facebook/mms-tts-tha", torch_dtype=torch.float16).to("cuda")
print(f"      Loaded in {time.time() - start:.2f}s")

# ==================== LOAD MMS-TTS ENGLISH ====================
print("\n[4/4] Loading MMS-TTS English...")
start = time.time()
tts_eng_processor = AutoProcessor.from_pretrained("facebook/mms-tts-eng")
tts_eng_model = VitsModel.from_pretrained("facebook/mms-tts-eng", torch_dtype=torch.float16).to("cuda")
print(f"      Loaded in {time.time() - start:.2f}s")

# Warmup GPU
print("\n[*] Warming up GPU...")
with torch.no_grad():
    dummy = torch.randn(1, 100).to("cuda")
    _ = dummy * 2
torch.cuda.synchronize()
print("    Done!")

print("\n" + "=" * 60)
print("ALL MODELS LOADED - READY FOR TESTING")
print("=" * 60)

# Language codes
LANG_CODES = {
    "en": "eng_Latn",
    "th": "tha_Thai"
}

def translate(text, source_lang, target_lang):
    """Translate text using NLLB"""
    start = time.time()

    nllb_tokenizer.src_lang = LANG_CODES[source_lang]
    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        generated = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(LANG_CODES[target_lang]),
            max_length=256
        )

    translated = nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    elapsed = time.time() - start

    return translated, elapsed

def text_to_speech(text, language):
    """Generate speech using MMS-TTS"""
    start = time.time()

    if language == "th":
        processor = tts_thai_processor
        model = tts_thai_model
    else:
        processor = tts_eng_processor
        model = tts_eng_model

    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform[0].cpu().numpy()
    sample_rate = model.config.sampling_rate
    duration = len(waveform) / sample_rate
    elapsed = time.time() - start

    return waveform, sample_rate, duration, elapsed

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    start = time.time()

    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    text = " ".join([s.text for s in segments])

    elapsed = time.time() - start
    return text, info.language, elapsed

def test_translation_flow(text, source_lang, target_lang):
    """Test full translation + TTS flow"""
    print(f"\n{'─' * 50}")
    print(f"INPUT ({source_lang.upper()}): {text}")
    print(f"{'─' * 50}")

    # Translation
    translated, trans_time = translate(text, source_lang, target_lang)
    print(f"[Translation] {trans_time*1000:.0f}ms")
    print(f"OUTPUT ({target_lang.upper()}): {translated}")

    # TTS
    waveform, sr, duration, tts_time = text_to_speech(translated, target_lang)
    print(f"[TTS] {tts_time*1000:.0f}ms → {duration:.2f}s audio")

    total = trans_time + tts_time
    print(f"{'─' * 50}")
    print(f"TOTAL: {total*1000:.0f}ms ({total:.3f}s)")

    return translated, total

# ==================== INTERACTIVE LOOP ====================
print("\n" + "=" * 60)
print("INTERACTIVE TEST MODE")
print("=" * 60)
print("Commands:")
print("  - Type English text → translates to Thai + TTS")
print("  - Type 'th:' + Thai text → translates to English + TTS")
print("  - Type 'quit' to exit")
print("=" * 60)

while True:
    try:
        user_input = input("\n> ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.startswith("th:"):
            # Thai to English
            thai_text = user_input[3:].strip()
            test_translation_flow(thai_text, "th", "en")
        else:
            # English to Thai
            test_translation_flow(user_input, "en", "th")

    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
