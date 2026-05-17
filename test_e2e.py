"""
End-to-End Test for Thai-English Translator
============================================
Tests: ElevenLabs TTS -> Whisper STT -> NLLB Translation -> MMS TTS
"""

import os
import sys
import time
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import requests
import numpy as np

# ElevenLabs is used here ONLY to generate clean reference audio for the
# pipeline test (Whisper STT -> NLLB MT -> MMS TTS). The runtime app does
# not use ElevenLabs. Set ELEVENLABS_API_KEY in your environment.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Test phrases
TEST_PHRASES = {
    "thai": "สวัสดีครับ ผมชื่อไมเคิล ยินดีที่ได้รู้จักครับ",
    "english": "Hello, my name is Michael. Nice to meet you. How are you today?"
}

# Results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "summary": {}
}

def log(msg):
    """Print with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def generate_audio_elevenlabs(text, language="en"):
    """Generate audio using ElevenLabs API"""
    log(f"Generating audio for: '{text[:50]}...' ({language})")

    start = time.time()

    url = "https://api.elevenlabs.io/v1/text-to-speech/bIHbv24MWmeRgasZH58o"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data)
    elapsed = time.time() - start

    if response.status_code == 200:
        log(f"  ElevenLabs TTS: {elapsed:.2f}s")
        return response.content, elapsed
    else:
        log(f"  ERROR: {response.status_code} - {response.text}")
        return None, elapsed

def test_whisper_stt(audio_bytes, expected_lang):
    """Test Whisper STT"""
    log("Testing Whisper STT...")

    try:
        from faster_whisper import WhisperModel

        start = time.time()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        # Load model (use small for faster testing)
        model_start = time.time()
        model = WhisperModel("small", device="cuda", compute_type="float16")
        model_load_time = time.time() - model_start
        log(f"  Model load: {model_load_time:.2f}s")

        # Transcribe
        transcribe_start = time.time()
        segments, info = model.transcribe(temp_path, beam_size=5)
        text = " ".join([s.text for s in segments])
        transcribe_time = time.time() - transcribe_start

        # Cleanup
        os.unlink(temp_path)

        elapsed = time.time() - start

        log(f"  Detected language: {info.language}")
        log(f"  Transcription: '{text[:100]}...'")
        log(f"  Transcribe time: {transcribe_time:.2f}s")
        log(f"  Total STT time: {elapsed:.2f}s")

        return {
            "success": True,
            "text": text,
            "detected_lang": info.language,
            "model_load_time": model_load_time,
            "transcribe_time": transcribe_time,
            "total_time": elapsed
        }

    except Exception as e:
        log(f"  ERROR: {e}")
        return {"success": False, "error": str(e)}

def test_nllb_translation(text, source_lang, target_lang):
    """Test NLLB Translation"""
    log(f"Testing NLLB Translation ({source_lang} -> {target_lang})...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        start = time.time()

        # Language codes
        lang_codes = {
            "en": "eng_Latn",
            "th": "tha_Thai"
        }

        # Load model
        model_start = time.time()
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        model_load_time = time.time() - model_start
        log(f"  Model load: {model_load_time:.2f}s")

        # Translate
        translate_start = time.time()
        tokenizer.src_lang = lang_codes[source_lang]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_codes[target_lang]),
                max_length=256
            )

        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        translate_time = time.time() - translate_start

        elapsed = time.time() - start

        log(f"  Input: '{text[:80]}...'")
        log(f"  Output: '{translated[:80]}...'")
        log(f"  Translate time: {translate_time:.2f}s")
        log(f"  Total time: {elapsed:.2f}s")

        return {
            "success": True,
            "input": text,
            "output": translated,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model_load_time": model_load_time,
            "translate_time": translate_time,
            "total_time": elapsed
        }

    except Exception as e:
        log(f"  ERROR: {e}")
        return {"success": False, "error": str(e)}

def test_mms_tts(text, language):
    """Test MMS TTS"""
    log(f"Testing MMS TTS ({language})...")

    try:
        import torch
        from transformers import VitsModel, AutoProcessor
        import scipy.io.wavfile as wavfile
        import io

        start = time.time()

        # Model name
        model_name = f"facebook/mms-tts-{'tha' if language == 'th' else 'eng'}"

        # Load model
        model_start = time.time()
        processor = AutoProcessor.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        model_load_time = time.time() - model_start
        log(f"  Model load: {model_load_time:.2f}s")

        # Generate speech
        generate_start = time.time()
        inputs = processor(text=text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        waveform = output.waveform[0].cpu().numpy()
        sample_rate = model.config.sampling_rate
        generate_time = time.time() - generate_start

        # Save to bytes
        waveform = waveform / np.max(np.abs(waveform)) * 0.9
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, (waveform * 32767).astype(np.int16))
        audio_bytes = buffer.getvalue()

        elapsed = time.time() - start

        log(f"  Generated {len(audio_bytes)} bytes")
        log(f"  Sample rate: {sample_rate} Hz")
        log(f"  Duration: {len(waveform) / sample_rate:.2f}s")
        log(f"  Generate time: {generate_time:.2f}s")
        log(f"  Total time: {elapsed:.2f}s")

        return {
            "success": True,
            "audio_size": len(audio_bytes),
            "sample_rate": sample_rate,
            "duration": len(waveform) / sample_rate,
            "model_load_time": model_load_time,
            "generate_time": generate_time,
            "total_time": elapsed
        }

    except Exception as e:
        log(f"  ERROR: {e}")
        return {"success": False, "error": str(e)}

def run_full_flow_test(source_text, source_lang, target_lang):
    """Run full translation flow test"""
    log(f"\n{'='*60}")
    log(f"FULL FLOW TEST: {source_lang.upper()} -> {target_lang.upper()}")
    log(f"{'='*60}")
    log(f"Input text: '{source_text}'")

    flow_start = time.time()
    flow_results = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_text": source_text,
        "steps": {}
    }

    # Step 1: Generate audio with ElevenLabs
    log("\n--- Step 1: Generate source audio (ElevenLabs) ---")
    audio_bytes, tts_time = generate_audio_elevenlabs(source_text, source_lang)
    flow_results["steps"]["elevenlabs_tts"] = {
        "success": audio_bytes is not None,
        "time": tts_time,
        "audio_size": len(audio_bytes) if audio_bytes else 0
    }

    if not audio_bytes:
        log("FAILED at Step 1")
        flow_results["success"] = False
        return flow_results

    # Step 2: Transcribe with Whisper
    log("\n--- Step 2: Transcribe (Whisper) ---")
    stt_result = test_whisper_stt(audio_bytes, source_lang)
    flow_results["steps"]["whisper_stt"] = stt_result

    if not stt_result.get("success"):
        log("FAILED at Step 2")
        flow_results["success"] = False
        return flow_results

    transcribed_text = stt_result["text"]

    # Step 3: Translate with NLLB
    log("\n--- Step 3: Translate (NLLB) ---")
    translate_result = test_nllb_translation(transcribed_text, source_lang, target_lang)
    flow_results["steps"]["nllb_translation"] = translate_result

    if not translate_result.get("success"):
        log("FAILED at Step 3")
        flow_results["success"] = False
        return flow_results

    translated_text = translate_result["output"]

    # Step 4: Generate target audio with MMS-TTS
    log("\n--- Step 4: Generate target audio (MMS-TTS) ---")
    tts_result = test_mms_tts(translated_text, target_lang)
    flow_results["steps"]["mms_tts"] = tts_result

    # Summary
    flow_time = time.time() - flow_start
    flow_results["total_time"] = flow_time
    flow_results["success"] = all(
        step.get("success", False)
        for step in flow_results["steps"].values()
    )

    log(f"\n{'='*60}")
    log(f"FLOW COMPLETE: {'SUCCESS' if flow_results['success'] else 'FAILED'}")
    log(f"Total time: {flow_time:.2f}s")
    log(f"{'='*60}")

    return flow_results

def main():
    log("="*60)
    log("THAI-ENGLISH TRANSLATOR - END-TO-END TEST")
    log("="*60)

    # Check CUDA
    import torch
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test 1: English -> Thai
    log("\n" + "="*60)
    log("TEST 1: ENGLISH -> THAI")
    log("="*60)
    result_en_th = run_full_flow_test(
        TEST_PHRASES["english"],
        "en",
        "th"
    )
    results["tests"].append(result_en_th)

    # Test 2: Thai -> English
    log("\n" + "="*60)
    log("TEST 2: THAI -> ENGLISH")
    log("="*60)
    result_th_en = run_full_flow_test(
        TEST_PHRASES["thai"],
        "th",
        "en"
    )
    results["tests"].append(result_th_en)

    # Summary
    log("\n" + "="*60)
    log("FINAL SUMMARY")
    log("="*60)

    results["summary"] = {
        "total_tests": len(results["tests"]),
        "passed": sum(1 for t in results["tests"] if t.get("success")),
        "failed": sum(1 for t in results["tests"] if not t.get("success")),
        "avg_time": sum(t.get("total_time", 0) for t in results["tests"]) / len(results["tests"])
    }

    log(f"Tests passed: {results['summary']['passed']}/{results['summary']['total_tests']}")
    log(f"Average flow time: {results['summary']['avg_time']:.2f}s")

    # Time breakdown
    log("\nTime breakdown (per step, first model load):")
    for test in results["tests"]:
        log(f"\n  {test['source_lang'].upper()} -> {test['target_lang'].upper()}:")
        for step_name, step_data in test.get("steps", {}).items():
            if isinstance(step_data, dict) and "time" in step_data:
                log(f"    {step_name}: {step_data['time']:.2f}s")
            elif isinstance(step_data, dict) and "total_time" in step_data:
                log(f"    {step_name}: {step_data['total_time']:.2f}s")

    # Save report
    report_path = Path(__file__).parent / "reports" / "06_e2e_test.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# End-to-End Test Report\n\n")
        f.write(f"**Date:** {results['timestamp']}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Tests passed: {results['summary']['passed']}/{results['summary']['total_tests']}\n")
        f.write(f"- Average flow time: {results['summary']['avg_time']:.2f}s\n\n")

        f.write("## Test Results\n\n")
        for i, test in enumerate(results["tests"], 1):
            f.write(f"### Test {i}: {test['source_lang'].upper()} -> {test['target_lang'].upper()}\n\n")
            f.write(f"- **Status:** {'✅ PASS' if test.get('success') else '❌ FAIL'}\n")
            f.write(f"- **Total time:** {test.get('total_time', 0):.2f}s\n")
            f.write(f"- **Source text:** {test.get('source_text', '')[:100]}...\n\n")

            f.write("| Step | Status | Time |\n")
            f.write("|------|--------|------|\n")
            for step_name, step_data in test.get("steps", {}).items():
                if isinstance(step_data, dict):
                    status = "✅" if step_data.get("success") else "❌"
                    time_val = step_data.get("time") or step_data.get("total_time") or 0
                    f.write(f"| {step_name} | {status} | {time_val:.2f}s |\n")
            f.write("\n")

        f.write("## Conclusions\n\n")
        if results['summary']['passed'] == results['summary']['total_tests']:
            f.write("All tests passed. The translation pipeline is working correctly.\n")
        else:
            f.write("Some tests failed. Review the errors above.\n")

    log(f"\nReport saved to: {report_path}")

    return results

if __name__ == "__main__":
    main()
