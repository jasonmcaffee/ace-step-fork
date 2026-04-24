# Audio Description (Understanding) in ACE-Step

This document analyzes how ACE-Step describes and understands existing audio tracks. The system uses a 5Hz Language Model (LLM) to analyze audio semantic codes and generate comprehensive metadata including captions, lyrics, BPM, key, time signature, and more.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Audio Understanding Flow](#audio-understanding-flow)
5. [API Reference](#api-reference)
6. [Data Structures](#data-structures)
7. [Code Examples](#code-examples)
8. [UI Integration](#ui-integration)
9. [Training/Dataset Builder Integration](#trainingdataset-builder-integration)
10. [Lyrics Generation Limitations](#lyrics-generation-limitations)

---

## Overview

ACE-Step can describe audio through a process called **"audio understanding"**. This is the reverse of music generation:

| Direction | Input | Output |
|-----------|-------|--------|
| **Generation** | Text prompt, lyrics, metadata | Audio codes → Audio file |
| **Understanding** | Audio codes | Caption, lyrics, metadata |

The understanding process works by:
1. Converting an audio file to semantic audio codes (using VAE + tokenizer)
2. Feeding these codes to the 5Hz Language Model
3. The LLM generates a structured description (caption, genre, lyrics, BPM, key, etc.)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Audio Understanding Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  Audio File  │───▶│     VAE      │───▶│   Audio Codes (5Hz tokens)   │   │
│  │  (mp3/wav)   │    │   Encoder    │    │   "<|audio_code_123|>..."    │   │
│  └──────────────┘    └──────────────┘    └───────────────┬──────────────┘   │
│                                                           │                  │
│                                                           ▼                  │
│                                          ┌────────────────────────────────┐  │
│                                          │      5Hz Language Model        │  │
│                                          │   (understand_audio_from_codes)│  │
│                                          │                                │  │
│                                          │  Instruction:                  │  │
│                                          │  "Understand the given musical │  │
│                                          │   conditions and describe the  │  │
│                                          │   audio semantics accordingly" │  │
│                                          └───────────────┬────────────────┘  │
│                                                          │                   │
│                                                          ▼                   │
│                                          ┌────────────────────────────────┐  │
│                                          │     UnderstandResult           │  │
│                                          │  - caption (str)               │  │
│                                          │  - lyrics (str)                │  │
│                                          │  - bpm (int)                   │  │
│                                          │  - duration (float)            │  │
│                                          │  - keyscale (str)              │  │
│                                          │  - language (str)              │  │
│                                          │  - timesignature (str)         │  │
│                                          └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Audio to Codes Conversion (`handler.py`)

The `AceStepHandler.convert_src_audio_to_codes()` method converts audio files to semantic code strings:

**Location**: `acestep/handler.py` (lines 1309-1365)

```python
def convert_src_audio_to_codes(self, audio_file) -> str:
    """
    Convert uploaded source audio to audio codes string.
    
    Args:
        audio_file: Path to audio file or None
        
    Returns:
        Formatted codes string like '<|audio_code_123|><|audio_code_456|>...' 
        or error message starting with '❌'
    """
```

**Process**:
1. Load and normalize audio to stereo 48kHz
2. Check if audio is silent (returns error if so)
3. Encode audio to latent space using VAE `tiled_encode`
4. Tokenize latents using the DiT model's tokenizer
5. Format indices as code string: `<|audio_code_123|><|audio_code_456|>...`

### 2. Understanding from Codes (`llm_inference.py`)

The `LLMHandler.understand_audio_from_codes()` method analyzes codes using the 5Hz LLM:

**Location**: `acestep/llm_inference.py` (lines 1373-1471)

```python
def understand_audio_from_codes(
    self,
    audio_codes: str,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Understand audio codes and generate metadata + lyrics.

    Returns:
        Tuple of (metadata_dict, status_message)
        metadata_dict contains:
            - bpm: int or str
            - caption: str
            - duration: int or str
            - keyscale: str
            - language: str
            - timesignature: str
            - lyrics: str (extracted from output after </think>)
    """
```

### 3. High-Level API (`inference.py`)

The `understand_music()` function provides a clean interface:

**Location**: `acestep/inference.py` (lines 709-834)

```python
def understand_music(
    llm_handler,
    audio_codes: str,
    temperature: float = 0.85,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> UnderstandResult:
    """
    Understand music from audio codes using the 5Hz Language Model.
    
    If audio_codes is empty or "NO USER INPUT", the LM will generate 
    a sample example instead of analyzing existing codes.
    """
```

---

## Audio Understanding Flow

### Step 1: Convert Audio to Codes

```python
from acestep.handler import AceStepHandler

# Initialize handler
handler = AceStepHandler()
handler.initialize_service(project_root, config_path, device)

# Convert audio file to codes
codes_string = handler.convert_src_audio_to_codes("/path/to/audio.mp3")
# Result: "<|audio_code_18953|><|audio_code_13833|><|audio_code_7821|>..."
```

### Step 2: Understand the Codes

```python
from acestep.inference import understand_music
from acestep.llm_inference import LLMHandler

# Initialize LLM handler
llm_handler = LLMHandler()
llm_handler.initialize(checkpoint_dir, lm_model_path="acestep-5Hz-lm-0.6B", backend="vllm")

# Understand the audio codes
result = understand_music(
    llm_handler=llm_handler,
    audio_codes=codes_string,
    temperature=0.85,
    use_constrained_decoding=True
)

if result.success:
    print(f"Caption: {result.caption}")
    print(f"Lyrics: {result.lyrics}")
    print(f"BPM: {result.bpm}")
    print(f"Duration: {result.duration}s")
    print(f"Key: {result.keyscale}")
    print(f"Language: {result.language}")
    print(f"Time Signature: {result.timesignature}")
```

---

## API Reference

### HTTP Endpoint: POST /describe_audio

**URL**: `/describe_audio`  
**Method**: `POST`  
**Content-Type**: `multipart/form-data` (for file upload) or `application/json` (for audio codes)

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file` | file | - | Audio file to analyze (mp3, wav, flac, etc.). Use with `multipart/form-data`. |
| `audio_codes` | string | `""` | Audio code tokens (e.g., `"<\|audio_code_123\|>..."`). Alternative to `audio_file`. Leave empty to generate a random sample example. |
| `vocal_language` | string | `""` | Language hint for lyrics generation (e.g., `"en"`, `"zh"`, `"ja"`). Guides the LLM to generate lyrics in the specified language. See [Lyrics Generation Limitations](#lyrics-generation-limitations). |
| `temperature` | float | `0.85` | LM sampling temperature (0.0-2.0). Higher = more creative. |
| `top_k` | int | `null` | Top-K sampling (0/null disables) |
| `top_p` | float | `null` | Top-P nucleus sampling (>=1 disables) |
| `repetition_penalty` | float | `1.0` | Repetition penalty |
| `use_constrained_decoding` | bool | `true` | Use FSM-based constrained decoding for structured output |
| `constrained_decoding_debug` | bool | `false` | Enable debug logging for constrained decoding |

#### Response Format

```json
{
    "data": {
        "caption": "A melancholic indie folk song with fingerpicked acoustic guitar...",
        "lyrics": "[Verse 1]\nUnder the pale moonlight...",
        "bpm": 72,
        "duration": 180,
        "keyscale": "A minor",
        "language": "en",
        "timesignature": "4",
        "status_message": "✅ Understanding completed successfully\nGenerated fields: bpm, caption, duration, keyscale, language, timesignature, lyrics"
    },
    "code": 200,
    "error": null,
    "timestamp": 1706688000000,
    "extra": null
}
```

#### cURL Examples

**Describe audio from uploaded file** (recommended):

```bash
curl -X POST http://localhost:8001/describe_audio \
  -F "audio_file=@/path/to/audio.mp3" \
  -F "temperature=0.85"
```

**Describe audio with language hint** (for English lyrics):

```bash
curl -X POST http://localhost:8001/describe_audio \
  -F "audio_file=@/path/to/audio.mp3" \
  -F "vocal_language=en" \
  -F "temperature=0.85"
```

**Describe audio from pre-computed codes** (JSON):

```bash
curl -X POST http://localhost:8001/describe_audio \
  -H "Content-Type: application/json" \
  -d '{
    "audio_codes": "<|audio_code_18953|><|audio_code_13833|><|audio_code_7821|>...",
    "temperature": 0.85
  }'
```

**Generate a random sample example** (no input):

```bash
curl -X POST http://localhost:8001/describe_audio \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### Error Responses

| Code | Error |
|------|-------|
| 400 | Audio file not found / Invalid audio codes |
| 500 | LLM not initialized / DiT model not initialized |

---

## Data Structures

### UnderstandResult

**Location**: `acestep/inference.py` (lines 207-242)

```python
@dataclass
class UnderstandResult:
    """Result of music understanding from audio codes.
    
    Attributes:
        # Metadata Fields
        caption: str = ""          # Generated caption describing the music
        lyrics: str = ""           # Generated or extracted lyrics
        bpm: Optional[int] = None  # Beats per minute
        duration: Optional[float] = None  # Duration in seconds
        keyscale: str = ""         # Musical key (e.g., "C Major")
        language: str = ""         # Vocal language code (e.g., "en", "zh")
        timesignature: str = ""    # Time signature (e.g., "4" for 4/4)
        
        # Status
        status_message: str = ""   # Status message from understanding
        success: bool = True       # Whether understanding completed successfully
        error: Optional[str] = None  # Error message if understanding failed
    """
```

### Audio Code Format

Audio codes are formatted as a string of tokens:

```
<|audio_code_18953|><|audio_code_13833|><|audio_code_7821|><|audio_code_14592|>...
```

- Each token represents a 5Hz semantic audio frame
- 30 seconds of audio ≈ 150 tokens (30s × 5Hz = 150 frames)
- Tokens are indices into a codebook vocabulary

---

## Code Examples

### Example 1: Full Audio Description Pipeline

```python
import os
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import understand_music

# Setup paths
project_root = "/path/to/ace"
checkpoint_dir = os.path.join(project_root, "checkpoints")

# Initialize DiT handler (for audio encoding)
dit_handler = AceStepHandler()
dit_handler.initialize_service(
    project_root=project_root,
    config_path="acestep-v15-turbo",
    device="cuda",
    use_flash_attention=True
)

# Initialize LLM handler (for understanding)
llm_handler = LLMHandler()
llm_handler.initialize(
    checkpoint_dir=checkpoint_dir,
    lm_model_path="acestep-5Hz-lm-0.6B",
    backend="vllm",
    device="cuda"
)

# Step 1: Convert audio to codes
audio_path = "/path/to/my_song.mp3"
codes = dit_handler.convert_src_audio_to_codes(audio_path)

if codes.startswith("❌"):
    print(f"Error: {codes}")
else:
    # Step 2: Understand the codes
    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=codes,
        temperature=0.85
    )
    
    if result.success:
        print(f"=== Audio Description ===")
        print(f"Caption: {result.caption}")
        print(f"BPM: {result.bpm}")
        print(f"Key: {result.keyscale}")
        print(f"Time Signature: {result.timesignature}")
        print(f"Duration: {result.duration}s")
        print(f"Language: {result.language}")
        print(f"\n=== Lyrics ===")
        print(result.lyrics)
    else:
        print(f"Error: {result.error}")
```

### Example 2: Generate Sample Example (No Input)

When audio_codes is empty or "NO USER INPUT", the LM generates a random sample:

```python
# Generate a random sample example
result = understand_music(
    llm_handler=llm_handler,
    audio_codes="",  # Empty triggers sample generation
    temperature=0.85
)

# result.caption = "An upbeat electronic dance track with pulsing synths..."
# result.lyrics = "[Verse 1]\nDancing through the night..."
# result.bpm = 128
# etc.
```

---

## UI Integration

### Gradio UI - Transcribe Button

**Location**: `acestep/gradio_ui/events/generation_handlers.py` (lines 588-631)

The Gradio UI has a "Transcribe" button that:
1. Takes audio codes from a text input
2. Calls `understand_music()` to get metadata
3. Populates the UI form fields with the result

```python
def transcribe_audio_codes(llm_handler, audio_code_string, constrained_decoding_debug):
    """
    Transcribe audio codes to metadata using LLM understanding.
    If audio_code_string is empty, generate a sample example instead.
    
    Returns:
        Tuple of (status_message, caption, lyrics, bpm, duration, 
                  keyscale, language, timesignature, is_format_caption)
    """
    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=audio_code_string,
        use_constrained_decoding=True,
        constrained_decoding_debug=constrained_decoding_debug,
    )
    
    if not result.success:
        return result.status_message, "", "", None, None, "", "", "", False
    
    return (
        result.status_message,
        result.caption,
        result.lyrics,
        result.bpm,
        result.duration,
        result.keyscale,
        result.language,
        result.timesignature,
        True  # is_format_caption flag
    )
```

### Event Binding

**Location**: `acestep/gradio_ui/events/__init__.py` (lines 196-200)

```python
generation_section["transcribe_btn"].click(
    fn=lambda codes, debug: gen_h.transcribe_audio_codes(llm_handler, codes, debug),
    inputs=[
        generation_section["text2music_audio_code_string"],
        generation_section["constrained_decoding_debug"]
    ],
    outputs=[
        # status, caption, lyrics, bpm, duration, keyscale, language, timesig, is_format
    ]
)
```

---

## Training/Dataset Builder Integration

The audio description functionality is used extensively in the training pipeline for auto-labeling audio datasets.

### DatasetBuilder.label_sample()

**Location**: `acestep/training/dataset_builder.py` (lines 450-590)

```python
def label_sample(
    self,
    idx: int,
    dit_handler,
    llm_handler,
    format_lyrics: bool = False,
    transcribe_lyrics: bool = False,
    skip_metas: bool = False,
    progress_callback=None,
) -> Tuple[AudioSample, str]:
    """
    Label a single audio sample using DiT for encoding and LLM for understanding.
    
    Args:
        idx: Index of sample to label
        dit_handler: DiT handler for audio encoding
        llm_handler: LLM handler for caption generation
        format_lyrics: If True, use LLM to format user-provided lyrics
        transcribe_lyrics: If True, use LLM to transcribe lyrics from audio
        skip_metas: If True, skip generating BPM/Key/TimeSig but still generate caption/genre
    """
```

### Auto-Label Workflow

1. **Scan directory** for audio files
2. For each audio file:
   - Convert audio to codes using `dit_handler.convert_src_audio_to_codes()`
   - Call `llm_handler.understand_audio_from_codes()` to get metadata
   - Populate the `AudioSample` with caption, genre, bpm, key, etc.
3. Save labeled dataset as JSON for LoRA training

---

## LLM Prompt Structure

The 5Hz LM uses a specific prompt structure for understanding:

**System Instruction** (`acestep/constants.py` line 72):
```
"Understand the given musical conditions and describe the audio semantics accordingly:"
```

**User Content**:
```
<|audio_code_18953|><|audio_code_13833|><|audio_code_7821|>...
```

**Expected Output Format**:
```
<think>
# BPM
120

# Genres
pop, electronic, dance

# Duration
180

# Keyscale
C Major

# Caption
An upbeat electronic pop song with pulsing synthesizers...

# Vocal Language
en

# Timesignature
4
</think>
# Lyric
[Verse 1]
Dancing through the night...

[Chorus]
We're alive...
```

The LLM uses constrained decoding to ensure the output follows this structured format.

---

## Key Constants

From `acestep/constants.py`:

```python
# Understanding instruction
DEFAULT_LM_UNDERSTAND_INSTRUCTION = "Understand the given musical conditions and describe the audio semantics accordingly:"

# Valid languages for vocal_language field
VALID_LANGUAGES = [
    'ar', 'az', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en',
    'es', 'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'ht', 'hu', 'id',
    'is', 'it', 'ja', 'ko', 'la', 'lt', 'ms', 'ne', 'nl', 'no',
    'pa', 'pl', 'pt', 'ro', 'ru', 'sa', 'sk', 'sr', 'sv', 'sw',
    'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yue', 'zh',
    'unknown'
]

# Valid keyscales
KEYSCALE_NOTES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
KEYSCALE_ACCIDENTALS = ['', '#', 'b', '♯', '♭']
KEYSCALE_MODES = ['major', 'minor']

# Metadata ranges
BPM_MIN = 30
BPM_MAX = 300
DURATION_MIN = 10  # seconds
DURATION_MAX = 600  # seconds
VALID_TIME_SIGNATURES = [2, 3, 4, 6]  # for 2/4, 3/4, 4/4, 6/8
```

---

## Lyrics Generation Limitations

### Important: This is NOT Speech-to-Text Transcription

The lyrics generated by the `/describe_audio` endpoint are **not transcribed from the audio**. They are **generated/hallucinated** by the 5Hz Language Model based on musical semantics. This is a fundamental limitation of the current architecture:

| What it does | What it does NOT do |
|--------------|---------------------|
| Generates lyrics that match the musical style, mood, and structure | Transcribe actual words sung in the audio |
| Produces lyrics in the requested language (via `vocal_language` hint) | Recognize or decode spoken/sung text |
| Creates stylistically appropriate verse/chorus structures | Perform automatic speech recognition (ASR) |

### Why This Happens

1. **Audio Codes are Semantic, Not Phonetic**: The VAE encoder converts audio to semantic codes that capture musical features (rhythm, melody, timbre, energy) but not explicit speech content.

2. **The LLM Never "Hears" Words**: The 5Hz LM receives tokenized musical features, not audio waveforms. It has no mechanism to decode human speech.

3. **Training Data**: The model was trained to associate musical semantics with lyrics that match the style, not to transcribe what is actually sung.

### Using the `vocal_language` Parameter

The `vocal_language` parameter helps guide the LLM to generate lyrics in a specific language:

```json
{
  "audio_path": "/path/to/english-song.mp3",
  "vocal_language": "en"
}
```

**Valid language codes**: `ar`, `az`, `bg`, `bn`, `ca`, `cs`, `da`, `de`, `el`, `en`, `es`, `fa`, `fi`, `fr`, `he`, `hi`, `hr`, `ht`, `hu`, `id`, `is`, `it`, `ja`, `ko`, `la`, `lt`, `ms`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sa`, `sk`, `sr`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`, `uk`, `ur`, `vi`, `yue`, `zh`

Without this hint, the LLM may generate lyrics in any language based on its training distribution.

### For Actual Lyrics Transcription

If you need to transcribe actual sung lyrics, use a dedicated speech-to-text (ASR) system like:
- **Whisper** (OpenAI) - Excellent multilingual ASR
- **Demucs + Whisper** - Separate vocals first, then transcribe
- **wav2vec2** - Facebook's ASR model
- Commercial APIs: Google Speech-to-Text, AWS Transcribe, etc.

---

## Summary

| Feature | Availability | Location |
|---------|--------------|----------|
| Audio → Codes conversion | ✅ Python API | `AceStepHandler.convert_src_audio_to_codes()` |
| Codes → Description | ✅ Python API | `understand_music()` / `LLMHandler.understand_audio_from_codes()` |
| HTTP API endpoint | ✅ Available | `POST /describe_audio` in `api_server.py` |
| Gradio UI | ✅ Available | "Transcribe" button in Generation tab |
| Training integration | ✅ Available | `DatasetBuilder.label_sample()` |

The audio description functionality is fully implemented and accessible via:
- **HTTP REST API**: `POST /describe_audio`
- **Python API**: `understand_music()` function
- **Gradio UI**: "Transcribe" button
- **Training pipeline**: Auto-labeling in Dataset Builder
