# Audio Extension in ACE-Step

This document analyzes how ACE-Step extends existing audio tracks to longer durations. The system uses the **repaint** task with specific parameters to generate new audio content that seamlessly continues from the original.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
4. [Extension Modes](#extension-modes)
5. [API Reference](#api-reference)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)
8. [Limitations](#limitations)

---

## Overview

Audio extension in ACE-Step allows you to:

| Feature | Description |
|---------|-------------|
| **Extend at end** | Add new content after the original audio ends |
| **Extend at beginning** | Prepend new content before the original audio starts |
| **Modify sections** | Replace any region with new generated content |
| **Infinite duration** | Chain multiple extensions for unlimited length |

The extension is powered by the **repaint** task type, which uses context from the original audio to generate coherent continuations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Audio Extension Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │  Original Audio  │  (e.g., 30 seconds)                                    │
│  │  [============]  │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Repaint Task Parameters                           │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  repainting_start: 25.0   (keep first 25s as context)          │  │   │
│  │  │  repainting_end: 60.0     (extend to 60s total)                │  │   │
│  │  │  caption: "continuation with building energy"                   │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────────┘ │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Extended Audio                                 │   │
│  │  [=========================][+++++++++++++++++++++++++++++++++++++]  │   │
│  │   Original context (0-25s)   New generated content (25-60s)          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Context Preservation

The model preserves audio content **before** `repainting_start` as context. This context informs the generation of new content, ensuring:
- Consistent tempo and rhythm
- Matching instrumentation and timbre
- Coherent musical progression
- Smooth transitions

### 2. Region Generation

Content between `repainting_start` and `repainting_end` is regenerated. For extension:
- **End extension**: `repainting_end` is set beyond the original audio duration
- **Beginning extension**: `repainting_start` is set to a negative value

### 3. Automatic Padding

When extending beyond the original duration, the system automatically pads the audio:
- Calculates `right_padding_duration = repainting_end - src_audio_duration`
- Creates silent padding frames to reach target duration
- Fills padded region with generated content

---

## Extension Modes

### Mode 1: Extend at End (Most Common)

Add new content after the original audio:

```
Original:    [===========]  (30 seconds)
Extended:    [=====][+++++++++++++++++]  (60 seconds)
             ↑     ↑                   ↑
             0    25s                 60s
                   └─ repainting_start
```

**Parameters:**
- `repainting_start`: A few seconds before end (e.g., 25.0 for 30s audio)
- `repainting_end`: Target total duration (e.g., 60.0)

The overlap (5 seconds in this example) provides context for smooth transition.

### Mode 2: Extend at Beginning

Add new content before the original audio:

```
Original:           [===========]  (30 seconds)
Extended:    [+++++][===========]  (40 seconds)
             ↑     ↑
           -10s    0
             └─ repainting_start (negative)
```

**Parameters:**
- `repainting_start`: Negative value (e.g., -10.0)
- `repainting_end`: A few seconds into original (e.g., 5.0 for overlap)

### Mode 3: Section Modification

Replace a middle section with new content:

```
Original:    [=====|xxxxx|=====]  (30 seconds)
Modified:    [=====|+++++|=====]  (30 seconds)
                   ↑     ↑
                  10s   20s
```

**Parameters:**
- `repainting_start`: Section start (e.g., 10.0)
- `repainting_end`: Section end (e.g., 20.0)

---

## API Reference

### HTTP Endpoint: POST /extend_audio

**URL**: `/extend_audio`  
**Method**: `POST`  
**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file` | file | **required** | Audio file to extend (mp3, wav, flac, etc.) |
| `extend_duration` | float | `30.0` | How many seconds to add to the audio |
| `overlap_duration` | float | `5.0` | Seconds of overlap for smooth transition (context) |
| `extend_direction` | string | `"end"` | Direction to extend: `"end"`, `"start"`, or `"both"` |
| `caption` | string | `""` | Description for the extended section |
| `lyrics` | string | `""` | Lyrics for the extended section (if vocals) |
| `vocal_language` | string | `"unknown"` | Language hint for lyrics generation |
| `bpm` | int | `null` | BPM (auto-detected if not provided) |
| `key_scale` | string | `""` | Musical key (auto-detected if not provided) |
| `time_signature` | string | `""` | Time signature (auto-detected if not provided) |
| `inference_steps` | int | `8` | Diffusion steps (8 for turbo, 32-100 for base) |
| `guidance_scale` | float | `7.0` | CFG strength (base model only) |
| `seed` | int | `-1` | Random seed (-1 for random) |
| `thinking` | bool | `true` | Use 5Hz LM for code generation |
| `lm_temperature` | float | `0.85` | LM sampling temperature |

#### Response Format

```json
{
    "data": {
        "audio_data": "data:audio/flac;base64,ZkxhQwAA...",
        "audio_format": "flac",
        "original_duration": 30.0,
        "extended_duration": 60.0,
        "repainting_start": 25.0,
        "repainting_end": 60.0,
        "extend_direction": "end",
        "status_message": "✅ Audio extended successfully from 30.0s to 60.0s"
    },
    "code": 200,
    "error": null,
    "timestamp": 1706688000000
}
```

**Decoding the audio:**

The `audio_data` field contains a base64-encoded data URL. Extract and decode it:

```python
import base64

audio_data_url = response["data"]["audio_data"]
# Strip "data:audio/flac;base64," prefix
b64_data = audio_data_url.split(",", 1)[1]
audio_bytes = base64.b64decode(b64_data)

# Save to file
with open("extended_audio.flac", "wb") as f:
    f.write(audio_bytes)
```
```

#### cURL Examples

**Extend audio by 30 seconds at the end:**

```bash
curl -X POST http://localhost:8001/extend_audio \
  -F "audio_file=@/path/to/original.mp3" \
  -F "extend_duration=30" \
  -F "caption=energetic continuation with building drums"
```

**Extend with specific musical parameters:**

```bash
curl -X POST http://localhost:8001/extend_audio \
  -F "audio_file=@/path/to/song.mp3" \
  -F "extend_duration=45" \
  -F "overlap_duration=10" \
  -F "caption=epic orchestral climax" \
  -F "bpm=120" \
  -F "key_scale=C major"
```

**Extend at the beginning:**

```bash
curl -X POST http://localhost:8001/extend_audio \
  -F "audio_file=@/path/to/song.mp3" \
  -F "extend_duration=15" \
  -F "extend_direction=start" \
  -F "caption=gentle ambient intro"
```

---

## Code Examples

### Example 1: Extend Audio Using Python API

```python
from acestep.inference import generate_music, GenerationParams, GenerationConfig

# Original audio is 30 seconds, extend to 60 seconds
params = GenerationParams(
    task_type="repaint",
    src_audio="/path/to/original.mp3",
    repainting_start=25.0,  # Keep first 25 seconds as context
    repainting_end=60.0,    # Extend to 60 seconds total
    caption="continuation with building energy and full instrumentation",
    thinking=True,
)

config = GenerationConfig(batch_size=1)

result = generate_music(
    dit_handler=dit_handler,
    llm_handler=llm_handler,
    params=params,
    config=config,
    save_dir="/output/dir"
)

if result.success:
    print(f"Extended audio saved to: {result.audios[0]['path']}")
```

### Example 2: Chain Extensions for Infinite Duration

```python
def extend_indefinitely(original_audio: str, segments: int = 5, segment_length: float = 30.0):
    """Extend audio through chained repaint operations."""
    current_audio = original_audio
    
    for i in range(segments):
        # Get current audio duration
        import torchaudio
        audio, sr = torchaudio.load(current_audio)
        current_duration = audio.shape[-1] / sr
        
        # Extend by segment_length with 5-second overlap
        overlap = 5.0
        params = GenerationParams(
            task_type="repaint",
            src_audio=current_audio,
            repainting_start=current_duration - overlap,
            repainting_end=current_duration + segment_length - overlap,
            caption=f"continuation segment {i+1}",
            thinking=True,
        )
        
        result = generate_music(dit_handler, llm_handler, params, config, save_dir)
        
        if result.success:
            current_audio = result.audios[0]['path']
            print(f"Segment {i+1}: Extended to {current_duration + segment_length - overlap:.1f}s")
        else:
            print(f"Error: {result.status_message}")
            break
    
    return current_audio
```

### Example 3: Prepend an Intro

```python
# Add a 15-second intro before the song
params = GenerationParams(
    task_type="repaint",
    src_audio="/path/to/song.mp3",
    repainting_start=-15.0,  # Negative = prepend
    repainting_end=5.0,      # 5 seconds overlap into original
    caption="atmospheric synth intro with gradual build",
    thinking=True,
)
```

---

## Best Practices

### 1. Overlap Duration

The overlap between original content and generated content is crucial for smooth transitions:

| Overlap | Use Case |
|---------|----------|
| 3-5 seconds | Quick transitions, percussion-heavy music |
| 5-10 seconds | Most music styles, balanced transition |
| 10-15 seconds | Complex orchestral, gradual mood changes |

### 2. Caption Guidance

Provide specific captions for the extended section:

| Good Caption | Bad Caption |
|--------------|-------------|
| "energetic guitar solo building to climax" | "more music" |
| "calm piano outro with fading reverb" | "ending" |
| "verse 2 with female vocals, same melody" | "continue" |

### 3. Preserve Musical Coherence

For best results:
- Match the BPM of the original (or let it auto-detect)
- Use consistent key/scale
- Describe the desired energy level relative to context

### 4. Segment Length

The repaint task has limits:
- **Minimum**: 3 seconds
- **Maximum**: 90 seconds per operation
- For longer extensions, chain multiple operations

---

## Limitations

### 1. Maximum Single Extension

Each repaint operation can generate 3-90 seconds. For longer extensions, chain multiple operations.

### 2. Audio Quality

Extended sections may have slightly different characteristics than the original. Use longer overlap for smoother blending.

### 3. Vocal Continuity

Extending vocal sections requires:
- Matching lyrics for the new section
- Correct vocal_language setting
- Similar vocal style description in caption

### 4. Tempo/Key Shifts

The model maintains the original tempo/key by default. Intentional changes require explicit caption guidance.

---

## Internal Implementation

### Repaint Task Flow

The extension is implemented through the repaint task in `acestep/handler.py`:

1. **Audio Loading**: Source audio is loaded and normalized to stereo 48kHz
2. **Padding Calculation**: `prepare_padding_info()` calculates required padding
3. **Mask Creation**: A mask is created marking the region to regenerate
4. **Context Encoding**: Original audio is encoded to latent space
5. **Generation**: DiT generates new content for masked region
6. **Decoding**: Latents are decoded back to audio

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `prepare_padding_info()` | `handler.py` | Calculate left/right padding for extension |
| `generate_music()` | `inference.py` | High-level generation orchestration |
| `_process_src_audio()` | `handler.py` | Load and normalize source audio |

### Padding Logic

From `acestep/handler.py`:

```python
# Calculate padding needed for extension
left_padding_duration = max(0, -repainting_start)  # Negative start = prepend
right_padding_duration = max(0, repainting_end - src_audio_duration)  # Beyond end = extend

# Pad audio with silence
padded_audio = torch.nn.functional.pad(
    src_audio,
    (left_padding_frames, right_padding_frames),
    mode='constant',
    value=0
)
```

---

## Summary

| Feature | Support | Notes |
|---------|---------|-------|
| Extend at end | ✅ Full | Set repainting_end > audio duration |
| Extend at beginning | ✅ Full | Set negative repainting_start |
| Section modification | ✅ Full | Set start/end within audio |
| Infinite extension | ✅ Via chaining | Multiple repaint operations |
| HTTP API endpoint | ✅ Available | `POST /extend_audio` |
| Python API | ✅ Available | `generate_music()` with repaint task |

Audio extension is a powerful feature for:
- Creating longer compositions from short sketches
- Adding intros and outros
- Modifying specific sections
- Building complete songs through iterative generation
