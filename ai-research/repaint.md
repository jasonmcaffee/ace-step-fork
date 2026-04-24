# ACE-Step Repaint Audio Functionality - Deep Analysis

## Overview

The **Repaint** task in ACE-Step is a powerful audio inpainting/outpainting feature that allows regenerating specific time segments of audio while preserving the rest. It's based on a **context completion** principle where the model references surrounding audio context to generate coherent content within a specified time interval.

---

## Table of Contents

1. [Core Parameters](#core-parameters)
2. [How Repaint Works (Technical Deep Dive)](#how-repaint-works-technical-deep-dive)
3. [Parameter Flow Through the System](#parameter-flow-through-the-system)
4. [API Usage](#api-usage)
5. [Gradio UI Implementation](#gradio-ui-implementation)
6. [Impact on Generated Audio](#impact-on-generated-audio)
7. [Advanced Use Cases](#advanced-use-cases)
8. [Comparison with Other Tasks](#comparison-with-other-tasks)

---

## Core Parameters

### Primary Repaint Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | `str` | `"text2music"` | Must be set to `"repaint"` for repaint functionality |
| `src_audio` | `str` | `None` | **Required** - Path to the source audio file to repaint |
| `repainting_start` | `float` | `0.0` | Start time in seconds of the region to regenerate |
| `repainting_end` | `float` | `-1` | End time in seconds (`-1` means until end of file) |
| `caption` | `str` | `""` | Description of desired content for the repainted section |

### Secondary Parameters That Affect Repaint

| Parameter | Type | Default | Impact on Repaint |
|-----------|------|---------|-------------------|
| `instruction` | `str` | Auto-generated | Auto-set to `"Repaint the mask area based on the given conditions:"` |
| `lyrics` | `str` | `""` | Lyrics for vocal content in the repainted region |
| `bpm` | `int` | `None` | Tempo hint - can help maintain rhythm consistency |
| `keyscale` | `str` | `""` | Musical key hint - helps with harmonic consistency |
| `inference_steps` | `int` | `8` | Quality/speed tradeoff for the repainted section |
| `guidance_scale` | `float` | `7.0` | How strongly to follow the caption (base model only) |

---

## How Repaint Works (Technical Deep Dive)

### Step 1: Source Audio Processing

When repaint is triggered, the source audio is processed:

```python
# From handler.py - determine_task_type()
is_repaint_task = (task_type == "repaint")
can_use_repainting = is_repaint_task or is_lego_task
```

The source audio is loaded and potentially padded for outpainting scenarios.

### Step 2: Padding for Outpainting

If `repainting_start` is negative (outpainting at the beginning) or `repainting_end` exceeds source duration (outpainting at the end), padding is applied:

```python
# From handler.py - prepare_padding_info()
left_padding_duration = max(0, -repainting_start)
right_padding_duration = max(0, actual_end - src_audio_duration)

# Create padded audio
left_padding_frames = int(left_padding_duration * 48000)
right_padding_frames = int(right_padding_duration * 48000)
```

### Step 3: Time-to-Latent Conversion

The time-based repaint region is converted to latent frame indices:

```python
# From handler.py - _prepare_batch()
# Audio sample rate: 48000 Hz
# Latent frame rate: 48000 / 1920 = 25 Hz (one latent frame per 40ms)

start_latent = int(adjusted_start_sec * self.sample_rate // 1920)  # 48000 / 1920 = 25Hz
end_latent = int(adjusted_end_sec * self.sample_rate // 1920)

# Clamp to valid range
start_latent = max(0, min(start_latent, max_latent_length - 1))
end_latent = max(start_latent + 1, min(end_latent, max_latent_length))
```

**Key conversion formula:**
- 1 second of audio = 25 latent frames
- `latent_frame = time_seconds * 48000 / 1920`

### Step 4: Chunk Mask Creation

A binary mask is created to indicate which latent frames should be regenerated:

```python
# From handler.py - _prepare_batch()
# Create mask: False = keep original, True = generate new
mask = torch.zeros(max_latent_length, dtype=torch.bool, device=self.device)
mask[start_latent:end_latent] = True
chunk_masks.append(mask)
spans.append(("repainting", start_latent, end_latent))
```

### Step 5: Source Latents Preparation

For repaint tasks, source latents are created by:
1. Encoding the source audio through VAE to get latent representation
2. **Replacing the repaint region with silence latents**

```python
# From handler.py - _prepare_batch()
if item_has_repainting:
    # Repaint task: src_latents = target_latents with inpainting region replaced by silence_latent
    # 1. Clone target_latents (encoded from src audio, preserving original audio)
    src_latent = target_latents[i].clone()
    # 2. Replace inpainting region with silence_latent
    start_latent, end_latent = repainting_ranges[i]
    src_latent[start_latent:end_latent] = silence_latent_tiled[start_latent:end_latent]
    src_latents_list.append(src_latent)
```

### Step 6: DiT Generation with Mask

The DiT model generates audio conditioned on:
- Text embeddings (caption, lyrics)
- Reference audio embeddings (if provided)
- Source latents (with silence in repaint region)
- Chunk mask (indicating which frames to generate)

```python
# From handler.py - _generate()
outputs = self.model.generate_audio(
    ...
    hidden_states=src_latents,
    src_latents=src_latents,
    chunk_masks=chunk_mask,  # Boolean mask: True = generate, False = keep
    ...
)
```

### Step 7: Final Audio Assembly

The model output is combined:
- Frames where `chunk_mask=True`: Use generated content
- Frames where `chunk_mask=False`: Keep original source audio

---

## Parameter Flow Through the System

### Gradio UI → Handler

```
Gradio UI Components:
├── repainting_start (gr.Number)
├── repainting_end (gr.Number)
├── src_audio (gr.Audio)
├── task_type (gr.Dropdown) → set to "repaint"
└── caption (gr.Textbox)
        ↓
generate_with_progress()
        ↓
GenerationParams(
    task_type="repaint",
    src_audio=src_audio,
    repainting_start=repainting_start,
    repainting_end=repainting_end,
    caption=caption,
)
        ↓
generate_music(dit_handler, llm_handler, params, config)
        ↓
dit_handler.generate_music(
    repainting_start=params.repainting_start,
    repainting_end=params.repainting_end,
    ...
)
```

### API Server → Handler

```python
# From api_server.py - GenerateMusicRequest
repainting_start: float = 0.0
repainting_end: Optional[float] = None

# Parameters are passed to GenerationParams
params = GenerationParams(
    task_type=req.task_type,  # "repaint"
    src_audio=req.src_audio_path,
    repainting_start=req.repainting_start,
    repainting_end=req.repainting_end if req.repainting_end else -1,
    ...
)
```

### Instruction Auto-Selection

When `task_type="repaint"`, the instruction is automatically set:

```python
# From constants.py
TASK_INSTRUCTIONS = {
    "repaint": "Repaint the mask area based on the given conditions:",
    ...
}

# From handler.py - get_default_instruction()
elif task_type == "repaint":
    return TASK_INSTRUCTIONS["repaint"]
```

---

## API Usage

### Python API (acestep.inference)

```python
from acestep.inference import GenerationParams, GenerationConfig, generate_music

params = GenerationParams(
    task_type="repaint",
    src_audio="/path/to/original.mp3",
    repainting_start=10.0,    # Start at 10 seconds
    repainting_end=20.0,      # End at 20 seconds (-1 for file end)
    caption="smooth jazz piano solo with subtle percussion",
    lyrics="[Instrumental]",  # Or actual lyrics for vocal sections
)

config = GenerationConfig(
    batch_size=2,
    audio_format="flac",
)

result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")
```

### REST API

```json
POST /release_task
{
    "task_type": "repaint",
    "src_audio_path": "/path/to/audio.mp3",
    "repainting_start": 10.0,
    "repainting_end": 20.0,
    "prompt": "smooth jazz piano solo with subtle percussion",
    "lyrics": "[Instrumental]"
}
```

---

## Gradio UI Implementation

### UI Components

Located in `acestep/gradio_ui/interfaces/generation.py`:

```python
# Repainting controls (visible only for repaint/lego tasks)
with gr.Group(visible=False) as repainting_group:
    gr.HTML(f"<h5>🎨 Repainting Controls (seconds)</h5>")
    with gr.Row():
        repainting_start = gr.Number(
            label="Repainting Start",
            value=0.0,
            step=0.1,
        )
        repainting_end = gr.Number(
            label="Repainting End",
            value=-1,
            minimum=-1,
            step=0.1,
        )
```

### Visibility Toggle

The repainting controls are shown/hidden based on task type:

```python
# From generation_handlers.py - on_task_type_change()
repainting_visible = task_type_value in ["repaint", "lego"]
return (
    ...
    gr.update(visible=repainting_visible),  # repainting_group
    ...
)
```

---

## Impact on Generated Audio

### How Each Parameter Affects Output

#### `repainting_start` and `repainting_end`

| Configuration | Effect |
|---------------|--------|
| `start=0, end=10` | Regenerate first 10 seconds (intro modification) |
| `start=10, end=20` | Regenerate middle section (10-20 seconds) |
| `start=30, end=-1` | Regenerate from 30 seconds to end (outro modification) |
| `start=-5, end=10` | Outpainting: Generate 5 seconds before + first 10 seconds |

**Operation Range:** 3 seconds to 90 seconds per repaint operation.

#### `caption`

The caption strongly influences the character of the repainted region:
- **Instrumental changes:** "energetic guitar solo" vs "soft piano melody"
- **Mood changes:** "intense climax build-up" vs "calm ambient fade"
- **Structure changes:** "verse with vocals" vs "instrumental chorus"

#### Context Influence

The model considers **surrounding context** when generating:
- Audio before `repainting_start` provides left context
- Audio after `repainting_end` provides right context
- This enables smooth transitions and coherent musical flow

---

## Advanced Use Cases

### 1. Infinite Duration Generation

Extend audio indefinitely through chained repaint operations:

```python
# Generate initial 30 seconds
result1 = generate_music(params_text2music)

# Extend with repaint (last 5 seconds overlap + 25 new seconds)
params_extend = GenerationParams(
    task_type="repaint",
    src_audio=result1.audios[0]["path"],
    repainting_start=25.0,  # 5 second overlap for smooth transition
    repainting_end=60.0,    # Extend to 60 seconds total
    caption="continuation with building energy",
)
result2 = generate_music(params_extend)
```

### 2. Section Replacement

Fix a problematic section:

```python
# Original audio has awkward vocals at 15-25 seconds
params = GenerationParams(
    task_type="repaint",
    src_audio="original.mp3",
    repainting_start=15.0,
    repainting_end=25.0,
    caption="smoother vocal transition with harmonies",
    lyrics="[Chorus]\nNew lyrics for this section...",
)
```

### 3. Structure Modification

Change verse to chorus:

```python
params = GenerationParams(
    task_type="repaint",
    src_audio="song.mp3",
    repainting_start=45.0,
    repainting_end=60.0,
    caption="energetic chorus with full instrumentation",
    lyrics="[Chorus]\nHook lyrics here...",
)
```

### 4. Audio Stitching

Intelligently connect two audio clips:

```python
# Concatenate audio1 and audio2, then repaint the junction
combined_audio = concatenate_audio(audio1, audio2)
junction_time = audio1_duration

params = GenerationParams(
    task_type="repaint",
    src_audio=combined_audio,
    repainting_start=junction_time - 5,  # 5 sec before junction
    repainting_end=junction_time + 5,    # 5 sec after junction
    caption="smooth transition maintaining musical flow",
)
```

---

## Comparison with Other Tasks

| Task | Source Audio | Repaint Region | Use Case |
|------|--------------|----------------|----------|
| **text2music** | Not used | N/A | Generate from scratch |
| **repaint** | Required | Specified region | Modify/extend specific section |
| **cover** | Required | Full audio | Style transformation |
| **lego** | Required | Specified region | Add specific instrument track |

### Repaint vs Cover

| Aspect | Repaint | Cover |
|--------|---------|-------|
| Region affected | Specific time range | Entire audio |
| Original preserved | Yes (outside region) | No (fully transformed) |
| `audio_cover_strength` | Not used | Used (0.0-1.0) |
| Context-aware | Yes (considers boundaries) | Globally transforms |

### Repaint vs Lego

| Aspect | Repaint | Lego |
|--------|---------|------|
| Purpose | General modification | Add instrument track |
| Output | Full mix | Specific track |
| Instruction | "Repaint the mask area..." | "Generate the {TRACK} track..." |
| Models | Turbo + Base | Base only |

---

## Technical Details

### Latent Space Conversion

- **Sample rate:** 48,000 Hz
- **Latent rate:** 25 Hz (one latent frame per 40ms)
- **Conversion:** `latent_frame = audio_samples / 1920`

### Memory Considerations

Repaint operations use similar memory as full generation for the repainted segment:
- Longer repaint regions = more GPU memory
- Batch size applies to variations of the repaint

### LM Behavior for Repaint

From `acestep/inference.py`:

```python
# LM is skipped for cover/repaint tasks - these use reference/src audio directly
skip_lm_tasks = {"cover", "repaint"}

if params.task_type in skip_lm_tasks:
    logger.info(f"Skipping LM for task_type='{params.task_type}' - using DiT directly")
```

The 5Hz Language Model is **not used** for repaint tasks because:
- Audio codes are derived from source audio
- Context completion relies on source audio encoding, not LM generation

---

## Summary

The **Repaint** functionality in ACE-Step provides powerful local audio manipulation by:

1. **Encoding** source audio to latent space
2. **Masking** the specified time region with silence latents
3. **Generating** new content for the masked region using DiT
4. **Conditioning** on surrounding context for smooth transitions

Key parameters:
- `repainting_start`: Where to begin regeneration (seconds)
- `repainting_end`: Where to end regeneration (-1 for file end)
- `src_audio`: The source audio to modify
- `caption`: Description of desired content for the region

This enables use cases like infinite duration generation, section replacement, structure modification, and intelligent audio stitching.
