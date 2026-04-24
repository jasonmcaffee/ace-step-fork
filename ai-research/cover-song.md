# ACE-Step Cover Song Functionality - Deep Analysis

## Overview

The **Cover** task in ACE-Step is a powerful audio transformation feature that allows generating cover versions of existing music while maintaining the structural elements (melody, rhythm, chords) but changing the style, timbre, and instrumentation. It uses source audio's **semantic structure information** to guide generation, enabling style transfer, remixing, and creative reinterpretation of existing tracks.

---

## Table of Contents

1. [Core Parameters](#core-parameters)
2. [How Cover Works (Technical Deep Dive)](#how-cover-works-technical-deep-dive)
3. [Parameter Flow Through the System](#parameter-flow-through-the-system)
4. [API Usage](#api-usage)
5. [Gradio UI Implementation](#gradio-ui-implementation)
6. [Impact on Generated Audio](#impact-on-generated-audio)
7. [Advanced Use Cases](#advanced-use-cases)
8. [Comparison with Other Tasks](#comparison-with-other-tasks)

---

## Core Parameters

### Primary Cover Parameters

| Parameter | Type | Default | API Server Support | Description |
|-----------|------|---------|-------------------|-------------|
| `task_type` | `str` | `"text2music"` | ✅ Yes | Must be set to `"cover"` for cover functionality |
| `src_audio` / `src_audio_path` | `str` | `None` | ✅ Yes | **Required** - Path to the source audio file to cover |
| `caption` | `str` | `""` | ✅ Yes | Description of desired style/transformation |
| `audio_cover_strength` | `float` | `1.0` | ✅ Yes | Strength of source audio influence (0.0-1.0) |
| `audio_code_string` | `str` | `""` | ✅ Yes | Pre-extracted 5Hz audio semantic codes (alternative to src_audio) |

### Secondary Parameters That Affect Cover

| Parameter | Type | Default | API Server Support | Impact on Cover |
|-----------|------|---------|-------------------|-----------------|
| `instruction` | `str` | Auto-generated | ✅ Yes | Auto-set to `"Generate audio semantic tokens based on the given conditions:"` |
| `lyrics` | `str` | `""` | ✅ Yes | New lyrics to use (if changing vocals) |
| `bpm` | `int` | `None` | ✅ Yes | Tempo hint - can help maintain or change rhythm |
| `keyscale` | `str` | `""` | ✅ Yes | Musical key hint - helps with harmonic consistency |
| `vocal_language` | `str` | `"en"` | ✅ Yes | Language for vocals in the cover |
| `audio_duration` | `float` | `None` | ✅ Yes | Target duration (defaults to source audio length) |
| `inference_steps` | `int` | `8` | ✅ Yes | Quality/speed tradeoff |
| `guidance_scale` | `float` | `7.0` | ✅ Yes | How strongly to follow the caption (base model only) |
| `thinking` | `bool` | `False` | ✅ Yes | Enable 5Hz LM for enhanced generation |

### Generation Parameters

| Parameter | Type | Default | API Server Support | Description |
|-----------|------|---------|-------------------|-------------|
| `seed` | `int` | `-1` | ✅ Yes | Random seed for reproducibility |
| `use_random_seed` | `bool` | `True` | ✅ Yes | Whether to use random seed |
| `batch_size` | `int` | `2` | ✅ Yes | Number of cover variations to generate |
| `audio_format` | `str` | `"mp3"` | ✅ Yes | Output format (mp3, wav, flac) |

### Advanced DiT Parameters

| Parameter | Type | Default | API Server Support | Description |
|-----------|------|---------|-------------------|-------------|
| `shift` | `float` | `3.0` | ✅ Yes | Timestep shift factor (1.0-5.0) |
| `infer_method` | `str` | `"ode"` | ✅ Yes | Diffusion method: `"ode"` or `"sde"` |
| `timesteps` | `str` | `None` | ✅ Yes | Custom timesteps (comma-separated) |
| `use_adg` | `bool` | `False` | ✅ Yes | Adaptive Dual Guidance (base model only) |
| `cfg_interval_start` | `float` | `0.0` | ✅ Yes | CFG start ratio |
| `cfg_interval_end` | `float` | `1.0` | ✅ Yes | CFG end ratio |

---

## How Cover Works (Technical Deep Dive)

### Step 1: Source Audio Processing

When a cover task is triggered, the source audio is processed and converted to semantic codes:

```python
# From handler.py - determine_task_type()
is_cover_task = (task_type == "cover")

# If audio codes are provided, it's automatically treated as cover
has_codes = bool(audio_code_string and str(audio_code_string).strip())
if has_codes:
    is_cover_task = True
```

The source audio is:
1. Loaded and normalized to **stereo 48kHz** format
2. Encoded to latent representations using the **VAE (Variational Autoencoder)**
3. Tokenized to obtain semantic code indices

### Step 2: Audio Code Extraction

Source audio is converted to 5Hz semantic codes that capture structural information:

```python
# From handler.py - convert_src_audio_to_codes()
def convert_src_audio_to_codes(self, audio_file) -> str:
    # Process audio file
    processed_audio = self.process_src_audio(audio_file)
    
    # Encode audio to latents using VAE
    latents = self._encode_audio_to_latents(processed_audio)  # [T, d]
    
    # Tokenize latents to get code indices
    # Returns format: '<|audio_code_123|><|audio_code_456|>...'
```

**Key conversion details:**
- Audio sample rate: 48000 Hz
- Latent frame rate: 48000 / 1920 = 25 Hz
- Codes are at 5Hz (1 code per 200ms)
- Codes capture: melody, rhythm, chords, orchestration, and partial timbre

### Step 3: Cover Task Instruction

When cover task is detected, the instruction is automatically set:

```python
# From constants.py
TASK_INSTRUCTIONS = {
    "cover": "Generate audio semantic tokens based on the given conditions:",
    # ...
}
```

### Step 4: Audio Cover Strength Control

The `audio_cover_strength` parameter controls how many denoising steps use the cover mode:

```python
# From handler.py - _prepare_batch()
if audio_cover_strength < 1.0:
    # When strength < 1.0, prepare non-cover text inputs for blending
    # This allows gradual transition between cover and text2music modes
    non_cover_text_input_ids = []
    non_cover_text_attention_masks = []
    # ...
```

**How it works:**
- `1.0`: Strong adherence to original structure (full cover mode)
- `0.5`: Balanced transformation (50% cover, 50% text-guided)
- `0.2`: Loose interpretation (style transfer mode)
- `0.0`: Pure text2music (source audio ignored)

### Step 5: Padding Behavior for Cover

Unlike repaint tasks, cover tasks use source audio directly without padding:

```python
# From handler.py - prepare_padding_info()
if is_cover_task:
    # Cover task: Use src_audio directly without padding
    batch_target_wavs = processed_src_audio
    padding_info_batch.append({
        'left_padding_duration': 0.0,
        'right_padding_duration': 0.0
    })
```

### Step 6: DiT Generation with Cover Conditioning

The DiT model generates audio conditioned on:
- Text embeddings (from caption)
- Lyric embeddings (if provided)
- Source audio latents (from cover audio)
- LM-generated hints (if thinking=True)

```python
# From handler.py - service_generate()
generate_kwargs = {
    "src_latents": src_latents,
    "is_covers": is_covers,
    "audio_cover_strength": audio_cover_strength,
    "precomputed_lm_hints_25Hz": precomputed_lm_hints_25Hz,
    # ...
}
```

---

## Parameter Flow Through the System

### 1. API Server Entry Point

```python
# From api_server.py - GenerateMusicRequest
class GenerateMusicRequest(BaseModel):
    src_audio_path: Optional[str] = None
    audio_code_string: str = ""
    audio_cover_strength: float = 1.0
    task_type: str = "text2music"
    # ...
```

### 2. Request Parsing

The API server supports multiple parameter naming conventions:

```python
# From api_server.py - _PARAM_ALIASES
"audio_cover_strength": ["audio_cover_strength", "audioCoverStrength"],
```

### 3. GenerationParams Construction

```python
# From api_server.py - _blocking_generate()
params = GenerationParams(
    task_type=req.task_type,
    src_audio=req.src_audio_path,
    audio_codes=req.audio_code_string,
    audio_cover_strength=req.audio_cover_strength,
    # ...
)
```

### 4. Handler Processing

```python
# From inference.py - generate_music()
result = dit_handler.generate_music(
    src_audio=params.src_audio,
    audio_code_string=audio_code_string_to_use,
    audio_cover_strength=params.audio_cover_strength,
    task_type=params.task_type,
    # ...
)
```

---

## API Usage

### REST API Example: Basic Cover

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -F "caption=jazz piano version" \
  -F "src_audio=@/path/to/original_song.mp3" \
  -F "task_type=cover" \
  -F "audio_cover_strength=0.8"
```

### REST API Example: Cover with New Lyrics

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "caption": "reggae version with tropical vibes",
    "lyrics": "[Verse 1]\nNew lyrics here...",
    "src_audio_path": "/path/to/source.mp3",
    "task_type": "cover",
    "audio_cover_strength": 0.7,
    "thinking": true
  }'
```

### REST API Example: Style Transfer (Low Cover Strength)

```bash
curl -X POST http://localhost:8001/v1/music/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "caption": "orchestral symphonic arrangement",
    "src_audio_path": "/path/to/pop_song.mp3",
    "task_type": "cover",
    "audio_cover_strength": 0.2
  }'
```

### Python API Example

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Initialize handlers
dit_handler = AceStepHandler()
llm_handler = LLMHandler()

# Initialize services
dit_handler.initialize_service(
    project_root="/path/to/project",
    config_path="acestep-v15-turbo",
    device="cuda"
)

# Configure cover generation
params = GenerationParams(
    task_type="cover",
    src_audio="original_song.mp3",
    caption="jazz piano version with smooth harmonies",
    audio_cover_strength=0.8,
    thinking=True,
)

config = GenerationConfig(
    batch_size=2,
    audio_format="flac",
)

# Generate cover
result = generate_music(dit_handler, llm_handler, params, config, save_dir="/output")

if result.success:
    for audio in result.audios:
        print(f"Generated: {audio['path']}")
```

---

## Gradio UI Implementation

### Task Type Selection

The cover task is available in the Gradio UI through the task type dropdown:

```python
# Available task types for turbo models
TASK_TYPES_TURBO = ["text2music", "repaint", "cover"]
```

### Audio Cover Strength Slider

The slider appears when cover task is selected:

```python
# From gradio_ui/interfaces/generation.py
audio_cover_strength = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    value=1.0,
    step=0.01,
    label="Audio Cover Strength",
    info="Control how many denoising steps use cover mode",
)
```

### Dynamic Label Changes

When LM is initialized (for text2music with codes), the label changes:

```python
# From gradio_ui/events/generation_handlers.py
def update_audio_cover_strength_visibility(task_type_value, init_llm_checked):
    if init_llm_checked and task_type_value != "cover":
        label = "LM codes strength"
        info = "Control how many denoising steps use LM-generated codes"
    else:
        label = "Audio Cover Strength"
        info = "Control how many denoising steps use cover mode"
```

---

## Impact on Generated Audio

### What Source Audio Controls

Source audio in cover tasks controls **semantic structure information**:

| Aspect | Controlled By | Strength Dependence |
|--------|---------------|---------------------|
| **Melody** | Source audio codes | Higher strength = closer melody |
| **Rhythm** | Source audio codes | Higher strength = same groove |
| **Chords** | Source audio codes | Higher strength = same progressions |
| **Orchestration** | Source audio codes | Higher strength = similar arrangement |
| **Partial Timbre** | Source audio codes | Moderate influence |
| **Style/Genre** | Caption | Lower strength = more caption influence |
| **New Timbre** | Caption | Lower strength = more style change |

### Audio Cover Strength Effects

| Strength | Behavior | Use Case |
|----------|----------|----------|
| `1.0` | Strong structure adherence | Faithful cover versions |
| `0.7-0.9` | Balanced transformation | Genre change with structure |
| `0.4-0.6` | Creative interpretation | Loose remixes |
| `0.1-0.3` | Style transfer | Dramatic reimagining |
| `0.0` | Pure text2music | Source audio ignored |

### Auto-Detection Behavior

When `audio_code_string` is provided with text2music task, it automatically switches to cover:

```python
# From handler.py - generate_music()
if task_type == "text2music":
    if _has_audio_codes(audio_code_string):
        # User has provided audio codes, switch to cover task
        task_type = "cover"
        instruction = TASK_INSTRUCTIONS["cover"]
```

---

## Advanced Use Cases

### 1. Remix Creation

Change style while maintaining structure:

```python
params = GenerationParams(
    task_type="cover",
    src_audio="pop_song.mp3",
    caption="electronic dance remix with heavy bass drops",
    audio_cover_strength=0.6,  # Balanced for creative freedom
)
```

### 2. Genre Transformation

Transform a song from one genre to another:

```python
params = GenerationParams(
    task_type="cover",
    src_audio="rock_song.mp3",
    caption="jazz trio arrangement with piano, upright bass, and drums",
    audio_cover_strength=0.7,
)
```

### 3. Vocal Replacement

Keep melody, change lyrics and vocal style:

```python
params = GenerationParams(
    task_type="cover",
    src_audio="original_vocal_track.mp3",
    caption="powerful female vocal performance",
    lyrics="""[Verse 1]
New lyrics replacing the original...
    """,
    audio_cover_strength=0.8,
    vocal_language="en",
)
```

### 4. Instrumental Version

Create instrumental from vocal track:

```python
params = GenerationParams(
    task_type="cover",
    src_audio="song_with_vocals.mp3",
    caption="instrumental version with strings and piano",
    lyrics="[Instrumental]",
    audio_cover_strength=0.7,
)
```

### 5. Style Transfer (Low Strength)

Dramatic reinterpretation:

```python
params = GenerationParams(
    task_type="cover",
    src_audio="classical_piece.mp3",
    caption="heavy metal arrangement with distorted guitars",
    audio_cover_strength=0.2,  # Low strength for more freedom
)
```

### 6. Using Pre-Extracted Audio Codes

For faster processing with pre-computed codes:

```python
# First, extract codes from source audio
codes = dit_handler.convert_src_audio_to_codes("source.mp3")

# Then use codes for cover generation
params = GenerationParams(
    task_type="cover",
    audio_codes=codes,  # Use codes instead of src_audio
    caption="acoustic guitar version",
    audio_cover_strength=0.8,
)
```

---

## Comparison with Other Tasks

### Cover vs Repaint

| Aspect | Cover | Repaint |
|--------|-------|---------|
| **Purpose** | Transform entire audio | Regenerate specific section |
| **Input** | Source audio or codes | Source audio + time range |
| **Output** | Full new version | Modified version |
| **Structure** | Maintains global structure | Maintains surrounding context |
| `repainting_start/end` | Not used | Required |
| `audio_cover_strength` | Used (0.0-1.0) | Not used |
| **LM Behavior** | LM skipped for task | LM skipped for task |

### Cover vs Text2Music

| Aspect | Cover | Text2Music |
|--------|-------|------------|
| **Purpose** | Transform existing audio | Generate from scratch |
| **Source Audio** | Required | Not used |
| **Structure Control** | From source audio | From text/LM |
| `audio_cover_strength` | Controls structure adherence | N/A (or controls LM codes) |
| **Use Case** | Remixes, covers, style transfer | Original compositions |

### Cover vs Lego/Complete (Base Model)

| Aspect | Cover | Lego/Complete |
|--------|-------|---------------|
| **Purpose** | Full transformation | Add/complete tracks |
| **Output** | Complete new version | Layered composition |
| **Track Control** | N/A | Specific track types |
| **Model Support** | Turbo + Base | Base only |

---

## LM Behavior for Cover

From `acestep/inference.py`:

```python
# LM is skipped for cover/repaint tasks - these use source/codes directly
skip_lm_tasks = {"cover", "repaint"}

if params.task_type in skip_lm_tasks:
    logger.info(f"Skipping LM for task_type='{params.task_type}' - using DiT directly")
```

The 5Hz Language Model is **not used for code generation** in cover tasks because:
- Audio codes are derived from source audio
- Structure comes from source, not LM generation

However, LM **may still be used for**:
- Metadata completion (BPM, key, duration) when missing
- Caption enhancement (if `use_cot_caption=True`)
- Language detection (if `use_cot_language=True`)

---

## Best Practices

### 1. Choosing Audio Cover Strength

| Goal | Recommended Strength |
|------|---------------------|
| Faithful cover | 0.8-1.0 |
| Genre change | 0.6-0.8 |
| Creative remix | 0.4-0.6 |
| Style transfer | 0.1-0.3 |

### 2. Caption Writing for Cover

**Good captions for cover:**
```
# Specific style transformation
"jazz trio version with piano, upright bass, and brush drums"

# Clear genre indication
"80s synthwave interpretation with analog synthesizers"

# Mood and instrumentation
"acoustic ballad version with gentle guitar and soft vocals"
```

**Avoid:**
```
# Too vague
"different version"

# Contradictory
"same style but different"
```

### 3. Generating Multiple Variations

Use batch generation to explore different interpretations:

```python
config = GenerationConfig(
    batch_size=4,          # Generate 4 variations
    use_random_seed=True,  # Different seeds for variety
)
```

### 4. Reproducible Covers

For consistent results:

```python
config = GenerationConfig(
    batch_size=1,
    use_random_seed=False,
    seeds=[42],  # Fixed seed
)
```

### 5. Combining with Reference Audio

For style reference without structure control:

```python
params = GenerationParams(
    task_type="cover",
    src_audio="song_for_structure.mp3",
    reference_audio="song_for_style.mp3",  # Acoustic timbre reference
    caption="cover with similar style to reference",
    audio_cover_strength=0.7,
)
```

---

## Troubleshooting

### Common Issues

**Issue**: Cover sounds too similar to original
- **Solution**: Lower `audio_cover_strength` to 0.5-0.7

**Issue**: Cover loses original structure
- **Solution**: Increase `audio_cover_strength` to 0.8-1.0

**Issue**: Style doesn't match caption
- **Solution**: Lower `audio_cover_strength`, make caption more specific

**Issue**: Poor quality output
- **Solution**: Increase `inference_steps`, try different seeds

**Issue**: Vocals don't match new lyrics
- **Solution**: Use `audio_cover_strength` of 0.6-0.7, ensure lyrics are formatted correctly

---

## API Server Parameter Reference

### Complete Cover Parameters Supported by API Server

| Parameter | Type | Default | Supported | Notes |
|-----------|------|---------|-----------|-------|
| `task_type` | str | `"text2music"` | ✅ Yes | Set to `"cover"` |
| `src_audio_path` | str | `None` | ✅ Yes | Server-side path |
| `src_audio` | File | `None` | ✅ Yes | File upload (multipart/form-data) |
| `audio_code_string` | str | `""` | ✅ Yes | Pre-extracted codes |
| `audio_cover_strength` | float | `1.0` | ✅ Yes | 0.0-1.0 |
| `caption` | str | `""` | ✅ Yes | Style description |
| `lyrics` | str | `""` | ✅ Yes | New lyrics |
| `vocal_language` | str | `"en"` | ✅ Yes | Language code |
| `bpm` | int | `None` | ✅ Yes | 30-300 |
| `key_scale` | str | `""` | ✅ Yes | Musical key |
| `time_signature` | str | `""` | ✅ Yes | Time signature |
| `audio_duration` | float | `None` | ✅ Yes | 10-600 seconds |
| `inference_steps` | int | `8` | ✅ Yes | Turbo: 1-20 |
| `guidance_scale` | float | `7.0` | ✅ Yes | Base model only |
| `seed` | int | `-1` | ✅ Yes | Random seed |
| `use_random_seed` | bool | `True` | ✅ Yes | Use random seed |
| `batch_size` | int | `2` | ✅ Yes | 1-8 |
| `audio_format` | str | `"mp3"` | ✅ Yes | mp3/wav/flac |
| `shift` | float | `3.0` | ✅ Yes | Timestep shift |
| `infer_method` | str | `"ode"` | ✅ Yes | ode/sde |
| `timesteps` | str | `None` | ✅ Yes | Custom timesteps |
| `thinking` | bool | `False` | ✅ Yes | Enable LM enhancement |
| `model` | str | `None` | ✅ Yes | Model selection |

---

## Summary

The **Cover** functionality in ACE-Step provides powerful audio transformation capabilities by:

1. **Extracting** semantic structure from source audio (melody, rhythm, chords)
2. **Conditioning** generation on source audio codes
3. **Blending** source structure with caption-guided style through `audio_cover_strength`
4. **Generating** new audio that maintains structure but changes style/timbre

Key parameters:
- `task_type`: Must be `"cover"`
- `src_audio`: Source audio for structure reference
- `caption`: Description of desired style/transformation
- `audio_cover_strength`: Controls structure adherence (0.0-1.0)
- `lyrics`: Optional new lyrics for vocal changes

This enables use cases like genre transformation, remixing, vocal replacement, instrumental versions, and creative reinterpretation of existing music.
