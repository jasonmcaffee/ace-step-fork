# Lyrics-to-Vocals Generation Research

## Overview

This document explores methods for generating **vocals only** (without instrumental accompaniment) from lyrics using ACE-Step. While ACE-Step is primarily designed for full music generation, there are several approaches to achieve vocal-focused output.

## Key Concepts

### The Challenge

ACE-Step's DiT model generates complete audio tracks including vocals AND instrumentation. Unlike dedicated text-to-speech singing (TTS-S) systems that synthesize isolated vocal performances, ACE-Step works at the music composition level.

**What ACE-Step CAN do:**
- Generate music with prominent vocals
- Extract vocals from existing mixes (Base model)
- Generate vocal tracks in context of backing tracks (Base model)
- Create acapella-style output with minimal accompaniment

**What ACE-Step CANNOT do natively:**
- True isolated vocal synthesis (dry vocal stem)
- Speech-to-singing conversion
- Precise phoneme-level vocal control

---

## Available Methods

### Method 1: Text2Music with Acapella Caption (Easiest)

Use the standard `text2music` task with captions emphasizing vocal-only output.

**Works with:** All models (Turbo, SFT, Base)

```python
from acestep.inference import GenerationParams, GenerationConfig, generate_music

params = GenerationParams(
    task_type="text2music",
    caption="acapella vocals only, no instruments, solo female singer, clear voice, intimate recording, dry vocals",
    lyrics="""[Verse 1]
Walking through the empty streets
Shadows dancing at my feet
Memories of what we had
Now they only make me sad

[Chorus]
But I'll keep on moving
Through the night
I'll keep believing
In the light""",
    vocal_language="en",
    bpm=80,
    duration=60,
)

config = GenerationConfig(batch_size=2)
result = generate_music(dit_handler, llm_handler, params, config)
```

**Caption Tips for Acapella:**
- "acapella", "a cappella", "vocals only"
- "no instruments", "no accompaniment", "unaccompanied"
- "solo voice", "solo singer"
- "dry vocals", "intimate recording"
- Avoid mentioning any instruments

**Limitations:**
- Model may still add subtle ambient textures or minimal accompaniment
- Results vary by seed - try multiple generations
- Works best for slower, intimate styles

---

### Method 2: Lego Task - Generate Vocal Track (Base Model)

Use the `lego` task to generate a vocals track in isolation. This requires the **Base model**.

**Works with:** Base model only

```python
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# First, create or use a minimal/silent source audio context
# Or use any backing track as context

params = GenerationParams(
    task_type="lego",
    src_audio="minimal_backing.mp3",  # Can be a simple beat or even silence
    instruction="Generate the VOCALS track based on the audio context:",
    caption="clear female vocals, emotional singing, intimate performance",
    lyrics="""[Verse]
In the quiet of the night
Stars are shining oh so bright
Every word I want to say
Whispered softly, fades away""",
    vocal_language="en",
    repainting_start=0.0,
    repainting_end=-1,  # Full length
)

config = GenerationConfig(batch_size=1)
result = generate_music(dit_handler, llm_handler, params, config)
```

**Available Track Names:**
| Track | Description |
|-------|-------------|
| `vocals` | Main vocals/lead singer |
| `backing_vocals` | Background harmonies, choir |
| `drums` | Drum kit, percussion patterns |
| `bass` | Bass guitar, synth bass |
| `guitar` | Electric/acoustic guitar |
| `keyboard` | Piano, organ, synth pads |
| `percussion` | Hand drums, shakers, etc. |
| `strings` | Orchestral strings |
| `synth` | Synthesizers |
| `fx` | Sound effects, ambient |
| `brass` | Horns, trumpets |
| `woodwinds` | Flute, saxophone, etc. |

**Note:** The `lego` task is designed to generate tracks **in context** of existing audio. For best results, provide at least a minimal backing track or rhythm reference.

---

### Method 3: Extract Vocals from Mix (Base Model)

If you have existing music and want to isolate vocals, use the `extract` task.

**Works with:** Base model only

```python
params = GenerationParams(
    task_type="extract",
    src_audio="full_song_mix.mp3",
    instruction="Extract the VOCALS track from the audio:",
)

config = GenerationConfig(batch_size=1)
result = generate_music(dit_handler, llm_handler, params, config)
```

**Use Cases:**
- Create vocal stems from mixed tracks
- Remove vocals for karaoke (extract instruments instead)
- Isolate specific instruments for remixing

---

### Method 4: Complete Task - Add Vocals to Backing (Base Model)

Use `complete` to add vocals to an existing instrumental track.

**Works with:** Base model only

```python
params = GenerationParams(
    task_type="complete",
    src_audio="instrumental_backing.mp3",
    instruction="Complete the input track with VOCALS:",
    caption="emotional male vocals, rock singing style",
    lyrics="""[Verse]
Standing at the edge of time
Watching as the stars align...""",
)
```

---

## API Endpoint: `/lego`

The `/lego` endpoint allows generating any specific track (vocals, drums, guitar, etc.) based on existing audio context.

**NOTE:** This task requires the **Base model** (not turbo). Start the server with `ACESTEP_CONFIG_PATH=acestep-v15-base`.

### Request Format (Multipart/form-data)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_file` | file | **Yes** | - | Context audio file to add track to |
| `track_name` | string | **Yes** | - | Track to generate (see options below) |
| `caption` | string | No | auto | Description for the generated track |
| `lyrics` | string | No | "" | Lyrics (for vocals/backing_vocals) |
| `vocal_language` | string | No | "en" | Language code for vocals |
| `bpm` | int | No | auto | Tempo (auto-detected from audio) |
| `key_scale` | string | No | auto | Musical key |
| `time_signature` | string | No | auto | Time signature |
| `repainting_start` | float | No | 0.0 | Start time for track generation |
| `repainting_end` | float | No | -1 | End time (-1 for full length) |
| `inference_steps` | int | No | 50 | Diffusion steps |
| `guidance_scale` | float | No | 7.0 | CFG strength |
| `seed` | int | No | -1 | Random seed |
| `audio_format` | string | No | "mp3" | Output format |

### Track Name Options

| Track | Description |
|-------|-------------|
| `vocals` | Main vocals/lead singer |
| `backing_vocals` | Background harmonies, choir |
| `drums` | Drum kit, percussion patterns |
| `bass` | Bass guitar, synth bass |
| `guitar` | Electric/acoustic guitar |
| `keyboard` | Piano, organ, synth pads |
| `percussion` | Hand drums, shakers, etc. |
| `strings` | Orchestral strings |
| `synth` | Synthesizers |
| `fx` | Sound effects, ambient |
| `brass` | Horns, trumpets |
| `woodwinds` | Flute, saxophone, etc. |

### Response Format

```json
{
    "data": {
        "audio_data": "data:audio/mp3;base64,//uQxAAA...",
        "audio_format": "mp3",
        "track_name": "vocals",
        "duration": 30.0,
        "repainting_start": 0.0,
        "repainting_end": 30.0,
        "instruction": "Generate the VOCALS track based on the audio context:",
        "status_message": "✅ VOCALS track generated successfully"
    },
    "code": 200,
    "error": null
}
```

### cURL Examples

**Add vocals to instrumental:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@instrumental_track.mp3" \
  -F "track_name=vocals" \
  -F "caption=emotional female vocals, soft and intimate" \
  -F "lyrics=[Verse]
In the morning light I see your face
Every shadow fades without a trace

[Chorus]
Hold me closer through the night
Everything will be alright" \
  -F "vocal_language=en"
```

**Add drums to guitar track:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@guitar_only.mp3" \
  -F "track_name=drums" \
  -F "caption=groovy rock drums with tight hi-hats"
```

**Add bass to existing mix:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@guitar_and_drums.mp3" \
  -F "track_name=bass" \
  -F "caption=deep funky bass line, slap bass style"
```

**Add strings for orchestral feel:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@piano_ballad.mp3" \
  -F "track_name=strings" \
  -F "caption=lush orchestral strings, emotional swells"
```

### Python Example

```python
import requests
import base64

url = "http://localhost:8001/lego"

with open("backing_track.mp3", "rb") as f:
    files = {"audio_file": ("backing_track.mp3", f, "audio/mpeg")}
    data = {
        "track_name": "vocals",
        "caption": "warm male vocals, indie folk style",
        "lyrics": "[Verse]\nWalking down the road...",
        "vocal_language": "en",
    }
    response = requests.post(url, files=files, data=data)

result = response.json()
if result["code"] == 200:
    # Decode and save the audio
    audio_data = result["data"]["audio_data"]
    # Remove data URL prefix
    base64_audio = audio_data.split(",")[1]
    audio_bytes = base64.b64decode(base64_audio)
    
    with open("output_with_vocals.mp3", "wb") as out:
        out.write(audio_bytes)
    print(f"Saved: {result['data']['status_message']}")
else:
    print(f"Error: {result['error']}")
```

---

## Quality Optimization

### Caption Engineering for Vocals

**High-quality acapella captions:**
```
"intimate acapella recording, solo female voice, crystal clear vocals, 
no instruments, no reverb, dry studio recording, emotional delivery"
```

**Avoid these terms:**
- "with piano" / "with guitar" (implies instruments)
- "band" / "orchestra" (implies full ensemble)
- "dance" / "pop" / "rock" (may trigger instrumental patterns)

### Vocal Style Descriptors

| Style | Description Keywords |
|-------|---------------------|
| Intimate | "whispered, soft, breathy, gentle, close mic" |
| Powerful | "belting, strong, chest voice, powerful, soaring" |
| Emotional | "heartfelt, emotive, passionate, expressive" |
| Clear | "crystal clear, pure tone, clean, precise" |
| Raspy | "raspy, husky, textured, gritty" |
| Falsetto | "falsetto, head voice, airy, light" |

### Structure Tags for Vocal Control

```text
[Intro - humming]
Mmm mmm mmm...

[Verse 1 - soft, whispered]
In the silence of the night
I hear your voice so clear

[Pre-Chorus - building]
Rising up, rising up

[Chorus - powerful, belting]
I WILL ALWAYS LOVE YOU!

[Bridge - spoken word]
And when I think about the time we shared...

[Outro - fading, soft]
Always... always...
```

---

## Model Comparison for Vocals

| Model | Acapella Quality | Extract Support | Lego Support |
|-------|-----------------|-----------------|--------------|
| Turbo | ⭐⭐⭐ | ❌ | ❌ |
| SFT | ⭐⭐⭐⭐ | ❌ | ❌ |
| Base | ⭐⭐⭐⭐⭐ | ✅ | ✅ |

**Recommendations:**
- **Quick vocal generation:** Use Turbo + acapella caption
- **Higher quality vocals:** Use SFT with more inference steps
- **True vocal isolation:** Use Base model with extract task
- **Vocals in context:** Use Base model with lego task

---

## Limitations & Considerations

### Current Limitations

1. **Not true TTS-S:** ACE-Step generates music, not isolated singing synthesis
2. **Instrumental bleed:** Even with acapella prompts, subtle accompaniment may appear
3. **Phoneme control:** No precise control over pronunciation or timing
4. **Vocal identity:** Cannot clone specific voices (no voice cloning feature)

### When to Use Alternative Tools

For these use cases, consider dedicated tools:

| Use Case | Recommended Tool |
|----------|------------------|
| Voice cloning | So-VITS-SVC, RVC |
| TTS singing | SunoAI Bark, Eleven Labs |
| Precise timing | MIDI + Synthesizer V |
| Karaoke tracks | Demucs, Spleeter |

---

## Future Enhancements

Potential improvements for vocals-only generation:

1. **Vocal-focused LoRA:** Fine-tune on acapella datasets
2. **Stem separation post-processing:** Auto-extract vocals after generation
3. **Vocal reference audio:** Use reference vocals for timbre matching
4. **Multi-voice support:** Generate harmonies and backing vocals

---

## Code Reference

### Task Type Constants

```python
# acestep/constants.py
TASK_TYPES = ["text2music", "repaint", "cover", "extract", "lego", "complete"]
TASK_TYPES_TURBO = ["text2music", "repaint", "cover"]  # Turbo models
TASK_TYPES_BASE = ["text2music", "repaint", "cover", "extract", "lego", "complete"]  # Base model

TRACK_NAMES = [
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"
]
```

### Instruction Templates

```python
TASK_INSTRUCTIONS = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "lego": "Generate the {TRACK_NAME} track based on the audio context:",
    "extract": "Extract the {TRACK_NAME} track from the audio:",
    "complete": "Complete the input track with {TRACK_CLASSES}:",
}
```

---

## Summary

| Method | Model Required | Best For |
|--------|---------------|----------|
| Text2Music + Acapella | Any | Quick vocal generation |
| Lego Task | Base | Vocals with backing context |
| Extract Task | Base | Isolating vocals from mixes |
| Complete Task | Base | Adding vocals to instrumentals |

The most practical approach for lyrics-to-vocals is using **text2music with carefully crafted acapella captions**. For professional stem isolation, the **Base model's extract task** provides the cleanest results.
