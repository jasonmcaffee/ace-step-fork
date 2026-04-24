# Lego Audio Endpoint Documentation

## Overview

The `/lego` endpoint generates a specific audio track (vocals, drums, guitar, etc.) based on existing audio context. This is the "Lego" task from ACE-Step - it intelligently adds new tracks that match the rhythm, harmony, and style of the input audio.

**Use Cases:**
- Add vocals to an instrumental track
- Add drums to a guitar recording
- Layer bass onto existing music
- Add orchestral strings to a piano piece
- Build multi-track compositions iteratively

---

## Endpoint Details

**URL:** `POST /lego`

**Content-Type:** `multipart/form-data`

**Authentication:** Optional API key via `Authorization` header

---

## Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_file` | file | **Yes** | - | Context audio file (mp3, wav, flac) |
| `track_name` | string | **Yes** | - | Track type to generate (see options below) |
| `caption` | string | No | auto | Description for the generated track style |
| `lyrics` | string | No | "" | Lyrics text (for vocals/backing_vocals tracks) |
| `vocal_language` | string | No | "en" | Language code for vocals |
| `bpm` | int | No | auto | Tempo (auto-detected from audio if not provided) |
| `key_scale` | string | No | auto | Musical key (e.g., "C Major", "Am") |
| `time_signature` | string | No | auto | Time signature (e.g., "4", "3") |
| `repainting_start` | float | No | 0.0 | Start time in seconds for track generation |
| `repainting_end` | float | No | -1 | End time in seconds (-1 = full audio length) |
| `inference_steps` | int | No | 50 | Diffusion steps (higher = better quality, slower) |
| `guidance_scale` | float | No | 7.0 | CFG strength (higher = more prompt adherence) |
| `seed` | int | No | -1 | Random seed (-1 = random) |
| `audio_format` | string | No | "mp3" | Output format: "mp3", "wav", "flac" |
| `batch_size` | int | No | 1 | Number of variations to generate (1-8) |

---

## Available Track Names

| Track Name | Description |
|------------|-------------|
| `vocals` | Main vocals / lead singer |
| `backing_vocals` | Background harmonies, choir, ad-libs |
| `drums` | Drum kit, electronic drums |
| `bass` | Bass guitar, synth bass |
| `guitar` | Electric or acoustic guitar |
| `keyboard` | Piano, organ, synth pads |
| `percussion` | Hand drums, shakers, tambourine |
| `strings` | Orchestral strings (violin, cello, etc.) |
| `synth` | Synthesizers, electronic sounds |
| `fx` | Sound effects, ambient textures |
| `brass` | Horns, trumpets, trombone |
| `woodwinds` | Flute, saxophone, clarinet |

---

## Response Format

```json
{
    "data": {
        "audios": [
            {
                "audio_data": "data:audio/mpeg;base64,//uQxAAA...",
                "audio_format": "mp3"
            },
            {
                "audio_data": "data:audio/mpeg;base64,//uQyBBB...",
                "audio_format": "mp3"
            }
        ],
        "track_name": "vocals",
        "batch_size": 2,
        "duration": 30.0,
        "repainting_start": 0.0,
        "repainting_end": 30.0,
        "instruction": "Generate the VOCALS track based on the audio context:",
        "status_message": "✅ 2 VOCALS track(s) generated successfully"
    },
    "code": 200,
    "error": null
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `audios` | array | Array of generated audio objects |
| `audios[].audio_data` | string | Base64-encoded audio as data URL |
| `audios[].audio_format` | string | Output format ("mp3", "wav", "flac") |
| `track_name` | string | The track type that was generated |
| `batch_size` | int | Number of variations generated |
| `duration` | float | Duration in seconds |
| `repainting_start` | float | Start time of generation region |
| `repainting_end` | float | End time of generation region |
| `instruction` | string | The instruction used for generation |
| `status_message` | string | Human-readable status |

### Error Response

```json
{
    "data": null,
    "code": 400,
    "error": "Invalid track_name: 'invalid'. Valid options: woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums, backing_vocals, vocals"
}
```

---

## Examples

### cURL

**Add vocals to instrumental:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@instrumental.mp3" \
  -F "track_name=vocals" \
  -F "caption=emotional female vocals, soft and breathy" \
  -F "lyrics=[Verse]
In the morning light I see your face
Every shadow fades without a trace

[Chorus]
Hold me closer through the night
Everything will be alright" \
  -F "vocal_language=en" \
  -F "audio_format=mp3"
```

**Add drums:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@guitar_track.mp3" \
  -F "track_name=drums" \
  -F "caption=groovy rock drums with tight hi-hats and punchy kick"
```

**Add bass:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@rhythm_section.mp3" \
  -F "track_name=bass" \
  -F "caption=deep funky bass line, slap bass style"
```

**Add strings:**
```bash
curl -X POST http://localhost:8001/lego \
  -F "audio_file=@piano_ballad.mp3" \
  -F "track_name=strings" \
  -F "caption=lush orchestral strings, emotional swells, cinematic"
```

---

### TypeScript

```typescript
interface LegoRequest {
  audioFile: File | Blob;
  trackName: string;
  caption?: string;
  lyrics?: string;
  vocalLanguage?: string;
  bpm?: number;
  keyScale?: string;
  timeSignature?: string;
  repaintingStart?: number;
  repaintingEnd?: number;
  inferenceSteps?: number;
  guidanceScale?: number;
  seed?: number;
  audioFormat?: "mp3" | "wav" | "flac";
  batchSize?: number;  // Number of variations (1-8)
}

interface AudioResult {
  audio_data: string;      // Base64 data URL
  audio_format: string;
}

interface LegoResponse {
  data: {
    audios: AudioResult[];  // Array of generated audio variations
    track_name: string;
    batch_size: number;
    duration: number;
    repainting_start: number;
    repainting_end: number;
    instruction: string;
    status_message: string;
  } | null;
  code: number;
  error: string | null;
}

async function generateTrack(request: LegoRequest): Promise<LegoResponse> {
  const formData = new FormData();
  
  // Required fields
  formData.append("audio_file", request.audioFile);
  formData.append("track_name", request.trackName);
  
  // Optional fields
  if (request.caption) formData.append("caption", request.caption);
  if (request.lyrics) formData.append("lyrics", request.lyrics);
  if (request.vocalLanguage) formData.append("vocal_language", request.vocalLanguage);
  if (request.bpm) formData.append("bpm", request.bpm.toString());
  if (request.keyScale) formData.append("key_scale", request.keyScale);
  if (request.timeSignature) formData.append("time_signature", request.timeSignature);
  if (request.repaintingStart !== undefined) formData.append("repainting_start", request.repaintingStart.toString());
  if (request.repaintingEnd !== undefined) formData.append("repainting_end", request.repaintingEnd.toString());
  if (request.inferenceSteps) formData.append("inference_steps", request.inferenceSteps.toString());
  if (request.guidanceScale) formData.append("guidance_scale", request.guidanceScale.toString());
  if (request.seed !== undefined) formData.append("seed", request.seed.toString());
  if (request.audioFormat) formData.append("audio_format", request.audioFormat);
  if (request.batchSize) formData.append("batch_size", request.batchSize.toString());

  const response = await fetch("http://localhost:8001/lego", {
    method: "POST",
    body: formData,
  });

  return response.json();
}

// Helper to convert base64 data URL to Blob
function dataUrlToBlob(dataUrl: string): Blob {
  const [header, base64] = dataUrl.split(",");
  const mimeMatch = header.match(/data:([^;]+)/);
  const mimeType = mimeMatch ? mimeMatch[1] : "audio/mpeg";
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}

// Helper to create downloadable URL
function createAudioUrl(dataUrl: string): string {
  const blob = dataUrlToBlob(dataUrl);
  return URL.createObjectURL(blob);
}

// ============================================
// Usage Examples
// ============================================

// Example 1: Add vocals to an instrumental (single variation)
async function addVocals(instrumentalFile: File) {
  const result = await generateTrack({
    audioFile: instrumentalFile,
    trackName: "vocals",
    caption: "warm female vocals, indie folk style",
    lyrics: `[Verse]
Walking down the road alone
Searching for a place called home

[Chorus]
But I know, I know
The light will find me`,
    vocalLanguage: "en",
  });

  if (result.code === 200 && result.data) {
    console.log(result.data.status_message);
    
    // Play the first audio
    const audioUrl = createAudioUrl(result.data.audios[0].audio_data);
    const audio = new Audio(audioUrl);
    audio.play();
    
    // Or download it
    const link = document.createElement("a");
    link.href = audioUrl;
    link.download = "with_vocals.mp3";
    link.click();
  } else {
    console.error("Error:", result.error);
  }
}

// Example 2: Generate multiple drum variations
async function addDrumsWithVariations(guitarFile: File) {
  const result = await generateTrack({
    audioFile: guitarFile,
    trackName: "drums",
    caption: "tight rock drums, punchy kick, crisp snare",
    batchSize: 4,  // Generate 4 variations to choose from
    inferenceSteps: 50,
    guidanceScale: 7.0,
  });

  if (result.code === 200 && result.data) {
    console.log(`Generated ${result.data.batch_size} ${result.data.track_name} variations`);
    
    // Process all variations
    result.data.audios.forEach((audio, index) => {
      console.log(`Variation ${index + 1}: ${audio.audio_format}`);
      // Save or play each variation...
    });
    
    return result.data.audios;
  }
  throw new Error(result.error || "Generation failed");
}

// Example 3: Build a full track iteratively
async function buildFullTrack(initialFile: File) {
  // Start with guitar
  let currentAudio: File | Blob = initialFile;
  
  // Add drums
  const withDrums = await generateTrack({
    audioFile: currentAudio,
    trackName: "drums",
    caption: "groovy indie drums",
  });
  
  if (withDrums.code !== 200 || !withDrums.data) {
    throw new Error("Failed to add drums");
  }
  
  // Convert first result to blob for next iteration
  currentAudio = dataUrlToBlob(withDrums.data.audios[0].audio_data);
  
  // Add bass
  const withBass = await generateTrack({
    audioFile: currentAudio,
    trackName: "bass",
    caption: "warm bass line, following the root notes",
  });
  
  if (withBass.code !== 200 || !withBass.data) {
    throw new Error("Failed to add bass");
  }
  
  currentAudio = dataUrlToBlob(withBass.data.audios[0].audio_data);
  
  // Add vocals
  const withVocals = await generateTrack({
    audioFile: currentAudio,
    trackName: "vocals",
    caption: "emotional male vocals",
    lyrics: "[Verse]\nYour lyrics here...",
  });
  
  return withVocals;
}

// Example 4: Using with file input element
document.getElementById("audioInput")?.addEventListener("change", async (e) => {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  const trackSelect = document.getElementById("trackSelect") as HTMLSelectElement;
  const trackName = trackSelect.value;

  try {
    const result = await generateTrack({
      audioFile: file,
      trackName: trackName,
      audioFormat: "mp3",
      batchSize: 2,  // Generate 2 variations
    });

    if (result.code === 200 && result.data) {
      // Play first variation
      const audioPlayer = document.getElementById("audioPlayer") as HTMLAudioElement;
      audioPlayer.src = createAudioUrl(result.data.audios[0].audio_data);
      audioPlayer.play();
      
      // Show variation count
      console.log(`Generated ${result.data.batch_size} variations`);
    }
  } catch (error) {
    console.error("Failed to generate track:", error);
  }
});
```

---

### Python

```python
import requests
import base64
from pathlib import Path

def generate_track(
    audio_file: str | Path,
    track_name: str,
    caption: str = "",
    lyrics: str = "",
    vocal_language: str = "en",
    audio_format: str = "mp3",
    batch_size: int = 1,
    **kwargs
) -> dict:
    """
    Generate a specific track using the /lego endpoint.
    
    Args:
        audio_file: Path to the context audio file
        track_name: Track to generate (vocals, drums, bass, etc.)
        caption: Description for the track style
        lyrics: Lyrics text (for vocals)
        vocal_language: Language code for vocals
        audio_format: Output format (mp3, wav, flac)
        batch_size: Number of variations to generate (1-8)
        **kwargs: Additional parameters (bpm, key_scale, seed, etc.)
    
    Returns:
        Response dict with audios array, track_name, status_message, etc.
    """
    url = "http://localhost:8001/lego"
    
    with open(audio_file, "rb") as f:
        files = {"audio_file": (Path(audio_file).name, f, "audio/mpeg")}
        data = {
            "track_name": track_name,
            "caption": caption,
            "lyrics": lyrics,
            "vocal_language": vocal_language,
            "audio_format": audio_format,
            "batch_size": str(batch_size),
            **{k: str(v) for k, v in kwargs.items() if v is not None}
        }
        response = requests.post(url, files=files, data=data)
    
    return response.json()


def save_audios_from_response(response: dict, output_prefix: str | Path) -> list[str]:
    """Save all audio variations from response to files.
    
    Returns list of saved file paths.
    """
    if response.get("code") != 200 or not response.get("data"):
        print(f"Error: {response.get('error')}")
        return []
    
    saved_files = []
    audios = response["data"]["audios"]
    
    for i, audio in enumerate(audios):
        audio_data = audio["audio_data"]
        audio_format = audio["audio_format"]
        
        # Remove data URL prefix: "data:audio/mpeg;base64,"
        base64_audio = audio_data.split(",")[1]
        audio_bytes = base64.b64decode(base64_audio)
        
        # Generate filename with variation number
        if len(audios) == 1:
            output_path = f"{output_prefix}.{audio_format}"
        else:
            output_path = f"{output_prefix}_v{i+1}.{audio_format}"
        
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        saved_files.append(output_path)
    
    print(f"Saved {len(saved_files)} file(s): {response['data']['status_message']}")
    return saved_files


# Example usage
if __name__ == "__main__":
    # Add vocals to instrumental (single variation)
    result = generate_track(
        audio_file="instrumental.mp3",
        track_name="vocals",
        caption="emotional female vocals, intimate and breathy",
        lyrics="""[Verse]
In the quiet of the night
Stars are shining oh so bright

[Chorus]
Hold me close and never let go
This is all I need to know""",
        vocal_language="en",
    )
    save_audios_from_response(result, "with_vocals")
    
    # Add drums with multiple variations
    result = generate_track(
        audio_file="guitar_only.mp3",
        track_name="drums",
        caption="tight rock drums with crisp hi-hats",
        batch_size=4,  # Generate 4 variations
    )
    saved = save_audios_from_response(result, "with_drums")
    # Creates: with_drums_v1.mp3, with_drums_v2.mp3, with_drums_v3.mp3, with_drums_v4.mp3
```

---

## Notes

### Model Requirements

The lego task works best with the **Base model**. While it may work with turbo models, the Base model provides better quality for track separation and generation tasks.

To use the Base model, start the server with:
```bash
set ACESTEP_CONFIG_PATH=acestep-v15-base
```

### Caption Tips for Different Tracks

| Track | Example Captions |
|-------|------------------|
| `vocals` | "warm female vocals, intimate", "powerful male rock vocals", "breathy indie vocals" |
| `drums` | "tight rock drums, punchy kick", "jazz brushes, subtle swing", "electronic 808 drums" |
| `bass` | "deep funk bass, slap style", "smooth jazz bass", "synth bass, sub-heavy" |
| `guitar` | "clean electric arpeggios", "distorted power chords", "fingerpicked acoustic" |
| `strings` | "lush orchestral strings", "emotional cello melody", "pizzicato violins" |
| `keyboard` | "warm rhodes piano", "bright grand piano", "ambient synth pads" |

### Iterative Track Building

You can build complex arrangements by calling the endpoint multiple times:

1. Start with a simple recording (e.g., guitar)
2. Add drums → save result
3. Use result as input → add bass
4. Use result as input → add vocals
5. Continue layering...

Each iteration adds a new track that's synchronized with the existing audio.
