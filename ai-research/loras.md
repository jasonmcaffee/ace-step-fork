# ACE-Step v1.5 LoRA Research Document

## Overview

This document provides accurate information about LoRA (Low-Rank Adaptation) availability and usage in ACE-Step v1.5, based on actual research of HuggingFace and the official GitHub repository.

**Last Updated**: February 6, 2026

---

## Table of Contents

1. [Current LoRA Availability](#current-lora-availability)
2. [Downloaded LoRAs](#downloaded-loras)
3. [LoRA Architecture](#lora-architecture)
4. [How to Use LoRAs](#how-to-use-loras)
5. [Creating Custom LoRAs](#creating-custom-loras)
6. [Future LoRAs (Announced)](#future-loras-announced)
7. [v1 LoRAs (Legacy)](#v1-loras-legacy)

---

## Current LoRA Availability

### ACE-Step v1.5 LoRAs

| LoRA | HuggingFace ID | Status | Downloads |
|------|---------------|--------|-----------|
| **Chinese New Year** | `ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA` | ✅ Released | 41+ |

**This is the ONLY official v1.5 LoRA currently available.** ACE-Step v1.5 was released just this week (February 2026), and more LoRAs are expected to follow.

---

## Downloaded LoRAs

The following LoRAs have been downloaded to this project's `loras/` directory:

### 1. ACE-Step-v1.5-chinese-new-year-LoRA

**Location**: `loras/ACE-Step-v1.5-chinese-new-year-LoRA/`

**Files**:
```
loras/ACE-Step-v1.5-chinese-new-year-LoRA/
├── adapter_config.json         (894 bytes)
├── adapter_model.safetensors   (88 MB)
└── README.md                   (2,930 bytes)
```

**Description**: 
- Chinese New Year themed LoRA for Chinese folk-pop music
- Trained on 12 New Year-themed songs for ~1 hour on A100 GPU
- Emulates female vocalist renowned for Chinese folk music performances
- Generates Chinese folk instruments (bass drum, dizi, erhu)
- Supports original folk-style composition

**Training Songs**:
- 万事如意 (All the Best)
- 好日子 (Good Days)
- 好运来 (May Fortune Come)
- 常回家看看 (Go Home Often)
- 恭喜发财 (Wishing You a Prosperous New Year)
- 拥军花鼓 (March of the Army and the Flower Drum)
- 春节序曲 (Spring Festival Overture)
- 步步高 (Step by Step Up)
- 祝酒歌 (Toast Song)
- 越来越好 (Getting Better and Better)
- 迎宾曲 (Welcome Overture)
- 难忘今宵 (Unforgettable Tonight)

**Base Model**: `ACE-Step/Ace-Step1.5`

**License**: CreativeML OpenRAIL-M (Research only, no commercial use)

**Usage Notes**:
- Only use with the **DiT model**, not the Think/LM model
- Use festive Chinese folk-pop style captions for best results

**Example Captions**:

```
An explosive and theatrical big band arrangement kicks off with a flurry of woodblock percussion and a dramatic orchestral hit, launching into a high-energy swing groove. A powerful female vocalist, singing in a clear and operatic style, soars over a dense mix of punchy brass stabs, a walking bassline, and a driving drum kit.
```

```
An energetic and celebratory Chinese folk-pop track driven by a powerful, galloping percussion ensemble of large drums and sharp cymbals. A piercing suona carries the main melodic hook with a bright, festive tone. A clear, powerful female vocal delivers the lyrics in a traditional, almost operatic style.
```

---

## LoRA Architecture

### Target Modules

LoRAs in ACE-Step target the attention projection layers in the DiT decoder:

```python
target_modules = [
    "q_proj",   # Query projection
    "k_proj",   # Key projection
    "v_proj",   # Value projection
    "o_proj"    # Output projection
]
```

### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` (rank) | 8 | LoRA capacity |
| `alpha` | 16 | Scaling factor (typically 2× rank) |
| `dropout` | 0.1 | Regularization |
| `bias` | "none" | Bias training mode |

### File Format

```
lora_adapter/
├── adapter_config.json      # PEFT configuration (required)
├── adapter_model.safetensors # LoRA weights (required)
└── README.md                 # Optional metadata
```

---

## How to Use LoRAs

### Method 1: Gradio UI (Recommended)

1. Start the Gradio UI: `acestep --port 7865`
2. Click "Initialize Service" to load the DiT model
3. In "🔧 LoRA Adapter" section:
   - Enter path: `./loras/ACE-Step-v1.5-chinese-new-year-LoRA`
   - Click "📥 Load LoRA"
4. Verify status shows "✅ LoRA loaded"
5. Adjust LoRA scale slider (0.0-1.0) for influence strength
6. Generate music with appropriate captions

### Method 2: Programmatic Python

```python
from acestep.handler import AceStepHandler

# Initialize handler
handler = AceStepHandler()
handler.initialize_service(
    project_root="./",
    config_path="acestep-v15-turbo",
    device="cuda",
)

# Load LoRA
status = handler.load_lora("./loras/ACE-Step-v1.5-chinese-new-year-LoRA")
print(status)

# Adjust scale (optional)
handler.set_lora_scale(0.8)  # 80% LoRA influence

# Generate music with festive caption
result = handler.generate_music(
    captions="Chinese folk-pop, celebratory, female vocalist, suona, drums",
    lyrics="[Verse]\n恭喜恭喜恭喜你...",
    audio_duration=60.0,
)

# Toggle LoRA on/off
handler.set_use_lora(False)  # Disable
handler.set_use_lora(True)   # Enable

# Unload when done
handler.unload_lora()
```

### LoRA Scale Effects

| Scale | Effect |
|-------|--------|
| `0.0` | LoRA disabled (base model only) |
| `0.25` | Subtle LoRA influence |
| `0.5` | Balanced blend |
| `0.75` | Strong LoRA influence |
| `1.0` | Full LoRA effect (default) |

---

## Creating Custom LoRAs

Since ACE-Step v1.5 has just been released, you may want to train your own LoRAs.

### Requirements

- **Audio**: 8+ songs (recommended: 20+ for better results)
- **Formats**: WAV, MP3, FLAC, OGG, OPUS
- **VRAM**: 12GB for training, <4GB for inference
- **Time**: ~1 hour on RTX 3090 with 8 songs

### Training via Gradio UI

1. Navigate to "🎓 LoRA Training" tab
2. **Dataset Builder**:
   - Enter audio directory path
   - Click "🔍 Scan"
   - Set dataset name and custom activation tag
3. **Auto-Label**: Click "🏷️ Auto Label All"
4. **Preprocess**: Set output directory, click "⚙️ Preprocess"
5. **Train**:
   - Rank: 64 (recommended)
   - Alpha: 128
   - Learning Rate: 1e-4
   - Epochs: 100
   - Click "▶️ Start Training"
6. **Export**: Enter path, click "📤 Export LoRA"

---

## Future LoRAs (Announced)

The following LoRAs are mentioned in the ACE-Step README but are **NOT YET RELEASED**:

### Lyric2Vocal (LoRA)
- **Status**: Announced, not released
- **Purpose**: Generate vocal samples directly from lyrics
- **Training**: Fine-tuned on pure vocal data

### Text2Samples (LoRA)
- **Status**: Announced, not released
- **Purpose**: Generate instrumental samples from text
- **Training**: Fine-tuned on pure instrumental/sample data

### StemGen (ControlNet-LoRA)
- **Status**: Coming Soon
- **Purpose**: Generate individual instrument stems
- **Type**: ControlNet-LoRA trained on multi-track data

### Singing2Accompaniment (ControlNet)
- **Status**: Coming Soon
- **Purpose**: Generate mixed track from vocal track
- **Type**: ControlNet (reverse of StemGen)

---

## v1 LoRAs (Legacy)

These LoRAs are for ACE-Step v1 (3.5B) and may **NOT be compatible** with v1.5:

| LoRA | HuggingFace ID | Status |
|------|---------------|--------|
| **RapMachine** | `ACE-Step/ACE-Step-v1-chinese-rap-LoRA` | v1 only |

### RapMachine (v1)

If you need the v1 RapMachine LoRA:

```bash
hf download ACE-Step/ACE-Step-v1-chinese-rap-LoRA --local-dir ./loras/v1-chinese-rap
```

**⚠️ Warning**: This is for ACE-Step v1 (3.5B model), not v1.5. Compatibility is not guaranteed.

---

## Summary

| Item | Status |
|------|--------|
| **v1.5 LoRAs Available** | 1 (Chinese New Year) |
| **Downloaded to Project** | ✅ `loras/ACE-Step-v1.5-chinese-new-year-LoRA` |
| **Lyric2Vocal** | ❌ Not yet released |
| **Text2Samples** | ❌ Not yet released |
| **StemGen** | ❌ Not yet released |
| **Singing2Accompaniment** | ❌ Not yet released |
| **RapMachine** | ⚠️ v1 only (legacy) |

### Key Takeaways

1. **ACE-Step v1.5 was just released this week** - LoRA ecosystem is still developing
2. **Only 1 official v1.5 LoRA exists**: Chinese New Year themed folk-pop
3. **The downloaded LoRA is at**: `loras/ACE-Step-v1.5-chinese-new-year-LoRA/`
4. **Use Gradio UI** for easiest LoRA management
5. **Train your own** if you need specific styles - infrastructure is ready
6. **v1 LoRAs may not be compatible** with v1.5 due to architecture changes
