# ACE-Step Fork - Jason's Additions

This fork of [ACE-Step](https://github.com/ace-step/ACE-Step) extends the music generation system with LLM-driven reasoning, new generation modes, audio analysis, LoRA support, and detailed research documentation.

---

## What Was Added

### 1. Two-Phase Generation Pipeline (`inference.py`)

A new high-level module (`acestep/inference.py`) was created to orchestrate the full generation pipeline. It introduces:

- **Phase 1**: Optional 5Hz Language Model (LM) generates structured metadata (BPM, key, duration, lyrics, captions) and audio codes with Chain-of-Thought reasoning.
- **Phase 2**: The DiT (Diffusion Transformer) model uses those codes and metadata to produce the final audio.

New data structures:
- `GenerationParams` — 40+ parameters covering text inputs, music metadata, DiT tuning, LM sampling, and CoT flags
- `GenerationConfig` — batch config with audio format selection
- `GenerationResult`, `UnderstandResult`, `CreateSampleResult`, `FormatSampleResult` — typed result containers

New functions:
- `generate_music()` — orchestrates both phases, handles batch seeds, saves output audio with UUIDs
- `understand_music()` — converts audio codes into structured descriptions (caption, lyrics, BPM, key, etc.)
- `create_sample()` — **Simple Mode**: takes a natural language query like "a soft Bengali love song" and generates the full sample
- `format_sample()` — **Format Mode**: enhances user-provided caption/lyrics and fills in missing metadata

---

### 2. New API Endpoints (`api_server.py`)

Extended the REST API with new endpoints and request fields:

**New request fields on existing endpoints:**
- `thinking: bool` — enable the 5Hz LM for code generation
- `sample_mode: bool` / `sample_query: str` — trigger Simple Mode from a description
- `use_format: bool` — run Format Mode to enhance input before generation
- `lm_temperature`, `lm_cfg_scale`, `lm_top_k`, `lm_top_p` — LM sampling controls
- `use_cot_caption`, `use_cot_language` — Chain-of-Thought reasoning flags
- `allow_lm_batch: bool` — enable batch LM processing

**New endpoints:**
| Endpoint | Description |
|---|---|
| `POST /describe_audio` | Analyze audio (file upload or codes) → caption, lyrics, BPM, key, etc. |
| `POST /extend_audio` | Extend audio duration using repaint task |
| `POST /lego` | Generate specific instrument tracks (vocals, drums, bass, guitar, etc.) |
| `POST /create_random_sample` | Simple Mode — generate from a natural language description |
| `POST /format_input` | Format Mode — enhance and structure user-provided caption/lyrics |
| `POST /release_task` | Submit a generation task (supports batching) |
| `POST /query_result` | Poll batch task results |

Other server features added:
- Model auto-download from HuggingFace or ModelScope (auto-detects network access)
- API key authentication support
- In-memory job queue with 1-hour task timeout
- Audio format selection: mp3, wav, flac

---

### 3. LoRA Support (`handler.py`)

Added LoRA (Low-Rank Adaptation) adapter management:

- `load_lora(lora_path)` — load a PEFT LoRA adapter onto the model
- `unload_lora()` — restore the base model weights
- `set_use_lora(bool)` — enable/disable LoRA at runtime
- `set_lora_scale(float)` — adjust LoRA influence (0.0–1.0)
- `get_lora_status()` — query current LoRA state

Also added:
- `convert_src_audio_to_codes()` — convert an audio file to semantic codes via VAE+tokenizer
- `get_available_acestep_v15_models()` — list locally available v1.5 model checkpoints
- `is_flash_attention_available()` — check for the `flash_attn` package

---

### 4. Language Model Integration (`llm_inference.py`)

Extended the LM inference layer with:

- `generate_with_stop_condition()` — generate audio codes + metadata with constrained/FSM-based decoding for structured output
- `understand_audio_from_codes()` — reverse-generate descriptions from audio codes
- `create_sample_from_query()` — full sample generation from a natural language query
- `format_sample_from_input()` — format and fill in missing metadata from user input

Supports constrained decoding (FSM-based) so the LM outputs valid structured formats (JSON-like metadata blocks).

---

### 5. Triton Fix Startup Scripts

Two `.bat` files were added to work around Triton's dependency on a C++ compiler:

- **`ace_start_server_with_triton_fix.bat`** — sets VS2022 compiler paths, sets `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B`, fixes Python UTF-8 encoding, then launches `api_server.py`
- **`ace_start_ui_with_triton_fix.bat`** — same VS2022 fix, then launches the Gradio UI (`acestep_v15_pipeline.py`)

---

### 6. AI Research Documentation (`ai-research/`)

Seven detailed research documents covering the advanced features:

| File | Topic |
|---|---|
| `repaint.md` | Audio inpainting/outpainting — time-region regeneration, latent padding math, infinite duration via chaining |
| `loras.md` | LoRA adapters — architecture, training, available adapters (e.g. Chinese New Year LoRA), future roadmap |
| `cover-song.md` | Cover/style transfer — audio code extraction, cover strength parameter, remixing and genre transformation |
| `describe-audio.md` | Audio analysis pipeline — audio→codes→LM description, `/describe_audio` endpoint docs, training dataset use |
| `extend-audio.md` | Duration extension — three extension modes, overlap for smooth transitions, chaining for infinite audio |
| `lyric-2-vocal.md` | Vocal generation — four methods (caption engineering, lego task, extract, complete), limitations per model |
| `lego.md` | Track-by-track generation — 12 track types, `/lego` endpoint, iterative arrangement building, client examples |

---

## Supported Task Types

| Task | Description |
|---|---|
| `text2music` | Standard text-to-music generation |
| `cover` | Style transfer — transform audio while preserving structure |
| `repaint` | Region regeneration or audio extension |
| `lego` | Generate a specific instrument track |
| `extract` | Isolate a track (Base model only) |
| `complete` | Add tracks to existing audio (Base model only) |

---

## Key Model

The 5Hz LM used for Chain-of-Thought reasoning is `acestep-5Hz-lm-4B` — a 4B-parameter language model that generates structured music metadata and semantic audio codes before the DiT diffusion pass.
