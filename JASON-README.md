# ACE-Step Fork - Jason's Additions

This fork of [ACE-Step](https://github.com/ace-step/ACE-Step) adds three new API endpoints, a bug fix for CUDA codebook indexing, a `vocal_language` hint parameter, new startup scripts for Windows/Triton, and detailed AI research documentation.

---

## What Was Actually Changed (vs Upstream)

### `acestep/api_server.py` — Three New Endpoints

573 lines added at the end of `create_app()`:

| Endpoint | Description |
|---|---|
| `POST /describe_audio` | Analyze audio (file upload or raw codes) → caption, lyrics, BPM, key, duration, language. Accepts `multipart/form-data` (with `audio_file`) or `application/json` (with `audio_codes`). |
| `POST /extend_audio` | Extend audio duration using the repaint task. |
| `POST /lego` | Generate a specific instrument track (vocals, drums, bass, guitar, etc.) using the lego task. |

Also added two imports: `TRACK_NAMES` and `understand_music`.

---

### `acestep/handler.py` — Codebook Index Clamping (Bug Fix)

Added ~20 lines to `_decode_audio_codes_to_latents()` that clamp codebook indices to the valid range before passing them to the decoder:

- Detects codebook size dynamically from the quantizer (falling back to 64000 for ACE-Step v1.5's FSQ with levels `[8,8,8,5,5,5]`)
- Clamps any out-of-range indices with a warning log
- Prevents CUDA index-out-of-bounds errors when LM-generated codes are slightly out of range

---

### `acestep/inference.py` — `vocal_language` Parameter on `understand_music()`

Added the `vocal_language: Optional[str]` parameter to `understand_music()`, which passes a language hint to the LLM so it generates lyrics in the correct language rather than guessing. Also a handful of encoding fixes (BOM, UTF-8 special characters in docstrings).

---

### `acestep/llm_inference.py` — `vocal_language` Support in LLM Call

Added ~10 lines wiring `vocal_language` through to the LLM: if provided and not `"unknown"`, it sets `user_metadata = {"language": ...}` and sets `skip_language = True` so the LM doesn't re-generate the language field. Also switched to `AutoTokenizer.from_pretrained(..., use_fast=True, trust_remote_code=True)`.

---

## New Files Added

### Startup Scripts (Windows / Triton Fix)

- **`ace_start_server_with_triton_fix.bat`** — Sets VS2022 compiler paths (required for Triton to compile CUDA kernels on Windows), sets `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B`, fixes Python UTF-8 encoding, then launches `api_server.py`.
- **`ace_start_ui_with_triton_fix.bat`** — Same VS2022 fix, then launches the Gradio UI (`acestep_v15_pipeline.py`).

### AI Research Documentation (`ai-research/`)

Seven detailed docs covering the upstream ACE-Step capabilities, written as reference for development:

| File | Topic |
|---|---|
| `repaint.md` | Audio inpainting/outpainting — time-region regeneration, latent padding math |
| `loras.md` | LoRA adapters — available adapters, architecture, training |
| `cover-song.md` | Cover/style transfer — audio codes, cover strength parameter |
| `describe-audio.md` | Audio analysis pipeline — `/describe_audio` endpoint reference |
| `extend-audio.md` | Duration extension — three modes, overlap for smooth transitions |
| `lyric-2-vocal.md` | Vocal generation — four methods, model limitations |
| `lego.md` | Track-by-track generation — 12 track types, `/lego` endpoint reference |

### LoRA Model

- **`loras/ACE-Step-v1.5-chinese-new-year-LoRA/`** — Pre-trained LoRA adapter files included in the repo.
