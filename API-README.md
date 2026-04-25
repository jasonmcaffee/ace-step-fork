# ACE-Step API Reference

## Starting the Server

Run the startup script from the project root:

```bat
ace_start_server_with_triton_fix.bat
```

The server starts on **port 8001** by default (`http://localhost:8001`).

To use a different port, set `ACESTEP_API_PORT` before launching or pass `--port`:

```bat
set ACESTEP_API_PORT=8080
python acestep\api_server.py
```

### Selecting a Model at Startup

Set `ACESTEP_CONFIG_PATH` to load a specific DiT model:

| Model name | Description |
|---|---|
| `acestep-v15-turbo` | Default. Fast generation (8 steps). |
| `acestep-v15-xl-base` | 4B-parameter XL base model. Higher quality, slower. ~50 steps. |
| `acestep-v15-xl-sft` | 4B-parameter XL supervised fine-tuned model. Best quality. ~50 steps. |
| `acestep-v15-base` | Standard base model. ~50 steps. |
| `acestep-v15-sft` | Standard SFT model. ~50 steps. |

```bat
set ACESTEP_CONFIG_PATH=acestep-v15-xl-base
set ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
python acestep\api_server.py
```

Models are downloaded automatically from HuggingFace on first use.

---

## Health Check

```bash
curl http://localhost:8001/health
```

```json
{
  "data": {
    "status": "ok",
    "models_initialized": true,
    "llm_initialized": true,
    "loaded_model": "acestep-v15-xl-base",
    "loaded_lm_model": "acestep-5Hz-lm-4B"
  },
  "code": 200
}
```

---

## Generate Music

Music generation is asynchronous: submit a task, then poll for the result.

### Step 1 — Submit a task

**`POST /release_task`**

```bash
curl -X POST http://localhost:8001/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "an upbeat electronic pop song with synths and driving drums",
    "lyrics": "[Instrumental]",
    "audio_duration": 30,
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "seed": -1,
    "audio_format": "mp3"
  }'
```

**Response:**

```json
{
  "data": {
    "task_id": "52f06ff8-0756-42af-984d-bf477dc3c416",
    "status": "queued",
    "queue_position": 1
  },
  "code": 200
}
```

### Step 2 — Poll for the result

**`POST /query_result`**

```bash
curl -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d '{"task_ids": ["52f06ff8-0756-42af-984d-bf477dc3c416"]}'
```

Poll every 5–15 seconds until `status` is `succeeded` or `failed`. The result contains a URL to download the audio:

```json
{
  "data": [
    {
      "task_id": "52f06ff8-...",
      "status": "succeeded",
      "result": {
        "first_audio_path": "/v1/audio?path=%2F...%2Faudio.mp3",
        "audio_paths": ["/v1/audio?path=..."],
        "bpm": 128,
        "duration": 30,
        "keyscale": "C Major"
      }
    }
  ],
  "code": 200
}
```

### Step 3 — Download the audio

```bash
curl "http://localhost:8001/v1/audio?path=%2F...%2Faudio.mp3" -o output.mp3
```

---

## Generating with the XL Model

### Option A — Server loaded with XL model at startup

If the server was started with `ACESTEP_CONFIG_PATH=acestep-v15-xl-base`, all requests automatically use it. No extra parameter needed:

```bash
curl -X POST http://localhost:8001/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cinematic orchestral piece with soaring strings and brass",
    "lyrics": "[Instrumental]",
    "audio_duration": 30,
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "seed": -1,
    "audio_format": "mp3"
  }'
```

> **Recommended steps:** 50 for XL base/sft, vs 8 for turbo.

### Option B — Multi-model server, specify model per request

If the server was started with both turbo and XL loaded (e.g. `ACESTEP_CONFIG_PATH=acestep-v15-turbo` and `ACESTEP_CONFIG_PATH2=acestep-v15-xl-base`), pass the `model` field to pick one:

```bash
curl -X POST http://localhost:8001/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cinematic orchestral piece with soaring strings and brass",
    "lyrics": "[Instrumental]",
    "model": "acestep-v15-xl-base",
    "audio_duration": 30,
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "seed": -1,
    "audio_format": "mp3"
  }'
```

---

## Key Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | `""` | Music style/mood/instrument description |
| `lyrics` | string | `""` | Lyric text. Use `"[Instrumental]"` for no vocals |
| `audio_duration` | float | auto | Target length in seconds (10–600) |
| `inference_steps` | int | `8` | Diffusion steps. Use 8 for turbo, 50 for base/XL |
| `guidance_scale` | float | `7.0` | CFG strength. Higher = closer to prompt |
| `seed` | int | `-1` | `-1` for random, any integer for reproducible output |
| `audio_format` | string | `"mp3"` | Output format: `mp3`, `wav`, `flac` |
| `model` | string | primary | Model name to use (multi-model servers only) |
| `thinking` | bool | `false` | Enable 5Hz LM Chain-of-Thought for audio codes |
| `bpm` | int | auto | Beats per minute (auto-detected if omitted) |
| `key_scale` | string | auto | Musical key, e.g. `"C Major"`, `"Am"` |
| `vocal_language` | string | `"en"` | Vocal language code: `en`, `zh`, `ja`, etc. |
| `batch_size` | int | `1` | Number of variations to generate |
| `task_type` | string | `"text2music"` | Task: `text2music`, `cover`, `repaint`, `lego` |

---

## Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server and model status |
| `/v1/models` | GET | List loaded models |
| `/v1/stats` | GET | Queue and job statistics |
| `/create_random_sample` | POST | Generate a random sample via LLM (no prompt needed) |
| `/format_input` | POST | Enhance a caption/lyrics with the LLM |
| `/describe_audio` | POST | Analyze an audio file → caption, BPM, key, lyrics |
| `/extend_audio` | POST | Extend audio duration using repaint |
| `/lego` | POST | Generate a specific instrument track (vocals, drums, etc.) |
| `/v1/audio` | GET | Download a generated audio file by path |
| `/v1/lora/load` | POST | Load a LoRA adapter |
| `/v1/lora/unload` | POST | Unload the active LoRA adapter |
| `/v1/lora/status` | GET | Get current LoRA state |
