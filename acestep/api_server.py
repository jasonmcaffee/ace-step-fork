"""FastAPI server for ACE-Step V1.5.

Endpoints:
- POST /release_task          Create music generation task
- POST /query_result          Batch query task results
- POST /create_random_sample  Generate random music parameters via LLM
- POST /format_input          Format and enhance lyrics/caption via LLM
- GET  /v1/models             List available models
- GET  /v1/audio              Download audio file
- GET  /health                Health check

NOTE:
- In-memory queue and job store -> run uvicorn with workers=1.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import urllib.parse
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from loguru import logger

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency
    load_dotenv = None  # type: ignore

from fastapi import FastAPI, Request, Header
from starlette.datastructures import UploadFile as StarletteUploadFile
from acestep.api.train_api_service import (
    initialize_training_state,
)
from acestep.api.jobs.store import _JobStore
from acestep.api.log_capture import install_log_capture
from acestep.api.route_setup import configure_api_routes
from acestep.api.server_cli import run_api_server_main
from acestep.api.lifespan_runtime import initialize_lifespan_runtime
from acestep.api.job_blocking_generation import run_blocking_generate
from acestep.api.job_execution_runtime import run_one_job_runtime
from acestep.api.job_model_selection import select_generation_handler
from acestep.api.job_runtime_state import (
    cleanup_job_temp_files as _cleanup_job_temp_files_state,
    ensure_models_initialized as _ensure_models_initialized,
    update_progress_job_cache as _update_progress_job_cache,
    update_terminal_job_cache as _update_terminal_job_cache,
)
from acestep.api.startup_model_init import initialize_models_at_startup
from acestep.api.worker_runtime import start_worker_tasks, stop_worker_tasks
from acestep.api.server_utils import (
    env_bool as _env_bool,
    get_model_name as _get_model_name,
    is_instrumental as _is_instrumental,
    map_status as _map_status,
    parse_description_hints as _parse_description_hints,
    parse_timesteps as _parse_timesteps,
)
from acestep.api.http.auth import (
    set_api_key,
    verify_api_key,
    verify_token_from_request,
)
from acestep.api.http.release_task_audio_paths import (
    save_upload_to_temp as _save_upload_to_temp,
    validate_audio_path as _validate_audio_path,
)
from acestep.api.http.release_task_models import GenerateMusicRequest
from acestep.api.http.release_task_param_parser import (
    RequestParser,
    _to_float as _request_to_float,
    _to_int as _request_to_int,
)
from acestep.api.runtime_helpers import (
    append_jsonl as _runtime_append_jsonl,
    atomic_write_json as _runtime_atomic_write_json,
    start_tensorboard as _runtime_start_tensorboard,
    stop_tensorboard as _runtime_stop_tensorboard,
    temporary_llm_model as _runtime_temporary_llm_model,
)
from acestep.api.model_download import (
    ensure_model_downloaded as _ensure_model_downloaded,
)

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.constants import (
    DEFAULT_DIT_INSTRUCTION,
    TASK_INSTRUCTIONS,
    TRACK_NAMES,
)
from acestep.inference import (
    generate_music,
    create_sample,
    format_sample,
    understand_music,
)
from acestep.ui.gradio.events.results_handlers import _build_generation_info

def _get_project_root() -> str:
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_file))


# =============================================================================
# Constants
# =============================================================================

RESULT_KEY_PREFIX = "ace_step_v1.5_"
RESULT_EXPIRE_SECONDS = 7 * 24 * 60 * 60  # 7 days
TASK_TIMEOUT_SECONDS = 3600  # 1 hour
JOB_STORE_CLEANUP_INTERVAL = 300  # 5 minutes - interval for cleaning up old jobs
JOB_STORE_MAX_AGE_SECONDS = 86400  # 24 hours - completed jobs older than this will be cleaned

LM_DEFAULT_TEMPERATURE = 0.85
LM_DEFAULT_CFG_SCALE = 2.5
LM_DEFAULT_TOP_P = 0.9


def _wrap_response(data: Any, code: int = 200, error: Optional[str] = None) -> Dict[str, Any]:
    """Wrap response data in standard format."""
    return {
        "data": data,
        "code": code,
        "error": error,
        "timestamp": int(time.time() * 1000),
        "extra": None,
    }


# =============================================================================
# Example Data for Random Sample
# =============================================================================

SIMPLE_MODE_EXAMPLES_DIR = os.path.join(_get_project_root(), "examples", "simple_mode")
CUSTOM_MODE_EXAMPLES_DIR = os.path.join(_get_project_root(), "examples", "text2music")


def _load_all_examples(sample_mode: str = "simple_mode") -> List[Dict[str, Any]]:
    """Load all example data files from the examples directory."""
    examples = []
    examples_dir = SIMPLE_MODE_EXAMPLES_DIR if sample_mode == "simple_mode" else CUSTOM_MODE_EXAMPLES_DIR
    pattern = os.path.join(examples_dir, "example_*.json")

    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                examples.append(data)
        except Exception as e:
            print(f"[API Server] Failed to load example file {filepath}: {e}")

    return examples


# Pre-load example data at module load time
SIMPLE_EXAMPLE_DATA: List[Dict[str, Any]] = _load_all_examples(sample_mode="simple_mode")
CUSTOM_EXAMPLE_DATA: List[Dict[str, Any]] = _load_all_examples(sample_mode="custom_mode")


_project_env_loaded = False


def _load_project_env() -> None:
    """Load .env at most once per process to avoid epoch-boundary stalls (e.g. Windows LoRA training)."""
    global _project_env_loaded
    if _project_env_loaded or load_dotenv is None:
        return
    try:
        project_root = _get_project_root()
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
        _project_env_loaded = True
    except Exception:
        # Optional best-effort: continue even if .env loading fails.
        pass


_load_project_env()


log_buffer, _stderr_proxy = install_log_capture(logger, sys.stderr)
sys.stderr = _stderr_proxy


def _to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    """Coerce v to int, returning default on failure."""
    if v is None:
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    """Coerce v to float, returning default on failure."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _to_bool(v: Any, default: bool = False) -> bool:
    """Coerce v to bool, handling string representations."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def create_app() -> FastAPI:
    store = _JobStore()

    # API Key authentication (from environment variable)
    api_key = os.getenv("ACESTEP_API_KEY", None)
    set_api_key(api_key)

    QUEUE_MAXSIZE = int(os.getenv("ACESTEP_QUEUE_MAXSIZE", "200"))
    WORKER_COUNT = int(os.getenv("ACESTEP_QUEUE_WORKERS", "1"))  # Single GPU recommended

    INITIAL_AVG_JOB_SECONDS = float(os.getenv("ACESTEP_AVG_JOB_SECONDS", "5.0"))
    AVG_WINDOW = int(os.getenv("ACESTEP_AVG_WINDOW", "50"))

    def _path_to_audio_url(path: str) -> str:
        """Convert local file path to downloadable relative URL"""
        if not path:
            return path
        if path.startswith("http://") or path.startswith("https://"):
            return path
        encoded_path = urllib.parse.quote(path, safe="")
        return f"/v1/audio?path={encoded_path}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = initialize_lifespan_runtime(
            app=app,
            store=store,
            queue_maxsize=QUEUE_MAXSIZE,
            avg_window=AVG_WINDOW,
            initial_avg_job_seconds=INITIAL_AVG_JOB_SECONDS,
            get_project_root=_get_project_root,
            initialize_training_state_fn=initialize_training_state,
            ace_handler_cls=AceStepHandler,
            llm_handler_cls=LLMHandler,
        )
        handler = runtime.handler
        llm_handler = runtime.llm_handler
        handler2 = runtime.handler2
        handler3 = runtime.handler3
        config_path2 = runtime.config_path2
        config_path3 = runtime.config_path3
        executor = runtime.executor

        async def _run_one_job(job_id: str, req: GenerateMusicRequest) -> None:
            llm: LLMHandler = app.state.llm_handler

            def _build_blocking_result(
                selected_handler: AceStepHandler,
                selected_model_name: str,
            ) -> Dict[str, Any]:
                return run_blocking_generate(
                    app_state=app.state,
                    req=req,
                    job_id=job_id,
                    store=store,
                    llm_handler=llm,
                    selected_handler=selected_handler,
                    selected_model_name=selected_model_name,
                    map_status=_map_status,
                    result_key_prefix=RESULT_KEY_PREFIX,
                    result_expire_seconds=RESULT_EXPIRE_SECONDS,
                    get_project_root=_get_project_root,
                    get_model_name=_get_model_name,
                    ensure_model_downloaded=_ensure_model_downloaded,
                    env_bool=_env_bool,
                    parse_description_hints=_parse_description_hints,
                    parse_timesteps=_parse_timesteps,
                    is_instrumental=_is_instrumental,
                    create_sample_fn=create_sample,
                    format_sample_fn=format_sample,
                    generate_music_fn=generate_music,
                    default_dit_instruction=DEFAULT_DIT_INSTRUCTION,
                    task_instructions=TASK_INSTRUCTIONS,
                    build_generation_info_fn=_build_generation_info,
                    path_to_audio_url_fn=_path_to_audio_url,
                    log_fn=print,
                )

            await run_one_job_runtime(
                app_state=app.state,
                store=store,
                job_id=job_id,
                req=req,
                ensure_models_initialized_fn=_ensure_models_initialized,
                select_generation_handler_fn=select_generation_handler,
                get_model_name=_get_model_name,
                build_blocking_result_fn=_build_blocking_result,
                update_progress_job_cache_fn=_update_progress_job_cache,
                update_terminal_job_cache_fn=_update_terminal_job_cache,
                map_status=_map_status,
                result_key_prefix=RESULT_KEY_PREFIX,
                result_expire_seconds=RESULT_EXPIRE_SECONDS,
                log_fn=print,
            )

        async def _cleanup_job_temp_files_for_job(job_id: str) -> None:
            await _cleanup_job_temp_files_state(app.state, job_id)

        workers, cleanup_task = start_worker_tasks(
            app_state=app.state,
            store=store,
            worker_count=WORKER_COUNT,
            run_one_job=_run_one_job,
            cleanup_job_temp_files=_cleanup_job_temp_files_for_job,
            cleanup_interval_seconds=JOB_STORE_CLEANUP_INTERVAL,
        )
        initialize_models_at_startup(
            app=app,
            handler=handler,
            llm_handler=llm_handler,
            handler2=handler2,
            handler3=handler3,
            config_path2=config_path2,
            config_path3=config_path3,
            get_project_root=_get_project_root,
            get_model_name=_get_model_name,
            ensure_model_downloaded=_ensure_model_downloaded,
            env_bool=_env_bool,
        )
        try:
            yield
        finally:
            stop_worker_tasks(
                workers=workers,
                cleanup_task=cleanup_task,
                executor=executor,
            )

    app = FastAPI(title="ACE-Step API", version="1.0", lifespan=lifespan)


    configure_api_routes(
        app=app,
        store=store,
        queue_maxsize=QUEUE_MAXSIZE,
        initial_avg_job_seconds=INITIAL_AVG_JOB_SECONDS,
        verify_api_key=verify_api_key,
        verify_token_from_request=verify_token_from_request,
        wrap_response=_wrap_response,
        get_project_root=_get_project_root,
        get_model_name=_get_model_name,
        ensure_model_downloaded=_ensure_model_downloaded,
        env_bool=_env_bool,
        simple_example_data=SIMPLE_EXAMPLE_DATA,
        custom_example_data=CUSTOM_EXAMPLE_DATA,
        format_sample=format_sample,
        to_int=_request_to_int,
        to_float=_request_to_float,
        request_parser_cls=RequestParser,
        request_model_cls=GenerateMusicRequest,
        validate_audio_path=_validate_audio_path,
        save_upload_to_temp=_save_upload_to_temp,
        default_dit_instruction=DEFAULT_DIT_INSTRUCTION,
        lm_default_temperature=LM_DEFAULT_TEMPERATURE,
        lm_default_cfg_scale=LM_DEFAULT_CFG_SCALE,
        lm_default_top_p=LM_DEFAULT_TOP_P,
        map_status=_map_status,
        result_key_prefix=RESULT_KEY_PREFIX,
        task_timeout_seconds=TASK_TIMEOUT_SECONDS,
        log_buffer=log_buffer,
        runtime_start_tensorboard=_runtime_start_tensorboard,
        runtime_stop_tensorboard=_runtime_stop_tensorboard,
        runtime_temporary_llm_model=_runtime_temporary_llm_model,
        runtime_atomic_write_json=_runtime_atomic_write_json,
        runtime_append_jsonl=_runtime_append_jsonl,
        create_sample=create_sample,
    )

    @app.post("/describe_audio")
    async def describe_audio_endpoint(request: Request, authorization: Optional[str] = Header(None)):
        """Describe/understand audio from uploaded audio file or audio codes."""
        content_type = (request.headers.get("content-type") or "").lower()
        temp_files: list[str] = []
        
        llm = app.state.llm_handler
        h = app.state.handler

        if not getattr(app.state, "_llm_initialized", False):
            if getattr(app.state, "_llm_init_error", None):
                return _wrap_response(None, code=500, error=f"LLM not initialized: {app.state._llm_init_error}")
            return _wrap_response(None, code=500, error="LLM not initialized. Please start the server with LLM enabled.")

        try:
            audio_codes = ""
            body: dict = {}
            
            if "multipart/form-data" in content_type:
                form = await request.form()
                body = {k: v for k, v in form.items() if not hasattr(v, 'read')}
                audio_file = form.get("audio_file")
                if isinstance(audio_file, StarletteUploadFile):
                    if not getattr(app.state, "_initialized", False):
                        return _wrap_response(None, code=500, error="DiT model not initialized for audio conversion")
                    audio_path = await _save_upload_to_temp(audio_file, prefix="describe_audio")
                    temp_files.append(audio_path)
                    audio_codes = h.convert_src_audio_to_codes(audio_path)
                    if audio_codes.startswith("❌"):
                        return _wrap_response(None, code=400, error=audio_codes)
                else:
                    audio_codes = str(body.get("audio_codes", "") or "")
            elif "json" in content_type:
                body = await request.json()
                audio_codes = body.get("audio_codes", "") or ""
            else:
                return _wrap_response(None, code=415, error="Unsupported Content-Type. Use multipart/form-data or application/json.")

            verify_token_from_request(body, authorization)
            
            temperature = _to_float(body.get("temperature"), 0.85)
            top_k = _to_int(body.get("top_k"))
            top_p = _to_float(body.get("top_p"))
            repetition_penalty = _to_float(body.get("repetition_penalty"), 1.0)
            use_constrained_decoding = _to_bool(body.get("use_constrained_decoding", True), True)
            constrained_decoding_debug = _to_bool(body.get("constrained_decoding_debug", False), False)
            vocal_language = body.get("vocal_language", "") or ""

            result = understand_music(
                llm_handler=llm,
                audio_codes=audio_codes,
                temperature=temperature,
                top_k=top_k if top_k and top_k > 0 else None,
                top_p=top_p if top_p and top_p < 1.0 else None,
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=use_constrained_decoding,
                constrained_decoding_debug=constrained_decoding_debug,
                vocal_language=vocal_language if vocal_language else None,
            )

            if not result.success:
                return _wrap_response(None, code=500, error=result.error or result.status_message)

            return _wrap_response({
                "caption": result.caption,
                "lyrics": result.lyrics,
                "bpm": result.bpm,
                "duration": result.duration,
                "keyscale": result.keyscale,
                "language": result.language,
                "timesignature": result.timesignature,
                "status_message": result.status_message,
            })
        except Exception as e:
            return _wrap_response(None, code=500, error=f"describe_audio error: {str(e)}")
        finally:
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass

    @app.post("/extend_audio")
    async def extend_audio_endpoint(request: Request, authorization: Optional[str] = Header(None)):
        """Extend audio duration using the repaint task."""
        content_type = (request.headers.get("content-type") or "").lower()
        temp_files: list[str] = []
        
        h = app.state.handler
        llm = app.state.llm_handler

        if not getattr(app.state, "_initialized", False):
            return _wrap_response(None, code=500, error="DiT model not initialized")

        try:
            if "multipart/form-data" not in content_type:
                return _wrap_response(None, code=415, error="Content-Type must be multipart/form-data with audio_file")
            
            form = await request.form()
            body = {k: v for k, v in form.items() if not hasattr(v, 'read')}
            verify_token_from_request(body, authorization)
            
            audio_file = form.get("audio_file")
            if not isinstance(audio_file, StarletteUploadFile):
                return _wrap_response(None, code=400, error="audio_file is required")
            
            src_audio_path = await _save_upload_to_temp(audio_file, prefix="extend_audio")
            temp_files.append(src_audio_path)
            
            try:
                import torchaudio
                audio_tensor, sr = torchaudio.load(src_audio_path)
                original_duration = audio_tensor.shape[-1] / sr
            except Exception as e:
                return _wrap_response(None, code=400, error=f"Failed to load audio file: {str(e)}")
            
            extend_duration = _to_float(body.get("extend_duration"), 30.0)
            overlap_duration = _to_float(body.get("overlap_duration"), 5.0)
            extend_direction = str(body.get("extend_direction", "end") or "end").lower()
            caption = str(body.get("caption", "") or "")
            lyrics = str(body.get("lyrics", "") or "")
            vocal_language = str(body.get("vocal_language", "unknown") or "unknown")
            bpm = _to_int(body.get("bpm"))
            key_scale = str(body.get("key_scale", "") or "")
            time_signature = str(body.get("time_signature", "") or "")
            inference_steps = _to_int(body.get("inference_steps")) or 8
            guidance_scale = _to_float(body.get("guidance_scale"), 7.0)
            seed = _to_int(body.get("seed")) or -1
            thinking = _to_bool(body.get("thinking", True), True)
            lm_temperature = _to_float(body.get("lm_temperature"), 0.85)
            audio_format = str(body.get("audio_format", "mp3") or "mp3")
            
            if extend_direction not in ["end", "start", "both"]:
                return _wrap_response(None, code=400, error=f"Invalid extend_direction: {extend_direction}")
            if overlap_duration < 0:
                overlap_duration = 0
            if overlap_duration > original_duration:
                overlap_duration = original_duration * 0.5
            
            if extend_direction == "end":
                repainting_start = original_duration - overlap_duration
                repainting_end = original_duration + extend_duration
                total_duration = repainting_end
            elif extend_direction == "start":
                repainting_start = -extend_duration
                repainting_end = overlap_duration
                total_duration = original_duration + extend_duration
            else:
                repainting_start = -extend_duration / 2
                repainting_end = original_duration + extend_duration / 2
                total_duration = original_duration + extend_duration
            
            from acestep.inference import GenerationParams, GenerationConfig, generate_music as _gen
            
            params = GenerationParams(
                task_type="repaint",
                src_audio=src_audio_path,
                repainting_start=repainting_start,
                repainting_end=repainting_end,
                caption=caption if caption else "continuation of the music",
                lyrics=lyrics,
                vocal_language=vocal_language,
                bpm=bpm,
                keyscale=key_scale,
                timesignature=time_signature,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                thinking=thinking,
                lm_temperature=lm_temperature,
            )
            config = GenerationConfig(batch_size=1, audio_format=audio_format)
            output_dir = os.environ.get("ACESTEP_OUTPUT_DIR", os.path.join(os.getcwd(), "outputs"))
            os.makedirs(output_dir, exist_ok=True)
            
            result = _gen(dit_handler=h, llm_handler=llm, params=params, config=config, save_dir=output_dir)
            
            if not result.success:
                return _wrap_response(None, code=500, error=result.status_message)
            if not result.audios:
                return _wrap_response(None, code=500, error="No audio generated")
            
            audio_path = result.audios[0].get("path", "")
            if not audio_path or not os.path.exists(audio_path):
                return _wrap_response(None, code=500, error="Audio file not saved")
            
            import base64
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            ext = os.path.splitext(audio_path)[1].lower()
            mime_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg"}
            mime_type = mime_types.get(ext, "audio/mpeg")
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            try:
                os.remove(audio_path)
            except Exception:
                pass
            
            return _wrap_response({
                "audio_data": f"data:{mime_type};base64,{audio_base64}",
                "audio_format": ext.lstrip(".") or "mp3",
                "original_duration": original_duration,
                "extended_duration": total_duration,
                "repainting_start": repainting_start,
                "repainting_end": repainting_end,
                "extend_direction": extend_direction,
                "status_message": f"Audio extended from {original_duration:.1f}s to {total_duration:.1f}s",
            })
        except Exception as e:
            logger.exception("Error in /extend_audio endpoint")
            return _wrap_response(None, code=500, error=f"extend_audio error: {str(e)}")
        finally:
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass

    @app.post("/lego")
    async def lego_endpoint(request: Request, authorization: Optional[str] = Header(None)):
        """Generate a specific instrument track (vocals, drums, guitar, etc.) using the lego task."""
        content_type = (request.headers.get("content-type") or "").lower()
        temp_files: list[str] = []
        
        h = app.state.handler
        llm = app.state.llm_handler

        if not getattr(app.state, "_initialized", False):
            return _wrap_response(None, code=500, error="DiT model not initialized")

        try:
            if "multipart/form-data" not in content_type:
                return _wrap_response(None, code=415, error="Content-Type must be multipart/form-data with audio_file")
            
            form = await request.form()
            body = {k: v for k, v in form.items() if not hasattr(v, 'read')}
            verify_token_from_request(body, authorization)
            
            audio_file = form.get("audio_file")
            if not isinstance(audio_file, StarletteUploadFile):
                return _wrap_response(None, code=400, error="audio_file is required")
            
            src_audio_path = await _save_upload_to_temp(audio_file, prefix="lego_context")
            temp_files.append(src_audio_path)
            
            track_name = str(body.get("track_name", "") or "").strip().lower()
            if not track_name:
                return _wrap_response(None, code=400, error="track_name is required. Options: " + ", ".join(TRACK_NAMES))
            if track_name not in TRACK_NAMES:
                return _wrap_response(None, code=400, error=f"Invalid track_name: '{track_name}'. Valid: " + ", ".join(TRACK_NAMES))
            
            try:
                import torchaudio
                audio_tensor, sr = torchaudio.load(src_audio_path)
                audio_duration = audio_tensor.shape[-1] / sr
            except Exception as e:
                return _wrap_response(None, code=400, error=f"Failed to load audio file: {str(e)}")
            
            caption = str(body.get("caption", "") or "")
            lyrics = str(body.get("lyrics", "") or "")
            vocal_language = str(body.get("vocal_language", "en") or "en")
            bpm = _to_int(body.get("bpm"))
            key_scale = str(body.get("key_scale", "") or "")
            time_signature = str(body.get("time_signature", "") or "")
            repainting_start = _to_float(body.get("repainting_start"), 0.0)
            repainting_end_raw = _to_float(body.get("repainting_end"), -1.0)
            repainting_end = audio_duration if repainting_end_raw < 0 else repainting_end_raw
            inference_steps = _to_int(body.get("inference_steps")) or 50
            guidance_scale = _to_float(body.get("guidance_scale"), 7.0)
            seed = _to_int(body.get("seed")) or -1
            audio_format = str(body.get("audio_format", "mp3") or "mp3")
            batch_size = min(max(_to_int(body.get("batch_size")) or 1, 1), 8)
            
            instruction = h.generate_instruction(task_type="lego", track_name=track_name)
            if not caption:
                caption = f"{track_name} track, matching the style and rhythm of the context"
            
            from acestep.inference import GenerationParams, GenerationConfig, generate_music as _gen
            
            params = GenerationParams(
                task_type="lego",
                instruction=instruction,
                src_audio=src_audio_path,
                repainting_start=repainting_start,
                repainting_end=repainting_end,
                caption=caption,
                lyrics=lyrics,
                vocal_language=vocal_language,
                bpm=bpm,
                keyscale=key_scale,
                timesignature=time_signature,
                duration=audio_duration,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                thinking=False,
            )
            config = GenerationConfig(batch_size=batch_size, audio_format=audio_format)
            output_dir = os.environ.get("ACESTEP_OUTPUT_DIR", os.path.join(os.getcwd(), "outputs"))
            os.makedirs(output_dir, exist_ok=True)
            
            result = _gen(dit_handler=h, llm_handler=llm, params=params, config=config, save_dir=output_dir)
            
            if not result.success:
                return _wrap_response(None, code=500, error=result.status_message)
            if not result.audios:
                return _wrap_response(None, code=500, error="No audio generated")
            
            import base64
            mime_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg"}
            audio_results = []
            for audio_info in result.audios:
                audio_path = audio_info.get("path", "")
                if not audio_path or not os.path.exists(audio_path):
                    continue
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                ext = os.path.splitext(audio_path)[1].lower()
                mime_type = mime_types.get(ext, "audio/mpeg")
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_results.append({"audio_data": f"data:{mime_type};base64,{audio_base64}", "audio_format": ext.lstrip(".") or "mp3"})
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            
            if not audio_results:
                return _wrap_response(None, code=500, error="No audio files saved")
            
            return _wrap_response({
                "audios": audio_results,
                "track_name": track_name,
                "batch_size": len(audio_results),
                "duration": audio_duration,
                "repainting_start": repainting_start,
                "repainting_end": repainting_end,
                "instruction": instruction,
                "status_message": f"{len(audio_results)} {track_name.upper()} track(s) generated",
            })
        except Exception as e:
            logger.exception("Error in /lego endpoint")
            return _wrap_response(None, code=500, error=f"lego error: {str(e)}")
        finally:
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass

    return app


app = create_app()


def main() -> None:
    """CLI entrypoint for API server startup."""

    run_api_server_main(env_bool=_env_bool)

if __name__ == "__main__":
    main()







