@echo off
cd /d "C:\jason\dev\ace"

set PORT=8019
set ACESTEP_API_PORT=%PORT%

REM Use GPU 1 (3090) instead of GPU 0 (5090)
set CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%


REM Use the 4B LLM model (most advanced, best quality)
set ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
set ACESTEP_CONFIG_PATH=acestep-v15-xl-sft

REM | Model name | Description |
REM |---|---|
REM | `acestep-v15-turbo` | Default. Fast generation (8 steps). |
REM | `acestep-v15-xl-base` | 4B-parameter XL base model. Higher quality, slower. ~50 steps. |
REM | `acestep-v15-xl-sft` | 4B-parameter XL supervised fine-tuned model. Best quality. ~50 steps. |
REM | `acestep-v15-base` | Standard base model. ~50 steps. |
REM | `acestep-v15-sft` | Standard SFT model. ~50 steps. |

REM Set exact VS2022 paths from your error trace
set VCToolsPath=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\
set INCLUDE=%VCToolsPath%include;%INCLUDE%
set LIB=%VCToolsPath%lib\x64;%LIB%
set PATH=%VCToolsPath%bin\Hostx64\x64;%PATH%

REM Fix Python console encoding for Unicode characters
set PYTHONIOENCODING=utf-8
chcp 65001 > nul

REM Verify vcruntime.h path (optional)
echo %INCLUDE% | findstr vcruntime
echo VS2022 environment ready. Starting ace...

cd C:\jason\dev\ace\python_embeded
python ..\acestep\api_server.py

cmd /k
