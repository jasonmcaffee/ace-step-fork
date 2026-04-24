@echo off
cd /d "C:\jason\dev\ace"

REM Set exact VS2022 paths from your error trace
set VCToolsPath=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\
set INCLUDE=%VCToolsPath%include;%INCLUDE%
set LIB=%VCToolsPath%lib\x64;%LIB%
set PATH=%VCToolsPath%bin\Hostx64\x64;%PATH%

REM Fix Python console encoding for Unicode characters
set PYTHONIOENCODING=utf-8
chcp 65001 > nul

REM Use the 4B LLM model (most advanced, best quality)
set ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B

REM Verify vcruntime.h path (optional)
echo %INCLUDE% | findstr vcruntime
echo VS2022 environment ready. Starting ace...

cd C:\jason\dev\ace\python_embeded
python ..\acestep\api_server.py

cmd /k
