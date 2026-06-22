@echo off
title Hand Tracker
echo Instalando dependencias...
pip install opencv-python mediapipe pillow -q
echo.
echo Iniciando Hand Tracker...
python hand_tracker.py
pause
