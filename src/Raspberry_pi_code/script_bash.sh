#!/bin/bash

# Ruta al interprete de Python (ajusta si es necesario)
PYTHON_EXECUTABLE=python3

# Ruta al script Python (ajusta si es necesario)
SCRIPT_PATH="main2.py"

# Ejecuta el script en una nueva terminal usando lxterminal
lxterminal --command="$PYTHON_EXECUTABLE $SCRIPT_PATH"
