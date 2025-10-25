#!/bin/bash

echo "Stopping all services..."

tmux kill-session -t model1 2>/dev/null && echo "Stopped model1" || echo "model1 not running"
tmux kill-session -t model2 2>/dev/null && echo "Stopped model2" || echo "model2 not running"
tmux kill-session -t codesim 2>/dev/null && echo "Stopped codesim" || echo "codesim not running"

fuser -k 8000/tcp 2>/dev/null
fuser -k 8001/tcp 2>/dev/null

echo "Done!"
