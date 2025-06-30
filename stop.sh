#!/bin/bash

echo "Stopping development servers..."

# Kill frontend processes
echo "Stopping frontend..."
if [ -f /tmp/frontend.pid ]; then
    PID=$(cat /tmp/frontend.pid)
    kill -9 $PID 2>/dev/null && echo "Frontend process $PID killed."
    rm /tmp/frontend.pid
else
    echo "Frontend PID file not found."
fi

# Kill backend processes  
echo "Stopping backend..."
if [ -f /tmp/backend.pid ]; then
    PID=$(cat /tmp/backend.pid)
    kill -9 $PID 2>/dev/null && echo "Backend process $PID killed."
    rm /tmp/backend.pid
else
    echo "Backend PID file not found."
fi

# Kill any remaining Django/Python processes by name
pkill -9 -f "manage.py runserver" 2>/dev/null && echo "Killed Django runserver processes"

# Kill any remaining Vite/Node processes by name
pkill -9 -f "vite.*dev" 2>/dev/null && echo "Killed Vite dev processes"
pkill -9 -f "node.*vite" 2>/dev/null && echo "Killed Node/Vite processes"

# Kill any remaining npm processes
pkill -9 -f "npm.*dev" 2>/dev/null && echo "Killed npm dev processes"

echo "All development servers stopped."