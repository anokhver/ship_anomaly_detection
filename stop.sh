#!/bin/bash

# Kill frontend
if [ -f /tmp/frontend.pid ]; then
    kill $(cat /tmp/frontend.pid) && echo "Frontend stopped."
    rm /tmp/frontend.pid
else
    echo "Frontend PID file not found."
fi

# Kill backend
if [ -f /tmp/backend.pid ]; then
    kill $(cat /tmp/backend.pid) && echo "Backend stopped."
    rm /tmp/backend.pid
else
    echo "Backend PID file not found."
fi