#!/bin/bash

# Start frontend
cd frontend && npm run dev &
FRONTEND_PID=$!

# Start backend
cd backend && python manage.py runserver 0.0.0.0:8000 &
BACKEND_PID=$!

# Save PIDs to a file
echo $FRONTEND_PID > /tmp/frontend.pid
echo $BACKEND_PID > /tmp/backend.pid

echo "Frontend PID: $FRONTEND_PID"
echo "Backend PID: $BACKEND_PID"

# Wait for both processes
wait
