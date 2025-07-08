# 1. Backend and frontend together
FROM python:3.11-slim

# Node.js for Vue + system deps
RUN apt-get update && \
    apt-get install -y \
        curl \
        gnupg \
        git \
        gcc \
        libpq-dev \
        python3-dev \
        build-essential \
        tree \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /workspace

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# Note: Frontend dependencies will be installed at runtime
# This allows for better development workflow with volume mounts

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Setting up development environment..."\n\
echo "Installing frontend dependencies..."\n\
cd /workspace/frontend && npm install && cd /workspace\n\
echo ""\n\
echo "Development environment ready!"\n\
echo ""\n\
exec "$@"' > /usr/local/bin/setup-dev.sh && chmod +x /usr/local/bin/setup-dev.sh

# Expose both ports
EXPOSE 8000 5173

# Use the setup script as entrypoint
ENTRYPOINT ["/usr/local/bin/setup-dev.sh"]

# Set default command to bash for interactive development
CMD ["/bin/bash"]

# run the backend server (in case starting script fails)
## python manage.py runserver 0.0.0.0:8000