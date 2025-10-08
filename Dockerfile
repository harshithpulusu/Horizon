FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 horizon && chown -R horizon:horizon /app
USER horizon

# Expose port for health checks
EXPOSE 8000

# Run the MCP server
CMD ["python", "-m", "mcp.server"]
