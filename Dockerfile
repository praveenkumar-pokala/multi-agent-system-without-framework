FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set trace directory environment variable so traces are persisted inside the container
ENV TRACE_DIR=/app/traces

# Expose Streamlit default port
EXPOSE 8501

# Launch the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]