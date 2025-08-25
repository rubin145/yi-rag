FROM python:3.12.8-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose the port Gradio will run on
EXPOSE 7860

# Run the app
CMD ["python", "app/app.py"]