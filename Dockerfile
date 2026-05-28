# Uses a slim Python 3.11 base to keep the image small.
# The build copies the full project into /app inside the container.

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first so Docker can cache the pip layer
# If requirements.txt does not change, Docker skips this slow step on rebuilds
COPY requirements.txt .

# Install dependencies
# --no-cache-dir keeps the image smaller by not storing the pip cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run the training script so the model pickle is baked into the image
# This means the image is self-contained: no external artifact files needed
RUN python model/train_and_save.py

# Expose the port FastAPI will listen on
EXPOSE 8000

# Start uvicorn when the container runs
# --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]