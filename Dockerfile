FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user vera ./vera

EXPOSE 7860
# One worker only: all state lives in process memory.
CMD ["uvicorn", "vera.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
