FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract language files for Kazakh, Russian, English
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata && \
    cd /usr/share/tesseract-ocr/5/tessdata && \
    wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata && \
    wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata && \
    wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/kaz.traineddata

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8090
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
