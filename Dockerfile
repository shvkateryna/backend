FROM python:3.10-slim

WORKDIR /app

# 🔧 Установка потрібних системних залежностей
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
