FROM python:3.9

WORKDIR /api

COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080