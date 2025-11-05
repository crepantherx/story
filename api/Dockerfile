FROM python:3.9

WORKDIR /src

COPY . /src/api
RUN pip install --no-cache-dir --upgrade -r /src/api/requirements.txt

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080