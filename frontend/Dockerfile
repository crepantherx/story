FROM python:3.9

WORKDIR /usr/src/frontend

COPY ./ .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

EXPOSE 8501