FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN pip install langchain_google_genai

COPY ./app /app/app

EXPOSE 8080

CMD ["streamlit","run","app/main.py","--server.port","8080"]
