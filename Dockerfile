FROM python:3.8-slim

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN pip install icevision[inference] icedata streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]