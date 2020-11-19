FROM python:3.8-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y gcc && apt-get -y clean && apt-get -y autoremove

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt \
    && pip install icevision[inference]==0.2.2 icedata streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]