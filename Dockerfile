FROM python:3.8

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install icevision[inference] icedata

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]