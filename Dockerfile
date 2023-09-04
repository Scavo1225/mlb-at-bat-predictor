FROM python:3.10.6-buster
RUN pip install --upgrade pip

COPY requirements_prod.txt requirements_prod.txt
RUN pip install -r requirements.txt

COPY mlb mlb
COPY setup.py setup.py

CMD uvicorn mlb.api.fast:app --host 0.0.0.0 --port $PORT
