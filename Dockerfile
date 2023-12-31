FROM python:3.10.6-slim
RUN pip install --upgrade pip

COPY requirements_prod.txt requirements_prod.txt
RUN pip install -r requirements_prod.txt

COPY mlb mlb
COPY setup.py setup.py

CMD uvicorn mlb.api.fast:app --host 0.0.0.0 --port $PORT
