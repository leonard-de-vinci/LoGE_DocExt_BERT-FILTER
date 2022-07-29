FROM python:3.8.13

WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./trigger_filters.py .
COPY ./enwiki_20180420_100d.pkl .