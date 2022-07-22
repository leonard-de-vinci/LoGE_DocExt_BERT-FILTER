FROM python:3.10.4

WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./trigger_filters.py .
COPY ./enwiki_20180420_100d.bin .