FROM python:3.12-bookworm

RUN pip install poetry poetry-plugin-export

RUN mkdir -p /src

WORKDIR /src

