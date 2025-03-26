FROM python:3.12.9-bookworm

ARG PACKAGE_VERSION=0.1.0

RUN pip install --upgrade pip

RUN mkdir -p /app

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY dist/deep_reason-${PACKAGE_VERSION}-py3-none-any.whl .

RUN pip install deep_reason-${PACKAGE_VERSION}-py3-none-any.whl

