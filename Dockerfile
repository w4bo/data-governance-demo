# FROM python:3.13-slim
FROM ubuntu:24.04

RUN mkdir -p /home
COPY ./requirements.txt  /home/

RUN apt-get update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install git unzip gcc python3-tk tk-dev libgl1 libglib2.0-0 -y \
&& apt-get install python3.13 python3.13-venv python3.13-full curl -y \
&& apt-get clean

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.13 get-pip.py \
    && rm get-pip.py

RUN python3.13 -m pip install --upgrade pip
RUN python3.13 -m pip install -r /home/requirements.txt

# Set PYTHONPATH
ENV PATH="/home/.local/bin/:${PATH}"
ENV PYTHONPATH="/home/:$PATH"
ENV TZ="Europe/Rome"

RUN curl -SL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose \
 && chmod +x /usr/local/bin/docker-compose

WORKDIR /home