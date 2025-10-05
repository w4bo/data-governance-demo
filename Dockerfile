FROM python:3.13-slim

RUN mkdir -p /home
COPY ./requirements.txt  /home/

RUN apt-get update \
&& apt-get install git unzip gcc python3-tk tk-dev libgl1 libglib2.0-0 -y \
&& apt-get clean

RUN pip install --upgrade pip
RUN pip install -r /home/requirements.txt
# Set PYTHONPATH
ENV PYTHONPATH="/home/:$PATH"
ENV TZ="Europe/Rome"

EXPOSE 8888

WORKDIR /home