FROM python:3.6.0
COPY . /source/LQ_DIP_HOMEWORK
WORKDIR /source/LQ_DIP_HOMEWORK
RUN pip3 install -r requestments.txt && python3 main.py