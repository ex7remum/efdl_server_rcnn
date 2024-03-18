FROM python:3.7

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /app
WORKDIR /app

COPY labels.json .
COPY server.py .
COPY maskrcnn_resnet50_fpn.pt .

ENTRYPOINT ["python3", "server.py"]
