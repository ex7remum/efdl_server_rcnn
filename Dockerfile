FROM python:3.7

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /app
WORKDIR /app


COPY proto/* /app/proto
COPY labels.json maskrcnn_resnet50_fpn.pt /app/
COPY *.py /app/

RUN python3 run_codegen.py

COPY supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["supervisord", "-c", "/etc/supervisord.conf"]
