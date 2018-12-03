FROM cytomineuliege/software-python3-base:latest

RUN pip install pyxit

ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]

