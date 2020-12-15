FROM python:3.8-slim-buster

RUN groupadd -r returning && useradd --no-log-init -r -g returning returning

WORKDIR /home/returning

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY data data
COPY models models
COPY returning returning
COPY main.py run.sh ./
RUN chmod +x run.sh

RUN chown -R returning:returning ./
USER returning

# EXPOSE 5000
CMD ["./run.sh"]