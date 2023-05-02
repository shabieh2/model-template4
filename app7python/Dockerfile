

#linux steps

FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get -y upgrade \
 && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    gcc \
    ca-certificates && \
    pip3 install uwsgi \
    && rm -rf /var/lib/apt/lists/*

ENV BUCKETNAME mlflow-bucket-61dbad0
ENV FILENAME lens2.csv


RUN pip install poetry


COPY pyproject.toml ./
RUN poetry config virtualenvs.create false
RUN poetry install

# Demonstrate the environment is activated:
RUN echo "Make sure scipt is installed:"
RUN python -c "import numpy"

COPY . .

# The code to run when container is started:



#RUN poetry run train

EXPOSE 8000



#ENTRYPOINT uvicorn model.src.serving.main:app --host 0.0.0.0 --port 8000 --workers 2

ENTRYPOINT gunicorn model.src.serving.serve:app --bind 0.0.0.0:8000
 





