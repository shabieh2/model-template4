

#linux steps

FROM --platform=linux/amd64 python:3.10-slim


RUN pip install poetry

#just a test


COPY pyproject.toml ./
RUN poetry config virtualenvs.create false
RUN poetry install

# Demonstrate the environment is activated:

COPY . .

# The code to run when container is started:



#RUN poetry run train

EXPOSE 8000





ENTRYPOINT gunicorn model.src.serving.serve:app --bind 0.0.0.0:8000
 

