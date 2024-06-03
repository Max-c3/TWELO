# base image

FROM python:3.10.6-buster
# -> works for streamlit!

# For m-chips...
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements_docker.txt requirements.txt
RUN pip install -r requirements.txt

COPY setup.py setup.py
COPY kestrix kestrix

RUN pip install .

# Set environment variable for port
# ENV PORT 8080

# Expose port 8000
# EXPOSE 8080

CMD uvicorn kestrix.api.fast:app --host 0.0.0.0 --port $PORT
#-> port 8000 correct?
