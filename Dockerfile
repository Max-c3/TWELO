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
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements_docker.txt requirements.txt
RUN pip install -r requirements.txt

COPY setup.py setup.py
COPY twelo twelo
# models repositary need to include the loaded_model in fast.py (see line 12)
COPY models models
# data repositary should be copied after having made make reset_local_data
COPY data data

RUN pip install .

CMD uvicorn twelo.api.fast:app --host 0.0.0.0 --port $PORT
