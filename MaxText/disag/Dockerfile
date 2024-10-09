# Ubuntu:22.04
# Use Ubuntu 22.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags\?page\=1\&name\=22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update && apt install -y --no-install-recommends apt-transport-https ca-certificates gnupg git python3.10 python3-pip curl nano vim

RUN update-alternatives --install     /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y

RUN python3 -m pip install --upgrade pip

ENV JAX_PLATFORMS=proxy
ENV JAX_BACKEND_TARGET=grpc://localhost:38681
ENV XCLOUD_ENVIRONMENT=GCP

RUN git clone https://github.com/google/JetStream.git && \
git clone https://github.com/liurupeng/maxtext.git

RUN cd /maxtext/ && \
pip install -r requirements.txt

RUN pip install setuptools==58 fastapi==0.103.2 uvicorn

RUN pip install /JetStream

RUN apt -y update && apt-get -y install python3-dev && apt-get -y install build-essential
RUN pip install \
    transformers==4.31.0 \
    nltk==3.8.1 \
    orbax-checkpoint==0.6.0 \
    evaluate==0.4.0 \
    absl-py==1.4.0 \
    rouge-score==0.1.2 \
    sentencepiece==0.1.99 \
    accelerate==0.21.0 \
    pathwaysutils@git+https://github.com/AI-Hypercomputer/pathways-utils.git

RUN pip install -U --pre jax==0.4.34.dev20241004 jaxlib==0.4.34.dev20241004 -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html

RUN gsutil cp gs://cloud-tpu-inference-public/jetstream-pathways/flax_proxy/scale_non_shard_axis_all_aqt_flax.py /usr/local/lib/python3.10/dist-packages/aqt/jax/v2/flax/aqt_flax.py

COPY entrypoint.sh /usr/bin/
RUN chmod +x /usr/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]