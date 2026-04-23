ARG BRAIN2VEC_VERSION="1.0.1"
ARG CUDA_VERSION="12.1"
ARG TORCH_VERSION="2.6.0"
ARG CUDNN_VERSION="8"

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

WORKDIR /app

COPY . /app/

RUN grep -v -E '^(torch)' requirements.txt > requirements2.txt
RUN pip install -r requirements2.txt && pip install --no-deps .

ENV MKL_SERVICE_FORCE_INTEL=1

CMD ["Brain2Vec"]
