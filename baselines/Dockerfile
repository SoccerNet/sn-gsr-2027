# NGC PyTorch image, multi-arch (x86_64 and arm64, e.g. DGX Spark / GB10).
# The default (25.08: CUDA 13, PyTorch 2.8) needs an NVIDIA driver >= 580.
# With an older driver, build on an older NGC release (not tested by the organizers), e.g.:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3 -t sngsr-baseline .   # driver >= 570
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:24.05-py3 -t sngsr-baseline .   # driver >= 550
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/baselines
COPY requirements.txt requirements-git.txt ./
RUN pip install --no-cache-dir -r requirements.txt --no-build-isolation \
    && pip install --no-cache-dir -r requirements-git.txt --no-build-isolation --no-deps

COPY . .
RUN pip install --no-cache-dir --no-deps -e . && mkdir -p /cache && chmod 777 /cache

# Model caches live under /cache so that a single host folder can be mounted,
# and HOME=/tmp + USER keep the image usable with `docker run --user $(id -u):$(id -g)`
# (an arbitrary uid has no home and no name in /etc/passwd, which torch.compile/triton need).
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/cache/huggingface \
    CLIP_HOME=/cache/clip \
    HOME=/tmp \
    USER=sngsr
CMD ["python", "run_baseline.py"]
