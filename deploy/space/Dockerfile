# Cue the Boom, the studio with no GPU, as a Hugging Face Space (Docker, CPU basic) or any CPU container host.
# What runs: the render API in instant mode (the sample shots made ahead of time, the instant elements, and a hand of
# your own carried with the LTX-2.5 elements rendered ahead of time in samples/library/bending) and the studio
# (Next.js, production build) on port 7860. Set CUE_PASSWORD (a Space secret) to put it behind HTTP Basic auth.
# Built by deploy/space/make_space.py into a Space folder; see deploy/space/README.md.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl xz-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ARG NODE_VERSION=v24.11.0
RUN curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" | tar -xJ -C /opt \
    && ln -s /opt/node-${NODE_VERSION}-linux-x64/bin/node /usr/local/bin/node \
    && ln -s /opt/node-${NODE_VERSION}-linux-x64/bin/npm /usr/local/bin/npm \
    && ln -s /opt/node-${NODE_VERSION}-linux-x64/bin/npx /usr/local/bin/npx

# a Space runs its container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user cuesync cuesync
COPY --chown=user samples samples
COPY --chown=user workflows workflows
COPY --chown=user web web
COPY --chown=user start.sh start.sh

# the core, and the hand tracker's own environment (MediaPipe's opencv-contrib clashes with the core's opencv-python)
RUN pip install --no-cache-dir "./cuesync[server]" \
    && python -m venv cuesync/.venv-track \
    && cuesync/.venv-track/bin/pip install --no-cache-dir mediapipe==1.1.1 onnxruntime==1.31.0 av==19.0.1
ENV CUESYNC_HAND_MODELS=/app/models NEXT_TELEMETRY_DISABLED=1
RUN PYTHONPATH=cuesync/src cuesync/.venv-track/bin/python -c \
    "import cuesync.hand_track as h; [h.model_path(n) for n in h.MODEL_FILES]"

RUN cd web && npm ci --no-audit --no-fund && CUESYNC_API_URL=http://127.0.0.1:8765 npm run build
RUN chown -R user /app
USER user
ENV HOME=/home/user
EXPOSE 7860
CMD ["bash", "start.sh"]
