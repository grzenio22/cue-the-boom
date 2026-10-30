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
COPY --chown=user effects effects
COPY --chown=user samples samples
COPY --chown=user workflows workflows
COPY --chown=user web web
COPY --chown=user LICENSE NOTICE.md ./
COPY --chown=user build/model-files.md build/model-files.md
COPY --chown=user tools/derive_fire_element.py tools/derive_fire_element.py
COPY --chown=user start.sh start.sh

# the core, and the hand tracker's own environment (MediaPipe's opencv-contrib clashes with the core's opencv-python)
RUN pip install --no-cache-dir "./cuesync[server]" \
    && python -m venv cuesync/.venv-track \
    && cuesync/.venv-track/bin/pip install --no-cache-dir mediapipe==1.1.1 onnxruntime==1.31.0 av==19.0.1
ENV CUESYNC_HAND_MODELS=/app/models NEXT_TELEMETRY_DISABLED=1
RUN PYTHONPATH=cuesync/src cuesync/.venv-track/bin/python -c \
    "import cuesync.hand_track as h; [h.model_path(n) for n in h.MODEL_FILES]" \
    && touch cuesync/.venv-track/.cuesync-ready
# (the ready mark: the studio offers a hand of your own only once the tracker's environment says it works; the
# matte runs in the same environment, with MediaPipe's skin segmenter and the hand's landmarks, no SAM)

RUN cd web && npm ci --no-audit --no-fund && CUESYNC_API_URL=http://127.0.0.1:8765 npm run build
# Runtime downloads need the model directory; Next writes caches and regenerated pages beneath .next.
# The preinstalled tracker environment and node_modules stay root-owned and readable by the runtime user.
RUN chown user /app/models && chown -R user /app/web/.next
# MediaPipe's CPU task library also links EGL/GLES; importing its Python package alone does not load it.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/* \
    && { cuesync/.venv-track/bin/python -c \
    "import ctypes, mediapipe; from pathlib import Path; ctypes.CDLL(str(Path(mediapipe.__file__).parent / 'tasks/c/libmediapipe.so'))" \
    || echo "warning: libmediapipe.so preload check skipped (the HF build sandbox hides the CPU count from tcmalloc)"; }
USER user
ENV HOME=/home/user
EXPOSE 7860
CMD ["bash", "start.sh"]
