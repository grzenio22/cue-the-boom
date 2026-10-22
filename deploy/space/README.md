# Prepare the CPU Space

The Space runs the render API in instant mode and the production Next.js studio on port 7860. It uses the
sample library and CPU effects. GPU rendering needs the full app described in the repository README.

## Assemble locally

From the repository root:

```sh
python deploy/space/make_space.py --out runs/space
```

Choose a new output directory for each assembly. The command refuses an existing destination or a destination
inside the source directories it copies. It never deletes an earlier assembly or publishes anything.

The output includes `cuesync/`, `effects/`, `samples/`, `workflows/`, `web/`, `LICENSE`, `NOTICE.md`,
`build/model-files.md`, and the container files. The root `effects/` definitions are required when building the
Python wheel. Local virtual environments, frontend caches, Python build/dist/egg-info output, and generated `effect_files/` copies are
excluded so an old local build cannot supply missing definitions.
Private tool/session directories, local environment files (except `.env.example`), `AGENTS.md`, and the
frontend's redundant `web/out` export are also excluded.

The repository's MIT license has exceptions documented in `NOTICE.md`: the Python package carries its Apache
license, the workflow carries the LTX license, and `samples/CREDITS.md` lists the footage sources and licenses.

## Check the package without downloading models

Run the packaging regressions in the project's Python development environment:

```sh
python -m pytest cuesync/tests/test_space_packaging.py -q
```

To build a wheel from the assembled source, use a Python environment with `setuptools>=69` and `wheel` installed:

```sh
python -m pip wheel --no-deps --no-build-isolation ./runs/space/cuesync --wheel-dir runs/space-wheels
```

Install that wheel with `--no-deps` into a fresh virtual environment, move to a directory outside this checkout,
unset `CUESYNC_EFFECTS`, and run:

```sh
python -I -c "from cuesync.effects import load_effect; print(load_effect('fireball').id)"
```

It must print `fireball`. Testing outside the checkout prevents the repository's `effects/` directory from
concealing an incomplete wheel. This check needs no GPU, model weights, or application runtime dependencies.

## Check the container before publishing

With Docker running:

```sh
docker build -t cue-the-boom-space runs/space
docker run --rm --workdir /tmp --entrypoint python cue-the-boom-space -I -c "from cuesync.effects import load_effect; print(load_effect('fireball').id)"
docker run --rm -p 127.0.0.1:7860:7860 cue-the-boom-space
```

The Docker build downloads OS, Python, Node, and hand-tracker dependencies and the small hand-tracker models.
It does not download the LTX video-generation weights. Open `http://localhost:7860`, play the opening demo,
and make a sample-hand performance. Also check **Use your own clip** after the hand tracker is ready.

For a password-protected run, pass `-e CUE_PASSWORD` to `docker run` after setting that variable in your shell.
On a hosted Space, configure `CUE_PASSWORD` as a Space secret. Publishing and selecting hosting resources are
separate steps; assembly and these local checks do neither.
