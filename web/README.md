# Cue the Boom web app

Next.js 16 (App Router) + TypeScript.

```powershell
npm install
```

```powershell
npm run dev
```

Open http://localhost:3000. Camera and microphone access need a **secure context**: `localhost` works on the laptop. To test on a phone over your LAN, run the dev server over HTTPS (self-signed) and open the printed `https://<your-ip>:3000` address:

```powershell
npm run dev -- --experimental-https --hostname 0.0.0.0
```

It talks to the render API (`cuesync serve`, see the root README) through this origin: `/api/*` is proxied to
`CUESYNC_API_URL` (default `http://127.0.0.1:8765`). The address is read when `npm run dev` starts and fixed by
`npm run build`, so set it before either if the API runs elsewhere.

## Pages

- `/` is the studio:
  1. Pick a sample shot or your own clip.
  2. Tap or draw where the effect comes from.
  3. Cue it: the microphone, the clip's own sound, or press-and-hold.
  4. Make it **Instant** (no GPU) or **Into the scene** (GPU), then drag the hit to re-time it.
- `/dub` is the older recorder. You download `mic.wav`, `face.<mp4|webm>` and `timing.json`; nothing is uploaded.

## `timing.json` (v1)

`noise_offset_s` is the plate time of the mic recording's first sample, before latency is subtracted. `cuesync` places the noise at `noise_offset_s - input_latency_s`:

```powershell
cuesync analyze mic.wav --timing timing.json -o cue.json
```

`face_offset_s` aligns the face video's audio to the mic by cross-correlation.

## Tests

```powershell
npm test
```
