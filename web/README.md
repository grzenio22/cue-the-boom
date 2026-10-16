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

## Pages

- `/dub`, **Dub Mode**:
  1. Choose a 1–15 s plate and tap the emission point.
  2. Arm the camera and mic. Every voice-processing option is switched off and the result is verified.
  3. Perform over the muted playback.

  You download `mic.wav`, `face.<mp4|webm>` and `timing.json`. Nothing is uploaded.
- `/take`, one-take: coming next.

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
