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
  1. **Perform this fire with your voice** selects the prepared palm clip. Selecting it does not request microphone access.
  2. **Use my microphone** grants access; **Record & make fire** starts the recording and clip playback.
  3. Say BOOM or a few quick pops. When the clip ends, the studio uploads the new WAV and automatically composes the existing fire with its timing and strength.
  4. **Download** the result or **Perform again** with another rhythm. Sample voices are a clearly labeled, collapsed alternative.

  **Use your own clip or explore effects** opens the full flow: choose or film a clip, tap or draw the source, and cue it with a microphone, the clip's own sound, or press-and-hold. Effects can be carried on a tracked hand, placed as an instant element, or generated into the scene when the required GPU resources are ready.
- `/dub` is the older recorder. You download `mic.wav`, `face.<mp4|webm>` and `timing.json`; nothing is uploaded.

The prepared-palm shortcut sends `mode=bending`, `effect=fire_burst`, and `performance_mode=reuse` to `/api/jobs`. This explicitly composes the saved source on the CPU even when GPU generation is available. It cannot silently fall back to a fresh generation or a prerecorded final video. The server validates the prepared resources and returns an ordinary queued job; interrupted monitoring resumes that job instead of submitting another one. A completed performance uses `/api/jobs/{id}/replay` for its next take.

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
