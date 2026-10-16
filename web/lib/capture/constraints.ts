/** Capture constraints: no voice processing (absolute loudness must survive), locked 24 fps video. */

export const AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
  sampleRate: { ideal: 48000 },
};

export function videoConstraints(facing: "user" | "environment"): MediaTrackConstraints {
  return {
    facingMode: facing,
    frameRate: { min: 24, ideal: 24, max: 24 },
    width: { ideal: 1080 },
    height: { ideal: 1920 },
  };
}

export interface VerifyResult {
  ok: boolean;
  mismatches: string[];
  settings: MediaTrackSettings;
}

/** Re-apply constraints (some browsers ignore them in getUserMedia) and report what the device actually gave us. */
export async function applyAndVerify(
  track: Pick<MediaStreamTrack, "kind" | "applyConstraints" | "getSettings">,
  wanted: MediaTrackConstraints,
): Promise<VerifyResult> {
  const mismatches: string[] = [];
  try {
    await track.applyConstraints(wanted);
  } catch (err) {
    mismatches.push(`applyConstraints rejected: ${err instanceof Error ? err.message : String(err)}`);
  }
  const settings = track.getSettings();
  if (track.kind === "audio") {
    for (const key of ["echoCancellation", "noiseSuppression", "autoGainControl"] as const) {
      if (settings[key] !== false) mismatches.push(key);
    }
  } else if (track.kind === "video") {
    const fps = settings.frameRate;
    if (typeof fps !== "number" || fps < 23.5 || fps > 24.5) mismatches.push("frameRate");
  }
  return { ok: mismatches.length === 0, mismatches, settings };
}
